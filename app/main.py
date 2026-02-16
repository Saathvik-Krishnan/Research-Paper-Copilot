import os
import re
import uuid
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient

# Local project modules:
# - ingest_pdf(): splits a PDF into chunks, creates embeddings, and stores them in Qdrant
# - ask_team(): LLM call (Gemini via your Agno stack) that answers using provided context
# - retrieve_chunks(): hybrid retrieval (vector + keyword/BM25) from Qdrant
# - settings: environment config (API keys, Qdrant URL)
from .agno_stack import ingest_pdf, ask_team
from .retrieval import retrieve_chunks
from .config import settings


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------
# This backend exposes REST endpoints used by your UI (Streamlit) or Swagger.
# Core responsibilities:
# 1) Ingest PDFs into a vector store (Qdrant)
# 2) Retrieve relevant chunks for a question
# 3) Ask the LLM to answer using ONLY retrieved sources (grounded RAG)
# 4) Provide evaluation metrics + logs for debugging and improvement
app = FastAPI(title="Research Paper Copilot (Agno)")


# -------------------------------------------------------------------
# Local storage locations
# -------------------------------------------------------------------
# UPLOAD_DIR: where raw PDFs are saved after upload (for traceability / re-ingest)
# LOG_DIR: where evaluation run logs are stored in JSONL format (easy to analyze later)
UPLOAD_DIR = "data/uploads"
LOG_DIR = "data/logs"
EVAL_LOG_PATH = os.path.join(LOG_DIR, "eval_runs.jsonl")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Request models (input validation + Swagger schema)
# -------------------------------------------------------------------
class AskRequest(BaseModel):
    # UUID of an ingested paper (maps to Qdrant collection "paper_<paper_id>")
    paper_id: str
    # Natural language question asked by the user
    question: str
    # Retrieval depth: how many chunks to fetch from Qdrant
    top_k: int = 5


class AskMultiRequest(BaseModel):
    # List of paper UUIDs to compare
    paper_ids: List[str]
    # Question to apply across all selected papers (comparison prompt)
    question: str
    # Retrieval depth per paper
    top_k_per_paper: int = 6
    # Number of sources per paper to include in the LLM prompt (controls prompt size)
    max_sources_per_paper: int = 5


class EvalRunRequest(BaseModel):
    # Paper UUID to evaluate on
    paper_id: str
    # Question for evaluation run
    question: str
    # Retrieval depth for evaluation run
    top_k: int = 8

    # Optional: if you already have a predicted answer, pass it; otherwise we generate it
    predicted_answer: Optional[str] = None

    # Optional: "gold" answer to compute simple overlap metrics
    ground_truth: Optional[str] = None

    # Optional flag kept for future experimentation (you can keep this False in your project)
    run_ragas: bool = False


# -------------------------------------------------------------------
# Text cleanup + chunk quality filters
# -------------------------------------------------------------------
def _clean_text(t: str) -> str:
    """
    Normalizes text so retrieval + dedupe behave better.
    - trims leading/trailing spaces
    - collapses repeated whitespace/newlines into single spaces
    """
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _letters_only_lower(t: str) -> str:
    """
    Converts text to only a-z letters in lowercase.
    Useful to detect OCR-broken words like:
      "R EF E R E N C E S" -> "references"
    """
    return re.sub(r"[^a-z]+", "", (t or "").lower())


def _looks_like_references(text: str) -> bool:
    """
    Heuristic to detect reference/bibliography sections.
    Why we remove them:
    - References are extremely "keyword-dense"
    - They often dominate retrieval but do NOT answer the question
    - They increase hallucination risk (model cites unrelated paper titles/authors)
    """
    t = (text or "").lower()
    compact = _letters_only_lower(text)

    # Direct detection (normal text) + OCR-broken detection (letters-only)
    if "references" in t or "bibliography" in t:
        return True
    if "references" in compact or "bibliography" in compact:
        return True

    # Secondary heuristic: heavy citation patterns usually indicate references
    bracket_cites = len(re.findall(r"\[\d{1,3}\]", t))
    year_cites = len(re.findall(r"\b(19\d{2}|20\d{2})\b", t))
    etal = t.count("et al")

    if (bracket_cites + year_cites + etal) >= 12:
        return True

    return False


def _looks_like_header_footer_noise(text: str) -> bool:
    """
    Detects repetitive journal header/footer boilerplate.
    Why we remove it:
    - Same lines repeat on every page
    - It pollutes retrieval and increases duplicates
    """
    t = (text or "").lower()
    header_noise = [
        "international research journal of engineering and technology",
        "www.irjet.net",
        "e-issn",
        "p-issn",
        "impact factor",
        "iso 9001",
    ]
    hits = sum(1 for h in header_noise if h in t)
    return hits >= 2


def _looks_like_junk_chunk(text: str) -> bool:
    """
    Central chunk filter for retrieval quality.
    A chunk is treated as junk if:
    - it is too short (usually not meaningful)
    - it looks like references
    - it looks like repeated header/footer noise
    """
    t = (text or "").strip()
    if len(t) < 120:
        return True
    if _looks_like_references(t):
        return True
    if _looks_like_header_footer_noise(t):
        return True
    return False


def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes near-duplicate chunks based on normalized text prefix.
    Why:
    - PDFs can generate duplicates due to OCR, page overlap, or repeated headers
    - Deduping improves evidence diversity and prompt efficiency
    """
    seen = set()
    out = []
    for c in chunks:
        txt = _clean_text(c.get("text", ""))
        key = txt[:500].lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _normalize_paper_id(pid: str) -> str:
    """
    Supports both formats:
      - "paper_<uuid>" (collection name style)
      - "<uuid>"       (API style)
    Always returns "<uuid>".
    """
    pid = (pid or "").strip()
    if pid.startswith("paper_"):
        return pid.replace("paper_", "", 1)
    return pid


def _get_meta(c: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unifies metadata from different retrieval payload formats.
    Some retrieval pipelines store metadata in:
      c["meta"]
    Others store metadata in:
      c["payload"]["meta_data"]

    We merge both so downstream code can read meta consistently.
    """
    meta = c.get("meta") or {}
    payload = c.get("payload") or {}
    payload_meta = payload.get("meta_data") or {}

    merged = dict(payload_meta)
    merged.update(meta)
    return merged


def _trim_citations(chunks: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """
    Produces a compact citations list for the UI response.
    Includes:
    - final hybrid score (ranking score)
    - vector score + BM25 score (useful for debugging hybrid retrieval)
    - page/chunk identifiers (traceability)
    - small text snippet (preview evidence)
    """
    out = []
    for c in (chunks[:limit] if chunks else []):
        meta = _get_meta(c)
        payload = c.get("payload") or {}
        out.append({
            "score": c.get("score"),
            "vec_score": meta.get("vec_score"),
            "bm25_score": meta.get("bm25_score"),
            "page": meta.get("page"),
            "chunk": meta.get("chunk"),
            "chunk_size": meta.get("chunk_size"),
            "name": payload.get("name") or c.get("name"),
            "text": (c.get("text") or "")[:500],
        })
    return out


# -------------------------------------------------------------------
# Lightweight evaluation helpers (no external judge needed)
# -------------------------------------------------------------------
def _safe_float(x: Any) -> Optional[float]:
    """
    Converts values to float safely.
    Returns None if conversion fails, avoiding crashes during metric calculation.
    """
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _avg(nums: List[Optional[float]]) -> Optional[float]:
    """
    Average of floats ignoring None.
    Returns None if list has no valid values.
    """
    vals = [x for x in nums if x is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _token_set(text: str) -> set:
    """
    Simple tokenizer:
    - lowercases
    - removes punctuation
    - keeps tokens length > 2
    Used for overlap metrics (fast and explainable).
    """
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    toks = [t for t in text.split() if len(t) > 2]
    return set(toks)


def _overlap_f1(pred: str, gt: str) -> Dict[str, float]:
    """
    Computes token overlap precision/recall/F1 between predicted answer and ground truth.
    This is a *proxy* metric:
    - fast
    - transparent
    - does not require another LLM judge
    """
    P = _token_set(pred)
    G = _token_set(gt)
    if not P or not G:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    inter = len(P & G)
    precision = inter / max(len(P), 1)
    recall = inter / max(len(G), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """
    Appends a dictionary as a JSON line to a log file.
    JSONL format is easy to parse later for analytics dashboards or debugging.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------
# Multi-paper prompt helpers (structured extraction then synthesis)
# -------------------------------------------------------------------
def _build_context_block_for_paper(paper_id: str, chunks: List[Dict[str, Any]], max_sources: int) -> str:
    """
    Builds a single string block of evidence for ONE paper.
    Adds stable citations like:
      [<paper_id> | Source 1]
    so the model can cite sources clearly and consistently.
    """
    parts = []
    for i, c in enumerate(chunks[:max_sources]):
        snippet = (c.get("text") or "")[:900]
        meta = _get_meta(c)
        page = meta.get("page")
        chunk_no = meta.get("chunk")
        score = c.get("score")
        parts.append(
            f"[{paper_id} | Source {i+1}] (score={score}, page={page}, chunk={chunk_no})\n{snippet}"
        )
    return "\n\n".join(parts)


def _pick_best_results_chunk(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Heuristic: if a chunk looks like it contains results/metrics, move it to the front.
    Reason:
    - metrics (accuracy/EER/FAR/FRR/etc.) are high-value facts for comparison
    - it reduces the chance the model ignores numbers or hallucinates them
    """
    if not chunks:
        return None

    keywords = [
        "result", "results", "experimental", "evaluation", "accuracy", "recognition rate",
        "eer", "far", "frr", "fmr", "fnmr", "table", "%", "dataset", "roc"
    ]

    best = None
    best_hits = -1
    for c in chunks:
        t = (c.get("text") or "").lower()
        hits = sum(1 for k in keywords if k in t)
        if hits > best_hits:
            best_hits = hits
            best = c

    return best if best_hits >= 2 else None


def _ask_single_paper_extraction(
    paper_id: str,
    user_question: str,
    chunks: List[Dict[str, Any]],
    max_sources: int
) -> str:
    """
    Runs a strict "per-paper" extraction step.
    Key goal:
    - force the model to talk about ONE paper only
    - produce structured output so the synthesis step is clean
    - keep citations in the form [<paper_id> | Source N]
    """
    if not chunks:
        return (
            "Contributions:\n- Not stated.\n\n"
            "Results/Metrics:\n- Not stated.\n\n"
            "Dataset/Setup:\n- Not stated.\n\n"
            "Citations:\n- None."
        )

    context_block = _build_context_block_for_paper(paper_id, chunks, max_sources=max_sources)

    prompt = (
        "You are a research-paper copilot.\n"
        "You MUST use ONLY the sources provided below.\n"
        "Do NOT mention other papers.\n"
        "Do NOT say you need more context.\n"
        "If something isn't in the sources, write: Not stated.\n"
        "CITATIONS RULE: Only cite using [<paper_id> | Source N]. Never cite [1], [2], authors, or external references.\n\n"
        f"{context_block}\n\n"
        "Task: Extract what THIS paper claims.\n"
        f"User question (for focus): {user_question}\n\n"
        "Return in EXACTLY this format:\n"
        "Contributions:\n"
        "- (1-4 bullets)\n\n"
        "Results/Metrics:\n"
        "- Include any numbers (accuracy, EER, FAR/FRR, FMR/FNMR, etc.)\n"
        "- If no numbers: Not stated.\n\n"
        "Dataset/Setup:\n"
        "- population size / samples / train-test / hardware setup if present\n"
        "- If missing: Not stated.\n\n"
        "Citations:\n"
        "- List only citations like: [<paper_id> | Source 1]\n"
    )

    result = ask_team(prompt, f"paper_{paper_id}")
    return (result.get("answer") or "").strip()


# -------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    """Simple health check for uptime monitoring and quick local debugging."""
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload endpoint for PDFs.

    What happens:
    1) We generate a unique paper_id (UUID)
    2) Save the uploaded PDF to disk for traceability
    3) Use collection name "paper_<paper_id>" in Qdrant
    4) ingest_pdf() handles chunking + embeddings + Qdrant upsert
    """
    paper_id = str(uuid.uuid4())
    safe_name = file.filename or "paper.pdf"
    path = os.path.join(UPLOAD_DIR, f"{paper_id}__{safe_name}")

    with open(path, "wb") as f:
        f.write(await file.read())

    collection = f"paper_{paper_id}"
    ingest_pdf(path, collection)

    return {"paper_id": paper_id}


@app.post("/ask")
def ask(req: AskRequest):
    """
    Single-paper question answering (classic RAG).

    Pipeline:
    1) Retrieve candidate chunks from Qdrant (hybrid: BM25 + embeddings)
    2) Clean / filter noisy chunks and dedupe
    3) Confidence gate: if top score is too low, respond "I don't know"
    4) Construct a grounded prompt containing top evidence chunks
    5) ask_team() generates the final answer with citations
    """
    if not req.paper_id:
        raise HTTPException(status_code=400, detail="paper_id is required")
    if not req.question:
        raise HTTPException(status_code=400, detail="question is required")
    if req.top_k <= 0 or req.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    paper_id = _normalize_paper_id(req.paper_id)
    collection_name = f"paper_{paper_id}"

    # Retrieve top-k candidate evidence chunks from the vector store
    raw_chunks = retrieve_chunks(collection_name, req.question, top_k=req.top_k)

    # Clean chunk text and remove noisy chunks (references, headers, too-short)
    cleaned: List[Dict[str, Any]] = []
    for c in raw_chunks:
        txt = _clean_text(c.get("text", ""))
        c["text"] = txt
        if not _looks_like_junk_chunk(txt):
            cleaned.append(c)

    # Deduplicate near-identical chunks to avoid repetitive context
    cleaned = _dedupe_chunks(cleaned)

    # If filtering was too aggressive, fall back to raw chunks
    chunks = cleaned if cleaned else raw_chunks

    # If nothing retrieved, return a safe response
    if not chunks:
        return {
            "answer": "I don't know. I couldn't retrieve any relevant sources from the uploaded paper.",
            "citations": [],
        }

    # Confidence gate: prevent answering when retrieval evidence is weak
    best_score = None
    try:
        best_score = float(chunks[0].get("score")) if chunks[0].get("score") is not None else None
    except Exception:
        best_score = None

    # Threshold tuned empirically; low score usually means irrelevant evidence
    if best_score is not None and best_score < 0.12:
        return {
            "answer": "I don't know. The retrieved sources don't contain enough evidence to answer that question.",
            "citations": _trim_citations(chunks, req.top_k),
        }

    # Build a short evidence block for the LLM prompt (limit sources to control tokens)
    max_sources = min(req.top_k, 5)
    context_parts: List[str] = []
    for i, c in enumerate(chunks[:max_sources]):
        snippet = (c.get("text") or "")[:900]
        score = c.get("score")
        context_parts.append(f"[Source {i+1}] (score={score})\n{snippet}")

    context_block = "\n\n".join(context_parts)

    # Grounded prompt: model must answer ONLY from these sources (or say "I don't know")
    grounded_question = (
        "You are a research-paper copilot.\n"
        "Use ONLY the sources below to answer.\n"
        "If the answer is not explicitly supported by the sources, say: \"I don't know.\" (do not guess)\n\n"
        f"{context_block}\n\n"
        f"Question: {req.question}\n\n"
        "Return:\n"
        "1) A clear answer (2-6 sentences)\n"
        "2) Citations at the end like: [Source 1], [Source 3]\n"
    )

    # LLM call (Gemini via Agno stack)
    result = ask_team(grounded_question, collection_name)

    return {
        "answer": result.get("answer"),
        "citations": _trim_citations(chunks, req.top_k),
    }


@app.post("/ask_multi")
def ask_multi(req: AskMultiRequest):
    """
    Multi-paper comparison endpoint.

    Strategy (to reduce hallucinations):
    1) Retrieve evidence per paper separately
    2) Run a strict extraction per paper (structured: contributions/metrics/dataset)
    3) Provide those structured extractions + top evidence to a synthesis prompt
    4) Return a final comparison + per-paper outputs
    """
    if not req.paper_ids or len(req.paper_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 paper_ids")
    if not req.question:
        raise HTTPException(status_code=400, detail="question is required")
    if req.top_k_per_paper <= 0 or req.top_k_per_paper > 15:
        raise HTTPException(status_code=400, detail="top_k_per_paper must be between 1 and 15")
    if req.max_sources_per_paper <= 0 or req.max_sources_per_paper > 8:
        raise HTTPException(status_code=400, detail="max_sources_per_paper must be between 1 and 8")

    paper_ids = [_normalize_paper_id(x) for x in req.paper_ids]

    per_paper_results: List[Dict[str, Any]] = []
    evidence_blocks: List[str] = []

    for pid in paper_ids:
        collection_name = f"paper_{pid}"

        # Retrieve evidence for this paper only
        raw = retrieve_chunks(collection_name, req.question, top_k=req.top_k_per_paper)

        # Clean + filter noisy chunks
        cleaned: List[Dict[str, Any]] = []
        for c in raw:
            txt = _clean_text(c.get("text", ""))
            c["text"] = txt
            if not _looks_like_junk_chunk(txt):
                cleaned.append(c)

        cleaned = _dedupe_chunks(cleaned)
        chunks = cleaned if cleaned else raw

        # Optional: move likely "results" chunk up to improve metric extraction
        results_chunk = _pick_best_results_chunk(chunks)
        if results_chunk is not None:
            chunks = [results_chunk] + [c for c in chunks if c is not results_chunk]

        # Strict per-paper extraction (prevents cross-paper contamination)
        mini_answer = _ask_single_paper_extraction(
            paper_id=pid,
            user_question=req.question,
            chunks=chunks,
            max_sources=min(req.max_sources_per_paper, len(chunks)),
        )

        per_paper_results.append({
            "paper_id": pid,
            "mini_answer": mini_answer,
            "citations": _trim_citations(chunks, req.top_k_per_paper),
        })

        # Keep a small evidence block for the synthesis stage
        if chunks:
            evidence_blocks.append(
                _build_context_block_for_paper(pid, chunks, max_sources=min(5, len(chunks)))
            )

    # If none of the papers returned evidence, we cannot do a grounded comparison
    any_evidence = any(bool(x.get("citations")) for x in per_paper_results)
    if not any_evidence:
        return {
            "question": req.question,
            "final_answer": "I don't know. I couldn't retrieve relevant sources from the selected papers.",
            "per_paper": per_paper_results,
        }

    # Synthesis inputs: structured extractions + top evidence excerpts
    mini_answers_text = "\n\n".join(
        [f"Paper {x['paper_id']} extraction:\n{x['mini_answer']}" for x in per_paper_results]
    )
    evidence_text = "\n\n---\n\n".join(evidence_blocks)

    # Synthesis prompt: compare papers without inventing missing metrics
    synth_prompt = (
        "You are a multi-paper research copilot.\n"
        "Use ONLY the per-paper extractions and the evidence excerpts below.\n"
        "Do NOT add external citations, author-year refs, or bracket refs like [1].\n"
        "If a metric is missing for a paper, write: Not stated.\n"
        "Only claim contradictions if the excerpts explicitly conflict.\n\n"
        f"User request: {req.question}\n\n"
        "Per-paper extractions:\n"
        f"{mini_answers_text}\n\n"
        "Top evidence excerpts:\n"
        f"{evidence_text}\n\n"
        "Return in EXACTLY this format:\n"
        "Paper Summaries:\n"
        "- Paper <paper_id>: 2-4 lines\n\n"
        "Comparison:\n"
        "- Similarities:\n"
        "- Differences:\n"
        "- Results comparison (numbers only; otherwise 'Not stated'):\n"
        "- Contradictions (if any):\n\n"
        "Citations:\n"
        "- Only use [<paper_id> | Source N]\n"
    )

    synth = ask_team(synth_prompt, f"paper_{paper_ids[0]}")

    return {
        "question": req.question,
        "final_answer": (synth.get("answer") or "").strip(),
        "per_paper": per_paper_results,
    }


@app.get("/chunks/{paper_id}")
def list_chunks(paper_id: str, limit: int = 20, offset: int = 0):
    """
    Chunk inspection endpoint (debugging / transparency).

    Reads raw stored chunks from Qdrant so you can validate:
    - ingestion worked
    - metadata exists (page/chunk ids)
    - what text is actually retrievable
    """
    if not paper_id:
        raise HTTPException(status_code=400, detail="paper_id is required")
    if limit <= 0 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    paper_id = _normalize_paper_id(paper_id)
    collection = f"paper_{paper_id}"

    # Qdrant client uses the configured URL (local docker container typically)
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    try:
        points, _next = client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read chunks from Qdrant. Is the collection '{collection}' present? Error: {e}",
        )

    out: List[Dict[str, Any]] = []
    for p in points:
        payload: Dict[str, Any] = p.payload or {}
        meta: Dict[str, Any] = payload.get("meta_data") or {}

        # Different ingestion pipelines may store the chunk text under different keys
        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("chunk")
            or payload.get("document")
            or ""
        )

        out.append({
            "id": str(getattr(p, "id", "")),
            "page": meta.get("page"),
            "chunk": meta.get("chunk"),
            "chunk_size": meta.get("chunk_size"),
            "vec_score": meta.get("vec_score"),
            "bm25_score": meta.get("bm25_score"),
            "text": (text or "")[:1500],
        })

    return {
        "paper_id": paper_id,
        "collection": collection,
        "limit": limit,
        "offset": offset,
        "count": len(out),
        "chunks": out,
    }


@app.post("/eval/run")
def eval_run(req: EvalRunRequest):
    """
    Evaluation endpoint (for improving retrieval + answer quality).

    Outputs:
    - retrieval_metrics: quick stats about returned scores (hybrid/vector/BM25)
    - grounded_token_ratio_proxy: overlap of answer tokens with context tokens
    - answer_metrics: token overlap F1 vs ground truth (if provided)
    - citations: evidence snippets used for the run
    - logs to JSONL for later analysis
    """
    if not req.paper_id:
        raise HTTPException(status_code=400, detail="paper_id is required")
    if not req.question:
        raise HTTPException(status_code=400, detail="question is required")
    if req.top_k <= 0 or req.top_k > 30:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 30")

    paper_id = _normalize_paper_id(req.paper_id)
    collection_name = f"paper_{paper_id}"

    t0 = time.time()

    # Retrieve evidence chunks for evaluation
    raw = retrieve_chunks(collection_name, req.question, top_k=req.top_k)

    # Clean/filter/dedupe to reflect your “real” retrieval quality
    cleaned: List[Dict[str, Any]] = []
    for c in raw:
        txt = _clean_text(c.get("text", ""))
        c["text"] = txt
        if not _looks_like_junk_chunk(txt):
            cleaned.append(c)

    cleaned = _dedupe_chunks(cleaned)
    chunks = cleaned if cleaned else raw

    # Compute retrieval score summaries
    hybrid_scores: List[Optional[float]] = []
    vec_scores: List[Optional[float]] = []
    bm25_scores: List[Optional[float]] = []

    for c in chunks:
        meta = _get_meta(c)
        hybrid_scores.append(_safe_float(c.get("score")))
        vec_scores.append(_safe_float(meta.get("vec_score")))
        bm25_scores.append(_safe_float(meta.get("bm25_score")))

    retrieval_metrics = {
        "k": req.top_k,
        "n_returned": len(chunks),
        "avg_hybrid_score": _avg(hybrid_scores),
        "avg_vec_score": _avg(vec_scores),
        "avg_bm25_score": _avg(bm25_scores),
        "top1_hybrid_score": hybrid_scores[0] if hybrid_scores else None,
        "top1_vec_score": vec_scores[0] if vec_scores else None,
        "top1_bm25_score": bm25_scores[0] if bm25_scores else None,
    }

    # Generate predicted answer if caller didn't provide one
    predicted = (req.predicted_answer or "").strip()
    if not predicted:
        max_sources = min(req.top_k, 5)

        context_parts: List[str] = []
        for i, c in enumerate(chunks[:max_sources]):
            snippet = (c.get("text") or "")[:900]
            score = c.get("score")
            context_parts.append(f"[Source {i+1}] (score={score})\n{snippet}")

        context_block = "\n\n".join(context_parts)

        prompt = (
            "You are a research-paper copilot.\n"
            "Use ONLY the sources below to answer.\n"
            "If the answer is not explicitly supported by the sources, say: \"I don't know.\" (do not guess)\n\n"
            f"{context_block}\n\n"
            f"Question: {req.question}\n\n"
            "Return:\n"
            "1) A clear answer (2-6 sentences)\n"
            "2) Citations at the end like: [Source 1], [Source 3]\n"
        )

        result = ask_team(prompt, collection_name)
        predicted = (result.get("answer") or "").strip()

    # Token overlap vs ground truth (only if provided)
    answer_metrics: Dict[str, Any] = {}
    if req.ground_truth:
        answer_metrics["gt_overlap"] = _overlap_f1(predicted, req.ground_truth)

    # Groundedness proxy: percent of answer tokens that appear in retrieved context
    ctx_text = " ".join([(c.get("text") or "") for c in chunks[: min(8, len(chunks))]])
    ctx_tokens = _token_set(ctx_text)
    ans_tokens = _token_set(predicted)

    grounded_ratio = 0.0
    if ans_tokens:
        grounded_ratio = round(len(ans_tokens & ctx_tokens) / max(len(ans_tokens), 1), 4)

    # Optional RAGAS block kept for future; typically disabled in your project
    ragas_scores = None
    ragas_error = None

    if req.run_ragas:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

            contexts = [(c.get("text") or "")[:1200] for c in chunks[: min(req.top_k, 8)]]

            data = {"question": [req.question], "answer": [predicted], "contexts": [contexts]}
            if req.ground_truth:
                data["ground_truth"] = [req.ground_truth]

            ds = Dataset.from_dict(data)

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            metrics = [faithfulness, answer_relevancy, context_precision]
            if req.ground_truth:
                metrics.append(context_recall)

            result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)
            ragas_scores = result.to_pandas().iloc[0].to_dict()
        except Exception as e:
            ragas_error = str(e)

    # Build final JSON response + persist to JSONL for later analysis
    out = {
        "paper_id": paper_id,
        "question": req.question,
        "predicted_answer": predicted,
        "ground_truth": req.ground_truth,
        "retrieval_metrics": retrieval_metrics,
        "grounded_token_ratio_proxy": grounded_ratio,
        "answer_metrics": answer_metrics,
        "ragas_scores": ragas_scores,
        "ragas_error": ragas_error,
        "citations": _trim_citations(chunks, req.top_k),
        "runtime_sec": round(time.time() - t0, 4),
        "ts": datetime.utcnow().isoformat() + "Z",
        "log_path": EVAL_LOG_PATH,
    }

    _append_jsonl(EVAL_LOG_PATH, out)
    return out
