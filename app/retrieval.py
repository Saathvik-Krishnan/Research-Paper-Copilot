

from typing import List, Dict, Any, Tuple, Optional
import re
import math

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from agno.knowledge.embedder.google import GeminiEmbedder
from .config import settings


# -------------------------
# Qdrant client
# -------------------------

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


# -------------------------
# Text + filtering helpers
# -------------------------

_word_re = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _word_re.findall((text or "").lower())


def _get_text_from_payload(payload: Dict[str, Any]) -> str:
    # Agno/Qdrant payload key variations
    return (
        payload.get("text")
        or payload.get("content")
        or payload.get("chunk")
        or payload.get("document")
        or ""
    )


def _get_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("meta_data") or {}


def _looks_like_references(text: str) -> bool:
    """
    Aggressive filter to avoid reference sections dominating retrieval.
    This is intentionally conservative: we prefer dropping obvious refs.
    """
    t = (text or "").lower().strip()
    if not t:
        return True

    # Hard markers
    if "references" in t[:200] or "bibliography" in t[:200]:
        return True

    # Lots of citation patterns
    bracket_cites = len(re.findall(r"\[\d{1,3}\]", t))
    year_cites = len(re.findall(r"\((19|20)\d{2}\)", t))
    doi = 1 if "doi:" in t or "doi.org" in t else 0
    etal = t.count("et al")

    # "pp. 123-130" patterns are common in references
    pp = len(re.findall(r"\bpp\.\s*\d+", t))

    # Many author-like separators / journal-like formatting
    commas = t.count(",")
    semicolons = t.count(";")

    score = bracket_cites + year_cites + (3 * doi) + etal + pp
    # If it smells like a reference dump, drop it
    if score >= 10:
        return True

    # Very low “sentence-ness” (too many commas/semicolons)
    # while being short-ish often means bibliography line
    if len(t) < 600 and (commas + semicolons) > 25:
        return True

    return False


def _clean_text_for_embedding(text: str, max_chars: int = 900) -> str:
    """
    Keep embedding calls cheap + stable.
    """
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# -------------------------
# Embedding helpers
# -------------------------

def _embed_query(embedder: GeminiEmbedder, text: str) -> List[float]:
    """
    GeminiEmbedder method names vary by agno version.
    """
    if hasattr(embedder, "get_embedding"):
        return list(embedder.get_embedding(text))
    if hasattr(embedder, "embed_text"):
        return list(embedder.embed_text(text))
    if hasattr(embedder, "embed_query"):
        return list(embedder.embed_query(text))
    if callable(embedder):
        return list(embedder(text))
    raise AttributeError(
        f"Could not find embedding method on GeminiEmbedder. Available: {dir(embedder)}"
    )


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0:
        return 0.0
    return dot / denom


# -------------------------
# Qdrant API compatibility
# -------------------------

def _qdrant_vector_search(client: QdrantClient, collection: str, q_vec: List[float], top_k: int) -> List[Any]:
    # Newer clients use query_points
    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=collection,
            query=q_vec,
            limit=top_k,
            with_payload=True,
        )
        return resp.points

    # Older clients use search
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection,
            query_vector=q_vec,
            limit=top_k,
            with_payload=True,
        )

    raise AttributeError(f"Unsupported QdrantClient API. Available: {dir(client)}")


def _qdrant_scroll_all(client: QdrantClient, collection: str, max_docs: int = 500, batch: int = 100) -> List[Any]:
    """
    Pull chunks so BM25 can rank.
    Uses 'next_page_offset' pattern (works better across qdrant-client versions).
    """
    all_points: List[Any] = []
    next_offset = None

    while len(all_points) < max_docs:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=min(batch, max_docs - len(all_points)),
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        all_points.extend(points)

    return all_points


def _normalize_scores(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmin = min(vals)
    vmax = max(vals)
    if vmax == vmin:
        return [1.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


# -------------------------
# Dedup + Diversity
# -------------------------

def _fingerprint(text: str) -> str:
    """
    Very cheap dedup fingerprint: first 30 tokens.
    Good enough to remove near-identical chunks.
    """
    toks = _tokenize(text)[:30]
    return " ".join(toks)


def _apply_dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        fp = _fingerprint(it.get("text", ""))
        if not fp or fp in seen:
            continue
        seen.add(fp)
        out.append(it)
    return out


def _mmr_select(
    embedder: GeminiEmbedder,
    query: str,
    candidates: List[Dict[str, Any]],
    k: int,
    lambda_mult: float = 0.72,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance:
    pick chunks that are relevant to query but diverse among themselves.
    """
    if not candidates:
        return []

    q_vec = _embed_query(embedder, query)

    # Pre-embed candidate texts (small pool only)
    cand_vecs: List[List[float]] = []
    for c in candidates:
        txt = _clean_text_for_embedding(c.get("text", ""))
        cand_vecs.append(_embed_query(embedder, txt))

    selected_idx: List[int] = []
    remaining = list(range(len(candidates)))

    # Precompute similarity to query
    sim_to_q = [_cosine(q_vec, v) for v in cand_vecs]

    while remaining and len(selected_idx) < k:
        best_i = None
        best_score = -1e9

        for i in remaining:
            if not selected_idx:
                mmr = sim_to_q[i]
            else:
                max_sim_to_selected = max(_cosine(cand_vecs[i], cand_vecs[j]) for j in selected_idx)
                mmr = lambda_mult * sim_to_q[i] - (1 - lambda_mult) * max_sim_to_selected

            # Tiny tie-breaker: keep your hybrid_score influence
            mmr += float(candidates[i].get("hybrid_score", 0.0)) * 0.01

            if mmr > best_score:
                best_score = mmr
                best_i = i

        if best_i is None:
            break

        selected_idx.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected_idx]


# -------------------------
# Main retrieval (hybrid + rerank)
# -------------------------

def retrieve_chunks(collection: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    "Pro RAG" retrieval:
      - Hybrid: vectors + BM25
      - Strong reference filtering
      - Dedup
      - MMR reranking for relevance + diversity

    Returns top_k chunks with citation-friendly metadata.
    """
    client = get_qdrant_client()
    embedder = GeminiEmbedder(api_key=settings.GOOGLE_API_KEY)

    # -------------------------
    # Vector search (pull more then rerank)
    # -------------------------
    q_vec = _embed_query(embedder, query)
    top_k_vec = max(top_k * 6, 20)
    vec_points = _qdrant_vector_search(client, collection, q_vec, top_k=top_k_vec)

    # -------------------------
    # BM25 search over scrolled points
    # -------------------------
    corpus_points = _qdrant_scroll_all(client, collection, max_docs=500)

    docs: List[str] = []
    doc_ids: List[str] = []
    doc_payloads: List[Dict[str, Any]] = []

    for p in corpus_points:
        payload = getattr(p, "payload", None) or {}
        text = _get_text_from_payload(payload)
        if not (text or "").strip():
            continue
        if _looks_like_references(text):
            continue

        docs.append(text)
        doc_ids.append(str(getattr(p, "id", "")))
        doc_payloads.append(payload)

    bm25_candidates: List[Tuple[str, float, str, Dict[str, Any]]] = []
    if docs:
        tokenized_docs = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized_docs)
        q_tokens = _tokenize(query)
        bm25_scores = bm25.get_scores(q_tokens)

        top_k_bm25 = max(top_k * 6, 20)
        top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k_bm25]

        for i in top_idx:
            bm25_candidates.append((doc_ids[i], float(bm25_scores[i]), docs[i], doc_payloads[i]))

    # -------------------------
    # Merge by id
    # -------------------------
    merged: Dict[str, Dict[str, Any]] = {}

    # vector results
    for p in vec_points:
        pid = str(getattr(p, "id", ""))
        payload = getattr(p, "payload", None) or {}
        text = _get_text_from_payload(payload)

        if not (text or "").strip():
            continue
        if _looks_like_references(text):
            continue

        merged[pid] = {
            "id": pid,
            "text": text,
            "payload": payload,
            "vec_score": float(getattr(p, "score", 0.0) or 0.0),
            "bm25_score": 0.0,
        }

    # bm25 results
    for pid, score, text, payload in bm25_candidates:
        if not (text or "").strip():
            continue
        if _looks_like_references(text):
            continue

        if pid not in merged:
            merged[pid] = {
                "id": pid,
                "text": text,
                "payload": payload,
                "vec_score": 0.0,
                "bm25_score": score,
            }
        else:
            merged[pid]["bm25_score"] = score

    merged_list = list(merged.values())

    # If nothing, return empty
    if not merged_list:
        return []

    # -------------------------
    # Normalize + hybrid score
    # -------------------------
    vec_scores = [x["vec_score"] for x in merged_list]
    bm25_scores_only = [x["bm25_score"] for x in merged_list]
    vec_n = _normalize_scores(vec_scores)
    bm25_n = _normalize_scores(bm25_scores_only)

    alpha = 0.65  # vector weight
    for i, item in enumerate(merged_list):
        item["hybrid_score"] = alpha * vec_n[i] + (1 - alpha) * bm25_n[i]

        # small heuristic: earlier pages slightly better (abstract/method)
        meta = _get_meta(item["payload"])
        page = meta.get("page")
        if isinstance(page, int):
            item["hybrid_score"] += max(0.0, (6 - page)) * 0.01  # boost pages 1..5 slightly

        # small heuristic: prefer chunks that look like narrative (not list of authors)
        txt = (item.get("text") or "")
        if len(_tokenize(txt)) > 40 and txt.count("\n") < 25:
            item["hybrid_score"] += 0.01

    merged_list.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # -------------------------
    # Dedup + MMR rerank (quality upgrade)
    # -------------------------
    merged_list = _apply_dedup(merged_list)

    # Only rerank a small pool for speed
    pool_size = min(len(merged_list), max(top_k * 8, 30))
    pool = merged_list[:pool_size]

    # MMR selection picks top_k diverse + relevant
    reranked = _mmr_select(embedder, query, pool, k=top_k, lambda_mult=0.72)

    # -------------------------
    # Output format for citations
    # -------------------------
    out: List[Dict[str, Any]] = []
    for item in reranked:
        payload = item.get("payload") or {}
        meta = _get_meta(payload)

        out.append({
            "score": float(item.get("hybrid_score", 0.0)),
            "text": (item.get("text") or "")[:1200],
            "payload": payload,
            "meta": {
                "page": meta.get("page"),
                "chunk": meta.get("chunk"),
                "chunk_size": meta.get("chunk_size"),
                "vec_score": float(item.get("vec_score", 0.0)),
                "bm25_score": float(item.get("bm25_score", 0.0)),
            }
        })

    return out
