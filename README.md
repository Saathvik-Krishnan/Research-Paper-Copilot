# Research Paper Copilot

Research Paper Copilot is a FastAPI + Streamlit RAG application for ingesting research PDFs, indexing them in Qdrant, and answering grounded questions using Gemini via Agno.

It supports:
- Single-paper QA with citations
- Multi-paper comparison with structured synthesis
- Retrieval diagnostics (`/chunks`)
- Evaluation runs with retrieval and answer-quality proxies (`/eval/run`)

## Tech Stack

- Backend: FastAPI
- Frontend: Streamlit
- LLM/Agent layer: Agno + Gemini (`gemini-2.0-flash`)
- Embeddings: Gemini embeddings
- Vector DB: Qdrant
- Retrieval: Hybrid vector + BM25 + MMR reranking

## Features

- PDF ingestion into paper-specific collections (`paper_<paper_id>`)
- Hybrid retrieval pipeline:
  - Vector search in Qdrant
  - BM25 keyword scoring
  - Score fusion + normalization
  - Deduplication and MMR reranking for diversity
- Noise filtering to reduce poor context:
  - Reference/bibliography filtering
  - Header/footer boilerplate filtering
  - Short/junk chunk removal
- Grounded prompting with "I don't know" behavior for weak evidence
- Multi-paper extraction-first workflow to reduce hallucinations
- Evaluation logging to JSONL for iteration and analysis

## Project Structure

```text
app/
  main.py           # FastAPI app + endpoints
  retrieval.py      # Hybrid retrieval (vector + BM25 + MMR)
  agno_stack.py     # Agno team, Gemini model, ingestion and ask helpers
  config.py         # Settings from .env
  copilot_app.py    # Streamlit UI

data/
  uploads/          # Uploaded PDFs
  memory/           # Per-paper chat memory in JSONL
  logs/             # Eval logs (eval_runs.jsonl)

requirements.txt
docker-compose.yml  # Qdrant service
.env.example
```

## Prerequisites

- Python 3.10+
- Google API key with Gemini access
- Qdrant Cloud cluster URL + API key (or local Docker Qdrant)

## Setup

1. Clone and enter the repo:

```bash
git clone https://github.com/Saathvik-Krishnan/Research-Paper-Copilot.git
cd Research-Paper-Copilot
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create `.env` from `.env.example` and set keys:

```env
GOOGLE_API_KEY=your_real_google_api_key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your_qdrant_api_key
```

5. Optional (only if using local Qdrant instead of Qdrant Cloud):

```bash
docker compose up -d
```

## Run the App

Start backend API:

```bash
uvicorn app.main:app --reload
```

Start Streamlit UI (new terminal):

```bash
streamlit run app/copilot_app.py
```

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Streamlit UI: shown in terminal output (usually `http://localhost:8501`)

## Deploy Backend on Render (Free)

1. Create a **Web Service** in Render from this GitHub repo.
2. Use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Branch: `main`
   - Runtime: Python 3
3. Add Render environment variables:
   - `GOOGLE_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
4. Deploy and verify:
   - `https://<your-render-url>/health`
   - `https://<your-render-url>/docs`

Note: root URL (`/`) may return `{"detail":"Not Found"}` if no root route is defined. Use `/health` or `/docs`.

## Deploy Frontend on Streamlit Community Cloud

1. Go to Streamlit Community Cloud and click **Deploy a public app from GitHub**.
2. Select:
   - Repo: `Saathvik-Krishnan/Research-Paper-Copilot`
   - Branch: `main`
   - Main file path: `app/copilot_app.py`
3. In **Manage app -> Settings -> Secrets**, add:

```toml
API_BASE = "https://<your-render-url>"
```

4. Save and reboot the app.
5. Confirm sidebar shows `API_BASE` as your Render URL (not `http://127.0.0.1:8000`).

## API Endpoints

### `GET /health`
Returns service health.

### `POST /ingest`
Upload a PDF and create a new `paper_id`.

### `POST /ask`
Ask a question about one paper.

Request body:

```json
{
  "paper_id": "<paper_id>",
  "question": "What is the main contribution?",
  "top_k": 5
}
```

### `POST /ask_multi`
Compare multiple papers using per-paper extraction + synthesis.

Request body:

```json
{
  "paper_ids": ["<paper_id_1>", "<paper_id_2>"],
  "question": "Compare results and limitations",
  "top_k_per_paper": 6,
  "max_sources_per_paper": 5
}
```

### `GET /chunks/{paper_id}`
Inspect stored chunks and metadata for debugging retrieval.

### `POST /eval/run`
Run retrieval/answer evaluation and append metrics to `data/logs/eval_runs.jsonl`.

Request body:

```json
{
  "paper_id": "<paper_id>",
  "question": "What dataset was used?",
  "top_k": 8,
  "ground_truth": "Optional reference answer",
  "run_ragas": false
}
```

## Typical Workflow

1. Upload PDF via `/ingest` (or Streamlit sidebar).
2. Save returned `paper_id`.
3. Ask questions via `/ask`.
4. Compare papers via `/ask_multi`.
5. Use `/chunks` and `/eval/run` to debug and improve retrieval quality.

## Notes

- Each paper is isolated in its own Qdrant collection (`paper_<paper_id>`).
- Local memory is stored per paper in `data/memory/<paper_id>_chat.jsonl`.
- Uploaded files and eval logs are kept under `data/` for traceability.
- Render free tier sleeps when idle; first request can be slow (cold start).
- Streamlit and Render local disk are ephemeral; use external storage if you need persistent uploads/memory/logs.

## Future Improvements

- Add tests for retrieval filters and endpoint contracts
- Add auth/rate limiting for production deployment
- Add CI checks (lint/format/test)
- Add optional persistent object storage for PDFs

## License

Add your preferred license (MIT/Apache-2.0/etc.) in a `LICENSE` file.
