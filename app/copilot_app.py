import json
import os
from datetime import datetime
from typing import Any, Dict, List

import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"


def get_api_base() -> str:
    env_api_base = os.getenv("API_BASE")
    if env_api_base:
        return env_api_base.rstrip("/")

    try:
        secret_api_base = st.secrets.get("API_BASE")
        if secret_api_base:
            return str(secret_api_base).rstrip("/")
    except Exception:
        pass

    return DEFAULT_API_BASE


API_BASE = get_api_base()
MEM_DIR = "data/memory"
os.makedirs(MEM_DIR, exist_ok=True)


def post_json(path: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_file(path: str, file_bytes: bytes, filename: str, timeout: int = 180) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    files = {"file": (filename, file_bytes, "application/pdf")}
    r = requests.post(url, files=files, timeout=timeout)
    r.raise_for_status()
    return r.json()


def mem_path(paper_id: str) -> str:
    return os.path.join(MEM_DIR, f"{paper_id}_chat.jsonl")


def append_memory(paper_id: str, role: str, content: str) -> None:
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "paper_id": paper_id,
        "role": role,
        "content": content,
    }
    with open(mem_path(paper_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_memory(paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    path = mem_path(paper_id)
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items[-limit:]


st.set_page_config(
    page_title="PaperSphere-AI: Agentic Multi-Paper RAG Pipeline with Hybrid BM25 + Vector Search and Metric-Aware Synthesis",
    layout="wide",
)

st.title("📄 PaperSphere-AI: Agentic Multi-Paper RAG Pipeline with Hybrid BM25 + Vector Search and Metric-Aware Synthesis")

with st.sidebar:
    st.subheader("API")
    st.write(f"API_BASE: `{API_BASE}`")
    st.caption("Set API_BASE env if needed. Example: API_BASE=http://127.0.0.1:8000")

    st.divider()
    st.subheader("Upload PDF")
    up = st.file_uploader("Upload a paper (PDF)", type=["pdf"])
    if up is not None:
        if st.button("Ingest PDF"):
            with st.spinner("Uploading + ingesting..."):
                res = post_file("/ingest", up.getvalue(), up.name)
            st.success("Ingested!")
            st.session_state["last_paper_id"] = res["paper_id"]
            st.code(res)

    st.divider()
    st.subheader("Paper IDs")
    last_pid = st.session_state.get("last_paper_id", "")
    st.text_input("Last paper_id", value=last_pid, key="paper_id_single")
    st.text_area(
        "Multiple paper_ids (one per line)",
        value=(last_pid + "\n").strip() if last_pid else "",
        key="paper_ids_multi_raw",
        height=120,
    )

    st.divider()
    st.subheader("Memory")
    st.caption("Local memory stored in data/memory/<paper_id>_chat.jsonl")
    if st.button("Clear Streamlit session memory"):
        for k in ["chat_buffer"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Cleared Streamlit session memory (files remain).")


# ---- main layout ----
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🧠 Ask (single paper)")
    q = st.text_area("Question", key="q_single", height=120)
    top_k = st.slider("top_k", 1, 20, 8, key="topk_single")

    if st.button("Ask single"):
        pid = (st.session_state.get("paper_id_single") or "").strip()
        if not pid:
            st.error("Enter paper_id in sidebar")
        elif not q.strip():
            st.error("Enter a question")
        else:
            append_memory(pid, "user", q.strip())
            with st.spinner("Calling /ask ..."):
                res = post_json("/ask", {"paper_id": pid, "question": q.strip(), "top_k": int(top_k)})
            ans = res.get("answer", "")
            append_memory(pid, "assistant", ans)
            st.success("Answer")
            st.write(ans)
            st.caption("Citations (top chunks):")
            st.json(res.get("citations", []))


with col2:
    st.subheader("📚 Ask Multi (compare papers)")
    q2 = st.text_area("Multi-paper question", key="q_multi", height=120)
    top_k_per = st.slider("top_k_per_paper", 1, 15, 8, key="topk_per")
    max_sources = st.slider("max_sources_per_paper", 1, 8, 6, key="max_sources")

    if st.button("Ask multi"):
        raw = st.session_state.get("paper_ids_multi_raw") or ""
        pids = [x.strip() for x in raw.splitlines() if x.strip()]
        if len(pids) < 2:
            st.error("Provide at least 2 paper_ids (one per line)")
        elif not q2.strip():
            st.error("Enter a question")
        else:
            with st.spinner("Calling /ask_multi ..."):
                res = post_json(
                    "/ask_multi",
                    {
                        "paper_ids": pids,
                        "question": q2.strip(),
                        "top_k_per_paper": int(top_k_per),
                        "max_sources_per_paper": int(max_sources),
                    },
                    timeout=240,
                )
            st.success("Multi-paper result")
            st.write(res.get("final_answer", ""))
            st.caption("Per-paper outputs:")
            st.json(res.get("per_paper", []))


st.divider()

st.subheader("📏 Eval ")
eval_col1, eval_col2 = st.columns([1, 1], gap="large")

with eval_col1:
    pid_eval = st.text_input("paper_id for eval", value=st.session_state.get("paper_id_single", ""), key="pid_eval")
    eval_q = st.text_area("Eval question", height=100, key="eval_q")
    eval_topk = st.slider("eval top_k", 1, 30, 8, key="eval_topk")
    gt = st.text_area("Optional ground_truth (for overlap score)", height=80, key="eval_gt")

    if st.button("Run eval (/eval/run)"):
        if not pid_eval.strip():
            st.error("paper_id required")
        elif not eval_q.strip():
            st.error("question required")
        else:
            payload = {
                "paper_id": pid_eval.strip(),
                "question": eval_q.strip(),
                "top_k": int(eval_topk),
                "ground_truth": gt.strip() if gt.strip() else None,
                "run_ragas": False,  # you decided to skip ragas
            }
            with st.spinner("Running eval..."):
                res = post_json("/eval/run", payload, timeout=900)
            st.success("Eval complete")
            st.json(res)

with eval_col2:
    st.subheader("📜 Memory viewer")
    pid_mem = st.text_input("paper_id to view memory", value=st.session_state.get("paper_id_single", ""), key="pid_mem")
    if st.button("Load memory"):
        msgs = load_memory(pid_mem.strip())
        if not msgs:
            st.info("No memory found for this paper yet.")
        else:
            for m in msgs:
                role = m.get("role")
                content = m.get("content")
                ts = m.get("ts")
                st.markdown(f"**{role}** — `{ts}`")
                st.write(content)
                st.markdown("---")
