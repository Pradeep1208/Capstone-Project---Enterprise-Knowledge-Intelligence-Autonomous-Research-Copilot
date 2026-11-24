"""Enterprise Knowledge Intelligence & Autonomous Research Copilot — Streamlit app."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from copilot.chunking import chunk_document
from copilot.config import SETTINGS
from copilot.embeddings import Embedder
from copilot.index import HybridIndex
from copilot.ingestion import SUPPORTED_EXTENSIONS, load_document
from copilot.llm import LLMClient
from copilot.rag import RAGPipeline
from copilot.reranker import CrossEncoderReranker, sigmoid
from copilot.retrieval import MODE_LABELS, MODES, Retriever

st.set_page_config(page_title="Enterprise Knowledge Copilot", page_icon="🔎", layout="wide")


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading embedding model (first run only)…")
def get_embedder() -> Embedder:
    return Embedder(SETTINGS.embedding_model)


@st.cache_resource(show_spinner="Loading reranker model (first run only)…")
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(SETTINGS.reranker_model)


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_base_index() -> HybridIndex:
    if HybridIndex.exists(SETTINGS.index_dir):
        return HybridIndex.load(SETTINGS.index_dir)
    return HybridIndex([], np.zeros((0, 0), dtype="float32"), SETTINGS.embedding_model)


@st.cache_resource
def get_llm() -> LLMClient:
    return LLMClient.from_secrets()


ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("index", None)        # base index + this session's uploads
ss.setdefault("uploaded", {})       # sha1 -> filename
ss.setdefault("n_queries", 0)


def active_index() -> HybridIndex:
    return ss.index if ss.index is not None else load_base_index()


def esc(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; financial answers contain many dollar signs."""
    return text.replace("$", r"\$")


# ----------------------------------------------------------------------------- sidebar
llm = get_llm()
with st.sidebar:
    st.title("🔎 Knowledge Copilot")
    page = st.radio("View", ["Chat", "Knowledge base", "Evaluation", "About"], label_visibility="collapsed")
    st.divider()

    st.subheader("Retrieval settings")
    mode = st.selectbox("Retrieval strategy", MODES, index=3, format_func=MODE_LABELS.get)
    top_k = st.slider("Passages sent to the model", 3, 10, SETTINGS.top_k)
    corrective = st.toggle("Corrective retrieval (rewrite & retry)", value=True)
    show_trace = st.toggle("Show retrieval trace", value=False)

    st.subheader("Add your own documents")
    files = st.file_uploader(
        f"Max {SETTINGS.max_upload_mb} MB each", type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    if st.button("Add to knowledge base", disabled=not files, width="stretch"):
        new_chunks, problems = [], []
        for f in files:
            data = f.getvalue()
            digest = hashlib.sha1(data).hexdigest()
            if digest in ss.uploaded:
                continue
            try:
                doc = load_document(data, f.name, doc_id=f"upload-{digest[:8]}", source=f"upload:{f.name}",
                                    max_mb=SETTINGS.max_upload_mb)
                new_chunks += chunk_document(doc, SETTINGS.chunker, SETTINGS.chunk_size, SETTINGS.chunk_overlap)
                ss.uploaded[digest] = f.name
            except ValueError as exc:
                problems.append(str(exc))
        for msg in problems:
            st.error(msg)
        if new_chunks:
            with st.spinner(f"Embedding {len(new_chunks)} passages…"):
                vectors = get_embedder().encode([c.text for c in new_chunks])
                ss.index = active_index().with_added(new_chunks, vectors)
            st.success(f"Added {len(new_chunks)} passages.")

    if ss.uploaded:
        st.caption("Uploaded this session: " + ", ".join(ss.uploaded.values()))
    if st.button("Clear chat and uploads", width="stretch"):
        ss.messages, ss.index, ss.uploaded, ss.n_queries = [], None, {}, 0
        st.rerun()

    st.divider()
    if llm.available:
        st.success(f"LLM connected: {llm.model}")
        st.caption("Retrieved passages are sent to the configured model provider. Do not upload confidential files.")
    else:
        st.warning("No LLM key configured — running in extractive mode (passages only).")


# ----------------------------------------------------------------------------- rendering helpers
def render_answer_details(ans, show_trace_flag: bool) -> None:
    ver = ans.verification
    if ver is not None:
        weak = [s for s in ver.sentences if s["supported"] is False]
        uncited = [s for s in ver.sentences if s["supported"] is None]
        if ver.invalid_ids:
            st.caption(f"⚠️ Removed citation numbers that do not match any retrieved passage: {ver.invalid_ids}")
        if weak:
            st.warning("Some statements look weakly supported by the cited passage — check them:\n\n"
                       + "\n".join(f"- {esc(s['text'])}" for s in weak))
        if uncited and not ans.abstained:
            st.caption(f"{len(uncited)} statement(s) carry no citation.")
    st.caption(f"Strategy: {MODE_LABELS[ans.mode]} · relevance {ans.relevance:.2f} · {ans.latency_ms / 1000:.1f}s"
               + (f" · {ans.prompt_tokens + ans.completion_tokens} tokens" if ans.used_llm else ""))
    if ans.flagged_injection:
        st.caption(f"🛡️ {ans.flagged_injection} passage(s) contained instruction-like text that was neutralised.")
    if ans.hits:
        with st.expander(f"Sources ({len(ans.hits)})", expanded=ans.abstained):
            for i, h in enumerate(ans.hits, start=1):
                st.markdown(f"**[{i}] {esc(h.chunk.citation_label())}**")
                bits = [f"ranking score {h.score:.3f}"]
                if h.vector_rank:
                    bits.append(f"vector rank {h.vector_rank}")
                if h.bm25_rank:
                    bits.append(f"BM25 rank {h.bm25_rank}")
                if h.rerank_score is not None:
                    bits.append(f"reranker relevance {float(sigmoid(h.rerank_score)):.2f}")
                st.caption(" · ".join(bits))
                st.text(h.chunk.text[:1500])
                st.divider()
    if show_trace_flag and ans.trace:
        with st.expander("Retrieval trace"):
            st.json(ans.trace)


# ----------------------------------------------------------------------------- pages
if page == "Chat":
    st.header("Enterprise Knowledge Intelligence & Research Copilot")
    index = active_index()
    if not len(index):
        st.info("No knowledge base is loaded. Upload documents in the sidebar, or build the demo index "
                "with `python scripts/build_index.py` (see README).")
    else:
        n_docs = len({c.doc_id for c in index.chunks})
        st.caption(f"Searching {n_docs} documents · {len(index):,} passages. Answers cite their sources; "
                   "when evidence is insufficient the assistant says so.")

    for msg in ss.messages:
        with st.chat_message(msg["role"]):
            st.markdown(esc(msg["content"]))
            if msg.get("answer") is not None:
                render_answer_details(msg["answer"], show_trace)

    if question := st.chat_input("Ask a research question about the documents…", disabled=not len(index)):
        if ss.n_queries >= SETTINGS.max_queries_per_session:
            st.warning("Session question limit reached. Use 'Clear chat and uploads' to start a new session.")
            st.stop()
        ss.n_queries += 1
        history = [{"role": m["role"], "content": m["content"]} for m in ss.messages[-6:]]
        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(esc(question))
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and drafting an answer…"):
                retriever = Retriever(index, get_embedder(), get_reranker(), SETTINGS.rrf_k)
                ans = RAGPipeline(retriever, llm, SETTINGS).answer(
                    question, history=history, mode=mode, top_k=top_k, corrective=corrective)
            st.markdown(esc(ans.text))
            render_answer_details(ans, show_trace)
        ss.messages.append({"role": "assistant", "content": ans.text, "answer": ans})

elif page == "Knowledge base":
    st.header("Knowledge base")
    index = active_index()
    if not len(index):
        st.info("The knowledge base is empty.")
    else:
        st.write(f"**{len(index):,} passages** · embedding model `{index.embedding_model}` · "
                 f"chunking `{SETTINGS.chunker}` ({SETTINGS.chunk_size} chars, {SETTINGS.chunk_overlap} overlap)")
        st.dataframe(pd.DataFrame(index.doc_summary()), width="stretch", hide_index=True)

elif page == "Evaluation":
    st.header("Evaluation results")
    st.caption("Produced offline by `scripts/eval_beir.py` and `scripts/eval_financebench.py` (see README). "
               "BEIR uses human relevance judgements; FinanceBench uses expert-written questions on real SEC filings.")
    files_found = sorted(SETTINGS.results_dir.glob("*_summary.csv"))
    if not files_found:
        st.info("No result files yet. Run the evaluation scripts and commit the CSVs in the `results/` folder.")
    for f in files_found:
        df = pd.read_csv(f)
        st.subheader(f.stem.replace("_", " "))
        st.dataframe(df, width="stretch", hide_index=True)
        if "config" in df.columns:
            metric_cols = [c for c in df.columns if any(t in c for t in ("ndcg@10", "recall@10", "mrr@10"))]
            if metric_cols:
                st.bar_chart(df.set_index("config")[metric_cols])

else:
    st.header("About this project")
    st.markdown(
        """
**Pipeline:** documents → parsing & cleaning → metadata → chunking → embeddings + BM25 index →
hybrid retrieval (Reciprocal Rank Fusion) → cross-encoder reranker → relevance check →
corrective retry (query rewrite) or abstain → grounded generation → citation verification.

**Limitations (please read):**
- Citations show *where* evidence came from; they do not guarantee an answer is correct. The verification step is a lexical heuristic.
- Tables in PDFs may be extracted as plain text, which can scramble columns.
- Scanned PDFs without a text layer are not supported (no OCR).
- Retrieval quality depends on the corpus; the relevance threshold should be calibrated on your own benchmark.
- This is a demonstration — do not upload confidential or personal data.
        """
    )
