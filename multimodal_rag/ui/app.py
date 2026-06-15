# ============================================================
#  ui/app.py — Streamlit Web Interface for Multimodal RAG
#  Run: streamlit run ui/app.py
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import tempfile
import time
from pathlib import Path

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1A3C5E; }
    .sub-header  { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .source-badge {
        background: #EAF3FB; border: 1px solid #C5D8E8;
        border-radius: 6px; padding: 4px 10px;
        font-size: 0.82rem; color: #1A3C5E; display: inline-block;
        margin: 2px;
    }
    .score-badge {
        background: #E8F5EE; border: 1px solid #B2DFDB;
        border-radius: 6px; padding: 2px 8px;
        font-size: 0.78rem; color: #1A5C3A;
    }
    .quiz-card {
        background: #FAFAFA; border: 1px solid #E0E0E0;
        border-radius: 10px; padding: 16px; margin-bottom: 14px;
    }
    .correct-ans { color: #1A7A3C; font-weight: 600; }
    .stButton>button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Multimodal RAG")
    st.markdown("---")

    # ── File Upload ──
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag & drop files",
        accept_multiple_files=True,
        type=["pdf", "pptx", "ppt", "txt", "md", "csv", "json",
              "mp4", "avi", "mov", "mkv"],
        help="Supported: PDF, PPTX, TXT, MD, CSV, JSON, MP4, AVI, MOV"
    )

    if uploaded_files:
        if st.button("⚡ Index Documents", type="primary", use_container_width=True):
            _index_uploads(uploaded_files)

    st.markdown("---")

    # ── Index Stats ──
    st.markdown("### 📊 Index Stats")
    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.session_state["stats_refresh"] = True

    try:
        from vectorstore.chroma_store import get_stats
        stats = get_stats()
        st.metric("Total Chunks", stats["total_chunks"])
        if stats["by_type"]:
            for dtype, count in stats["by_type"].items():
                icon = {"pdf":"📄","pptx":"📊","text":"📝",
                        "video_transcript":"🎬","video_frame":"🖼️"}.get(dtype, "📌")
                st.markdown(f"{icon} **{dtype}**: {count}")
    except Exception:
        st.info("No documents indexed yet.")

    st.markdown("---")
    if st.button("🗑️ Clear Index", use_container_width=True):
        if st.session_state.get("confirm_clear"):
            from vectorstore.chroma_store import delete_collection
            delete_collection()
            st.success("Index cleared!")
            st.session_state["confirm_clear"] = False
            st.rerun()
        else:
            st.session_state["confirm_clear"] = True
            st.warning("Click again to confirm deletion.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    from config import LLM_BACKEND, EMBED_MODEL
    st.markdown(f"**LLM:** `{LLM_BACKEND}`")
    st.markdown(f"**Embedder:** `{EMBED_MODEL}`")
    st.markdown(f"**Store:** `ChromaDB`")


# ── Helper: Index uploads ────────────────────────────────────
def _index_uploads(uploaded_files):
    from ingestion.pipeline import ingest_file
    from utils.chunker import chunk_documents
    from vectorstore.chroma_store import add_chunks

    progress = st.sidebar.progress(0, text="Starting...")
    total = len(uploaded_files)
    all_chunks = []

    for i, uf in enumerate(uploaded_files):
        progress.progress((i) / total, text=f"Parsing {uf.name}...")
        suffix = Path(uf.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name

        try:
            raw = ingest_file(tmp_path)
            # Patch metadata source name
            for c in raw:
                c["metadata"]["source"] = uf.name
            all_chunks.extend(raw)
        except Exception as e:
            st.sidebar.error(f"Failed: {uf.name} — {e}")
        finally:
            os.unlink(tmp_path)

    if all_chunks:
        progress.progress(0.8, text="Chunking & embedding...")
        chunks = chunk_documents(all_chunks)
        n = add_chunks(chunks)
        progress.progress(1.0, text="Done!")
        st.sidebar.success(f"✅ Indexed {n} chunks from {total} file(s)!")
        time.sleep(1)
        progress.empty()
        st.rerun()
    else:
        progress.empty()
        st.sidebar.warning("No content extracted from uploaded files.")


# ── Main Tabs ────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 Multimodal RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions, generate quizzes, and create presentations from your documents.</div>', unsafe_allow_html=True)

tab_qa, tab_quiz, tab_ppt = st.tabs(["💬 Ask a Question", "📝 Quiz Generator", "📊 PPT Generator"])


# ════════════════════════════════════════
#  TAB 1: Q&A
# ════════════════════════════════════════
with tab_qa:
    st.markdown("### Ask anything from your uploaded documents")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Your question", placeholder="e.g. What is the attention mechanism?")
    with col2:
        filter_type = st.selectbox(
            "Filter by type",
            ["All", "pdf", "pptx", "text", "video_transcript"],
            index=0
        )
        top_k = st.slider("Top K chunks", 3, 12, 6)

    if st.button("🔍 Search & Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    from retrieval.rag_engine import answer_query
                    ft = None if filter_type == "All" else filter_type
                    result = answer_query(query, filter_type=ft, top_k=top_k)

                    st.markdown("#### 💡 Answer")
                    st.markdown(result["answer"])

                    if result["sources"]:
                        st.markdown("#### 📎 Sources Used")
                        cols = st.columns(min(len(result["sources"]), 3))
                        for i, src in enumerate(result["sources"]):
                            with cols[i % len(cols)]:
                                st.markdown(
                                    f'<span class="source-badge">{src["label"]}</span> '
                                    f'<span class="score-badge">score: {src["score"]}</span>',
                                    unsafe_allow_html=True
                                )

                    with st.expander("🔍 View retrieved context"):
                        for i, hit in enumerate(result["context_used"]):
                            st.markdown(f"**Chunk {i+1}** — Score: `{hit['score']}`")
                            st.markdown(f"*Source: {hit['metadata'].get('source','?')}*")
                            st.text(hit["text"][:500] + "..." if len(hit["text"]) > 500 else hit["text"])
                            st.divider()

                except Exception as e:
                    st.error(f"Error: {e}")


# ════════════════════════════════════════
#  TAB 2: QUIZ
# ════════════════════════════════════════
with tab_quiz:
    st.markdown("### Generate MCQ Quiz from your documents")

    col1, col2, col3 = st.columns(3)
    with col1:
        quiz_topic = st.text_input("Quiz topic", placeholder="e.g. Transformer Architecture")
    with col2:
        n_q = st.slider("Number of questions", 3, 15, 5)
    with col3:
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)

    if st.button("🎯 Generate Quiz", type="primary"):
        if not quiz_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Generating quiz..."):
                try:
                    from quiz.quiz_generator import generate_quiz
                    quiz = generate_quiz(quiz_topic, n_questions=n_q, difficulty=difficulty)

                    st.markdown(f"#### 📝 Quiz: {quiz.topic}")
                    st.markdown(f"*{len(quiz.questions)} questions | Difficulty: {difficulty}*")
                    st.divider()

                    # Store in session for interactive mode
                    st.session_state["quiz"] = quiz
                    st.session_state["quiz_answers"] = {}

                except Exception as e:
                    st.error(f"Failed to generate quiz: {e}")

    # ── Display quiz interactively ──
    if "quiz" in st.session_state:
        quiz = st.session_state["quiz"]
        answers = st.session_state.get("quiz_answers", {})

        for i, q in enumerate(quiz.questions):
            with st.container():
                st.markdown(f'<div class="quiz-card">', unsafe_allow_html=True)
                st.markdown(f"**Q{i+1}. {q.question}**")

                opts = {o.label: o.text for o in q.options}
                choice = st.radio(
                    f"Select answer for Q{i+1}",
                    options=list(opts.keys()),
                    format_func=lambda x: f"{x}. {opts[x]}",
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                answers[i] = choice
                st.markdown('</div>', unsafe_allow_html=True)

        st.session_state["quiz_answers"] = answers

        if st.button("✅ Submit & Check Answers", type="primary"):
            correct = 0
            st.markdown("### 📊 Results")
            for i, q in enumerate(quiz.questions):
                user_ans = answers.get(i, "")
                is_correct = user_ans == q.answer
                if is_correct:
                    correct += 1
                    st.success(f"**Q{i+1}** ✅ Correct! ({q.answer})")
                else:
                    st.error(f"**Q{i+1}** ❌ Your answer: {user_ans} | Correct: {q.answer}")
                with st.expander(f"Explanation for Q{i+1}"):
                    st.markdown(f'<span class="correct-ans">✔ {q.answer}. {dict((o.label, o.text) for o in q.options)[q.answer]}</span>', unsafe_allow_html=True)
                    st.markdown(q.explanation)
                    if q.source:
                        st.caption(f"Source: {q.source}")

            pct = int(correct / len(quiz.questions) * 100)
            st.markdown(f"### 🎯 Score: {correct}/{len(quiz.questions)} ({pct}%)")
            if pct == 100:
                st.balloons()


# ════════════════════════════════════════
#  TAB 3: PPT GENERATOR
# ════════════════════════════════════════
with tab_ppt:
    st.markdown("### Generate a Presentation from your documents")

    col1, col2, col3 = st.columns(3)
    with col1:
        ppt_topic = st.text_input("Presentation topic", placeholder="e.g. Introduction to Transformers")
    with col2:
        n_slides = st.slider("Number of slides", 4, 15, 8)
    with col3:
        ppt_style = st.selectbox("Style", ["academic", "professional", "minimal"], index=0)

    if st.button("🎨 Generate Presentation", type="primary"):
        if not ppt_topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Generating presentation... (this may take 30-60 seconds)"):
                try:
                    from pptgen.ppt_generator import generate_ppt
                    output_path = generate_ppt(ppt_topic, n_slides=n_slides, style=ppt_style)

                    st.success(f"✅ Presentation generated: {Path(output_path).name}")
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download .pptx",
                            data=f,
                            file_name=Path(output_path).name,
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            type="primary",
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"Failed to generate presentation: {e}")
