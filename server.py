"""
Flask Web Server
================
REST API for the RAG chatbot. Serves the frontend and handles:
  POST /api/upload   → ingest PDF
  POST /api/chat     → answer a question
  GET  /api/status   → pipeline health check
  POST /api/reset    → clear conversation history
"""

import os
import time
import logging
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from rag_pipeline import RAGPipeline
from llm_client import LLMClient, ConversationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
print("Static folder:", app.static_folder)
print("Exists:", os.path.exists(app.static_folder))
CORS(app)

# ── Global state ──────────────────────────────
pipeline = RAGPipeline(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=800,
    chunk_overlap=150,
    top_k=5,
)
llm = LLMClient()
conversation = ConversationManager()

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "rag_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

pipeline_ready = False
indexed_filename = None


# ── Routes ────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), "static"), "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "ready": pipeline_ready,
        "indexed_file": indexed_filename,
        "total_chunks": pipeline.vector_store.index.ntotal if pipeline.vector_store else 0,
        "llm_model": llm.model,
        "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
    })


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    global pipeline_ready, indexed_filename

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files supported"}), 400

    # Save uploaded file
    save_path = UPLOAD_FOLDER / file.filename
    file.save(str(save_path))
    logger.info(f"📁 File saved: {save_path}")

    force = request.form.get("force_reindex", "false").lower() == "true"

    try:
        start = time.time()
        pipeline.ingest_pdf(str(save_path), force_reindex=force)
        elapsed = round(time.time() - start, 2)

        pipeline_ready = True
        indexed_filename = file.filename
        conversation.clear()

        return jsonify({
            "success": True,
            "filename": file.filename,
            "total_chunks": pipeline.vector_store.index.ntotal,
            "time_taken": elapsed,
        })

    except Exception as e:
        logger.exception("Ingestion failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    if not pipeline_ready:
        return jsonify({"error": "No document indexed yet. Please upload a PDF first."}), 400

    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        # 1. Retrieve relevant chunks
        start = time.time()
        chunks = pipeline.retrieve(query)
        retrieval_ms = round((time.time() - start) * 1000)

        # 2. Generate answer
        start = time.time()
        result = llm.answer(query, chunks)
        llm_ms = round((time.time() - start) * 1000)

        # 3. Update conversation history
        conversation.add("user", query)
        conversation.add("assistant", result["answer"], metadata={
            "chunks_used": result["chunks_used"],
            "model": result["model"],
        })

        return jsonify({
            "answer": result["answer"],
            "model": result["model"],
            "chunks_used": result["chunks_used"],
            "sources": [
                {
                    "page": c["page_num"],
                    "score": c["score"],
                    "preview": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                }
                for c in chunks
            ],
            "timing": {
                "retrieval_ms": retrieval_ms,
                "llm_ms": llm_ms,
            },
        })

    except Exception as e:
        logger.exception("Chat error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    conversation.clear()
    return jsonify({"success": True, "message": "Conversation history cleared."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting RAG server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
