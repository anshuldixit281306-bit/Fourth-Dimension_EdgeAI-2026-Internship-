"""
Campus Handbook Guide — Flask Web Application
=============================================
Target HW  : NVIDIA Jetson Nano (4 GB RAM)
Backend    : handbook_bot.py  (Hybrid RAG — BM25 + FAISS + llama3.2:1b)
Server     : Flask (lightweight WSGI — no extra dependencies)
Mode       : Fully Offline — zero external API calls
Access     : http://localhost:5000  or  http://JETSON_IP:5000

Routes
------
GET  /          → Main chat page
POST /ask       → Submit a question, get an answer + citation JSON
POST /upload    → Upload a PDF file
POST /rebuild   → Rebuild the FAISS + BM25 index
GET  /status    → JSON system status (models, index, RAM/CPU)
POST /clear     → Clear server-side chat history

Design decisions for Jetson Nano
---------------------------------
- Models loaded once at startup and held in RAM between requests.
- Index rebuild runs in a background thread so HTTP stays responsive.
- psutil metrics (RAM / CPU) are sampled fresh on each /status call.
- Flask's built-in server is used; no Gunicorn/uWSGI needed at this scale.
- Threads: Flask debug=False + threaded=True keeps memory low.
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import logging
import threading
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
)
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────────────────────────
# psutil — optional system monitor
# ─────────────────────────────────────────────────────────────────
try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    psutil = None
    PSUTIL_OK = False

# ─────────────────────────────────────────────────────────────────
# Backend — Hybrid RAG
# ─────────────────────────────────────────────────────────────────
try:
    import handbook_bot as bot
except ImportError as _e:
    print(
        f"[FATAL] Cannot import handbook_bot.py — place it in the same directory.\n{_e}",
        file=sys.stderr,
    )
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════
# FLASK APP SETUP
# ═════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.urandom(32)          # session signing key (ephemeral)

# ── Upload config ─────────────────────────────────────────────────
UPLOAD_FOLDER  = Path("uploads")
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"]   = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB cap

UPLOAD_FOLDER.mkdir(exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ═════════════════════════════════════════════════════════════════
# GLOBAL STATE  — models live in RAM between requests
# ═════════════════════════════════════════════════════════════════

_state: dict = {
    "embed_model"      : None,
    "faiss_index"      : None,
    "chunks"           : None,
    "bm25"             : None,
    "is_ready"         : False,
    "is_rebuilding"    : False,
    "rebuild_error"    : None,
    "rebuild_elapsed"  : None,
    "startup_error"    : None,
    "chat_history"     : [],          # list of {"role", "text", "citation", "ts"}
}

_state_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")


# ═════════════════════════════════════════════════════════════════
# STARTUP — load or build index on first run
# ═════════════════════════════════════════════════════════════════

def _do_startup() -> None:
    """Run in a background thread at server start."""
    log.info("Startup: initialising Hybrid RAG pipeline…")
    try:
        em, fi, ch, bm = bot.setup()
        with _state_lock:
            _state["embed_model"]  = em
            _state["faiss_index"]  = fi
            _state["chunks"]       = ch
            _state["bm25"]         = bm
            _state["is_ready"]     = True
            _state["startup_error"] = None
        log.info("Startup complete — %d chunks loaded", len(ch))
    except Exception as exc:
        log.error("Startup FAILED: %s", exc)
        with _state_lock:
            _state["startup_error"] = str(exc)
            _state["is_ready"]      = False


threading.Thread(target=_do_startup, daemon=True, name="startup").start()


# ═════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════

def _system_status() -> dict:
    """Build the JSON payload returned by GET /status."""
    with _state_lock:
        ready       = _state["is_ready"]
        rebuilding  = _state["is_rebuilding"]
        chunks      = _state["chunks"] or []
        fi          = _state["faiss_index"]
        embed_ok    = _state["embed_model"] is not None
        err         = _state["rebuild_error"] or _state["startup_error"]

    # PDF info
    pdf_path  = Path(bot.cfg.PDF_PATH)
    pdf_name  = pdf_path.name if pdf_path.exists() else None
    pdf_size  = pdf_path.stat().st_size if pdf_path.exists() else 0

    # RAM / CPU
    if PSUTIL_OK:
        mem      = psutil.virtual_memory()
        ram_used = round(mem.used  / (1024 ** 3), 2)
        ram_total= round(mem.total / (1024 ** 3), 2)
        ram_pct  = mem.percent
        cpu_pct  = psutil.cpu_percent(interval=None)
    else:
        ram_used = ram_total = ram_pct = cpu_pct = None

    # Ollama ping
    ollama_ok = False
    try:
        import ollama as _ol
        _ol.list()
        ollama_ok = True
    except Exception:
        pass

    return {
        "ready"          : ready,
        "rebuilding"     : rebuilding,
        "error"          : err,
        "ollama_ok"      : ollama_ok,
        "embed_ok"       : embed_ok,
        "faiss_vectors"  : fi.ntotal if fi else 0,
        "chunks_count"   : len(chunks),
        "pdf_name"       : pdf_name,
        "pdf_size_kb"    : round(pdf_size / 1024, 1),
        "ollama_model"   : bot.cfg.OLLAMA_MODEL,
        "embed_model"    : bot.cfg.EMBED_MODEL,
        "top_k"          : bot.cfg.TOP_K,
        "bm25_pool"      : bot.cfg.BM25_POOL,
        "score_thresh"   : bot.cfg.SCORE_THRESH,
        "chunk_words"    : bot.cfg.CHUNK_WORDS,
        "ram_used_gb"    : ram_used,
        "ram_total_gb"   : ram_total,
        "ram_pct"        : ram_pct,
        "cpu_pct"        : cpu_pct,
        "chat_count"     : len(_state["chat_history"]),
    }


# ═════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main chat UI."""
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask
    Body (JSON): { "question": "..." }
    Returns JSON: { "answer": "...", "citation": "...", "elapsed": 1.23 }
    """
    data     = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Empty question."}), 400

    with _state_lock:
        ready        = _state["is_ready"]
        rebuilding   = _state["is_rebuilding"]
        embed_model  = _state["embed_model"]
        faiss_index  = _state["faiss_index"]
        chunks       = _state["chunks"]
        bm25         = _state["bm25"]

    if rebuilding:
        return jsonify({"error": "Index is being rebuilt — please wait."}), 503

    if not ready:
        err = _state.get("startup_error") or "System not ready. Check if a PDF is uploaded."
        return jsonify({"error": err}), 503

    try:
        t0        = time.perf_counter()
        retrieved = bot.retrieve(question, embed_model, faiss_index, chunks, bm25)
        prompt    = bot.build_prompt(question, retrieved)
        answer    = bot.ask_llm(prompt)
        elapsed   = round(time.perf_counter() - t0, 2)

        pages = sorted({c["page"] for c in retrieved}) if retrieved else []
        if pages:
            label    = "Pages" if len(pages) > 1 else "Page"
            citation = f"📄 Source {label}: " + ", ".join(str(p) for p in pages)
        else:
            citation = "ℹ️ No matching section found in the handbook."

        entry = {
            "role"    : "assistant",
            "text"    : answer,
            "citation": citation,
            "ts"      : datetime.now().strftime("%H:%M"),
        }
        with _state_lock:
            _state["chat_history"].append({
                "role": "user",
                "text": question,
                "citation": "",
                "ts": datetime.now().strftime("%H:%M"),
            })
            _state["chat_history"].append(entry)

        return jsonify({
            "answer"  : answer,
            "citation": citation,
            "elapsed" : elapsed,
        })

    except Exception as exc:
        log.error("Query error: %s", exc)
        return jsonify({"error": f"Query failed: {exc}"}), 500


# ─────────────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    """
    POST /upload  (multipart/form-data, field name: 'pdf')
    Saves file as handbook.pdf and returns JSON status.
    """
    if "pdf" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    f = request.files["pdf"]
    if f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    try:
        # Save to upload folder first, then move to bot.cfg.PDF_PATH
        safe_name = secure_filename(f.filename)
        tmp_path  = UPLOAD_FOLDER / safe_name
        f.save(str(tmp_path))
        shutil.move(str(tmp_path), bot.cfg.PDF_PATH)

        size_kb = round(Path(bot.cfg.PDF_PATH).stat().st_size / 1024, 1)
        log.info("PDF uploaded: %s (%.1f KB)", safe_name, size_kb)

        return jsonify({
            "success" : True,
            "filename": safe_name,
            "size_kb" : size_kb,
            "message" : f"'{safe_name}' uploaded ({size_kb} KB). Click Rebuild Index to index it.",
        })

    except Exception as exc:
        log.error("Upload error: %s", exc)
        return jsonify({"error": f"Upload failed: {exc}"}), 500


# ─────────────────────────────────────────────────────────────────
@app.route("/rebuild", methods=["POST"])
def rebuild():
    """
    POST /rebuild
    Starts a background thread that rebuilds FAISS + BM25 index.
    Returns immediately with 202 Accepted.
    """
    with _state_lock:
        if _state["is_rebuilding"]:
            return jsonify({"error": "A rebuild is already in progress."}), 409

        if not Path(bot.cfg.PDF_PATH).exists():
            return jsonify({"error": "No PDF found. Upload a PDF first."}), 400

        # Mark as rebuilding immediately
        _state["is_rebuilding"] = True
        _state["is_ready"]      = False
        _state["rebuild_error"] = None

    threading.Thread(target=_do_rebuild, daemon=True, name="rebuild").start()

    return jsonify({"message": "Index rebuild started."}), 202


def _do_rebuild() -> None:
    """Background thread: delete old artifacts, rebuild, update state."""
    # Remove stale artifacts
    for p in (bot.cfg.INDEX_PATH, bot.cfg.CHUNKS_PATH):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    log.info("Rebuild: starting…")
    t0 = time.perf_counter()
    try:
        em, fi, ch, bm = bot.setup()
        elapsed = round(time.perf_counter() - t0, 1)

        with _state_lock:
            _state["embed_model"]     = em
            _state["faiss_index"]     = fi
            _state["chunks"]          = ch
            _state["bm25"]            = bm
            _state["is_ready"]        = True
            _state["is_rebuilding"]   = False
            _state["rebuild_error"]   = None
            _state["rebuild_elapsed"] = elapsed

        log.info("Rebuild complete — %d chunks in %.1fs", len(ch), elapsed)

    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 1)
        log.error("Rebuild FAILED after %.1fs: %s", elapsed, exc)
        with _state_lock:
            _state["is_rebuilding"] = False
            _state["is_ready"]      = False
            _state["rebuild_error"] = str(exc)


# ─────────────────────────────────────────────────────────────────
@app.route("/status")
def status():
    """GET /status → JSON system snapshot (polled by the front-end)."""
    return jsonify(_system_status())


# ─────────────────────────────────────────────────────────────────
@app.route("/clear", methods=["POST"])
def clear():
    """POST /clear → wipe server-side chat history."""
    with _state_lock:
        _state["chat_history"].clear()
    log.info("Chat history cleared")
    return jsonify({"success": True})


# ─────────────────────────────────────────────────────────────────
@app.route("/history")
def history():
    """GET /history → return full chat history as JSON."""
    with _state_lock:
        hist = list(_state["chat_history"])
    return jsonify({"history": hist})


# ═════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # threaded=True: each request runs in its own thread, keeping the
    # server responsive while the RAG pipeline processes a query.
    # debug=False: required for production / Jetson stability.
    # host="0.0.0.0": accessible from any device on the LAN.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
