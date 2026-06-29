# 📘 Campus Handbook Guide

> **A fully offline, privacy-first campus handbook chatbot** powered by Hybrid RAG — designed and optimised for the NVIDIA Jetson Nano (4 GB RAM).

---

## Overview

**Campus Handbook Guide** lets students and staff ask natural-language questions about any campus handbook PDF and receive accurate, page-cited answers — without sending a single byte to the internet.

It combines **BM25 sparse retrieval** (keyword matching) with **FAISS dense vector search** (semantic similarity) to form a Hybrid RAG pipeline, re-ranks candidates with cosine similarity, and generates answers using **llama3.2:1b** via Ollama — all running locally on device.

The **Flask web UI** is dark-themed, ChatGPT-style, and includes live RAM/CPU monitoring, system status indicators, one-click PDF upload and index rebuild, and chat export. It is accessible from **any device on the same network** — no browser extensions or installs needed on the client.

---

## Features

| Feature | Detail |
|---|---|
| 🔒 Fully Offline | Ollama + Embeddings + FAISS run locally, no internet required after one-time model download |
| 🌐 Web UI | Flask-served dark UI — open in any browser on the LAN |
| 📄 PDF Upload | File-picker upload; auto hash-check to avoid redundant rebuilds |
| ⚡ Hybrid RAG | BM25 pre-filter → FAISS dense re-rank → top-K chunks |
| 📑 Page Citations | Every answer shows exact source page numbers |
| 🔄 Auto Index | Detects PDF changes by MD5 hash; rebuilds automatically |
| 📊 RAM/CPU Monitor | Live psutil metrics polled every 4 seconds |
| 🤖 Status Panel | Ollama, embeddings, FAISS, and chunk count shown in the sidebar |
| 💾 Export Chat | Download conversation as Markdown directly from the browser |
| 📱 Mobile Responsive | Works on phones and tablets on the same Wi-Fi network |
| 🖥 Multi-device | Access from any laptop, phone, or tablet — Jetson Nano is the server |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Campus Handbook Guide                         │
│                                                                  │
│  Browser (any device on LAN)                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  index.html + style.css + script.js                        │  │
│  │  • Chat UI    • PDF upload    • Status panel               │  │
│  │  • Rebuild button    • Export chat    • RAM/CPU monitor     │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │ HTTP (LAN)                           │
│  Jetson Nano — Flask Server (app.py)                             │
│  ┌────────────────────────▼───────────────────────────────────┐  │
│  │  GET  /          → Serve chat UI                           │  │
│  │  POST /ask       → Run Hybrid RAG query → return JSON      │  │
│  │  POST /upload    → Save PDF → handbook.pdf                 │  │
│  │  POST /rebuild   → Start background index rebuild          │  │
│  │  GET  /status    → Return system snapshot (JSON)           │  │
│  │  POST /clear     → Clear chat history                      │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │                                      │
│  ┌────────────────────────▼───────────────────────────────────┐  │
│  │              handbook_bot.py  (unchanged)                   │  │
│  │                                                            │  │
│  │  PDF ──▶ page-aware 100-word chunks                        │  │
│  │          │                                                 │  │
│  │          ├──▶ BM25 index  (in-RAM)                         │  │
│  │          └──▶ FAISS index (on-disk)                        │  │
│  │                                                            │  │
│  │  Query ──▶ BM25 pool (top-10)                              │  │
│  │             └──▶ Dense re-rank (cosine similarity)         │  │
│  │                   └──▶ top-K chunk(s)                      │  │
│  │                         └──▶ Prompt                        │  │
│  │                               └──▶ Ollama llama3.2:1b      │  │
│  │                                     └──▶ Answer + Pages    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Storage : handbook.pdf │ handbook.index │ chunks.pkl            │
│  Models  : all-MiniLM-L6-v2  │  llama3.2:1b (via Ollama)        │
└──────────────────────────────────────────────────────────────────┘
```

### Key components

- **Flask** — lightweight WSGI server; serves the UI and JSON API endpoints.
- **BM25** — custom implementation, no external library; catches exact keyword matches (room numbers, codes, abbreviations).
- **FAISS IndexFlatIP** — brute-force cosine similarity on L2-normalised vectors; lightweight, no GPU training needed.
- **all-MiniLM-L6-v2** — 384-dimensional embeddings; fast on CPU, ~80 MB model size.
- **llama3.2:1b via Ollama** — 1-billion parameter LLM; fits comfortably in 4 GB RAM alongside the embedding model.
- **Hybrid retrieval** — BM25 narrows the search space to 10 candidates; dense vectors re-rank for semantic accuracy.
- **Background threads** — index rebuild and startup run in daemon threads so Flask stays responsive.

---

## Project Structure

```
Campus-Handbook-Chatbot/
├── app.py                  # Flask server — routes + API endpoints
├── handbook_bot.py         # Backend: Hybrid RAG pipeline (unchanged)
├── requirements.txt        # Python dependencies
├── .gitignore              # Files excluded from version control
├── README.md               # This file
│
├── templates/
│   └── index.html          # Main chat UI (Jinja2 template)
│
├── static/
│   ├── style.css           # Dark theme, responsive layout
│   └── script.js           # Chat logic, status polling, upload, export
│
├── uploads/                # Temporary upload staging (gitignored)
├── faiss_index/            # Reserved for index artifacts (gitignored)
├── docs/                   # Additional documentation
│
├── handbook.pdf            # Your PDF (gitignored)
├── handbook.index          # FAISS vector index (gitignored)
└── chunks.pkl              # Chunk metadata + PDF hash (gitignored)
```

---

## Installation

### Prerequisites

- Python 3.8 or later
- [Ollama](https://ollama.com/download/linux) installed and running
- `llama3.2:1b` model pulled

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/campus-handbook-guide.git
cd campus-handbook-guide
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Jetson Nano:** Do **not** install PyTorch via pip. Use the NVIDIA-provided wheel instead (see [Jetson Nano deployment](#jetson-nano-deployment) below).

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ollama setup

Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama service:
```bash
ollama serve &
```

Pull the model (one-time, requires internet):
```bash
ollama pull llama3.2:1b
```

Verify it works:
```bash
ollama run llama3.2:1b "Hello"
```

### 5. Download the Embedding Model (one-time)

The embedding model must be downloaded once while connected to the internet:

```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
```

After this, the application can rebuild indexes and answer questions completely offline.

---

## Usage

### Start the application

```bash
python app.py
```

### Access the UI

| Device | URL |
|---|---|
| Jetson Nano itself | `http://localhost:5000` |
| Laptop / phone on same network | `http://<JETSON_IP>:5000` |

Find your Jetson's IP:
```bash
hostname -I
```

### Workflow

1. **Upload PDF** — Click `📂 Upload PDF` in the sidebar. Accepts `.pdf` files up to 50 MB.
2. **Build Index** — Click `🔄 Rebuild Index`. The sidebar shows a spinner while the index builds in the background; status updates automatically every 4 seconds.
3. **Ask questions** — Type in the input box and press **Enter** (or Shift+Enter for newline).
4. **View citations** — Each answer shows the source page number(s) in blue below the response.
5. **Monitor resources** — The RAM and CPU bars update live in the sidebar.
6. **Export** — Click `💾 Export Chat` to download the conversation as a Markdown file.

### Flask API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serve the main chat page |
| `POST` | `/ask` | `{"question": "..."}` → `{"answer": "...", "citation": "...", "elapsed": 1.23}` |
| `POST` | `/upload` | Multipart PDF upload → `{"success": true, "filename": "...", "size_kb": 120}` |
| `POST` | `/rebuild` | Trigger background index rebuild → `202 Accepted` |
| `GET` | `/status` | System snapshot: models, RAM/CPU, chunk count, PDF info |
| `POST` | `/clear` | Clear server-side chat history |
| `GET` | `/history` | Return full chat history as JSON |

### Sample Questions

```
What is the policy on late submissions?
How do I apply for a leave of absence?
What are the library opening hours?
Who is the Dean of Students?
What GPA is required for the Dean's List?
```

---

## Jetson Nano Deployment

### Hardware requirements

| Component | Minimum |
|---|---|
| RAM | 4 GB (unified) |
| Storage | 16 GB microSD (Class 10 or faster) |
| OS | JetPack 4.6 / Ubuntu 18.04 |

### PyTorch for Jetson

Do **not** use `pip install torch`. Install from NVIDIA's aarch64 wheel:

```bash
# JetPack 4.6 (CUDA 10.2)
pip install torch==1.12.0 \
  --find-links https://nvidia.box.com/v/torch-1120-cp36-linux-aarch64
```

Check the [NVIDIA Jetson Zoo](https://elinux.org/Jetson_Zoo#PyTorch_.28Caffe2.29) for the wheel matching your JetPack version.

### Build index on a development machine (recommended)

Embedding 1,000 chunks takes ~15 minutes on Jetson Nano. Build once on a laptop and copy the artifacts:

```bash
# On dev machine
python3 -c "import handbook_bot as b; b.setup()"

# Copy to Jetson
scp handbook.index chunks.pkl jetson@<IP>:~/campus-handbook-guide/
```

On Jetson, the app will detect the prebuilt index and skip the rebuild on startup.

### Run on Jetson

```bash
# Start Ollama (if not already running as a service)
ollama serve &

# Launch the Flask server
python3 app.py
```

Access from any device on the LAN: `http://<JETSON_IP>:5000`

### Run as a systemd service (auto-start on boot)

Create `/etc/systemd/system/handbook-guide.service`:

```ini
[Unit]
Description=Campus Handbook Guide — Flask Server
After=network.target ollama.service

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/campus-handbook-guide
ExecStart=/home/jetson/campus-handbook-guide/venv/bin/python app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable handbook-guide
sudo systemctl start handbook-guide
sudo systemctl status handbook-guide
```

### Memory tips

- Keep only `llama3.2:1b` pulled (`ollama list` to check).
- Close other applications before launching.
- The app loads models once at startup and holds them in RAM — do not rebuild the index repeatedly during a session.
- Reduce `EMBED_BATCH` in `Config` (e.g., `8` instead of `16`) if you run out of RAM during rebuild.

---

## Configuration

Edit the `Config` class in `handbook_bot.py` to tune behaviour:

```python
class Config:
    PDF_PATH     = "handbook.pdf"
    INDEX_PATH   = "handbook.index"
    CHUNKS_PATH  = "chunks.pkl"
    CHUNK_WORDS  = 100       # words per chunk
    CHUNK_OVERLAP= 20        # overlap between chunks
    EMBED_MODEL  = "all-MiniLM-L6-v2"
    EMBED_BATCH  = 16        # reduce for lower RAM  (try 8 on Jetson)
    BM25_POOL    = 10        # BM25 candidates before dense re-rank
    TOP_K        = 1         # chunks sent to LLM
    SCORE_THRESH = 0.25      # min cosine score (0–1)
    OLLAMA_MODEL = "llama3.2:1b"
```

---

## Troubleshooting

### App starts but shows "Initialising…" indefinitely
- Ensure Ollama is running: `ollama serve`
- Confirm the model is pulled: `ollama list`
- Check `python app.py` terminal output for errors.

### "No PDF found. Upload a PDF first."
Upload a PDF via the sidebar button, then click **Rebuild Index**.

### "Could not reach the local Ollama model"
1. Check Ollama is running: `ollama list`
2. Start it: `ollama serve`
3. Confirm the model is pulled: `ollama pull llama3.2:1b`
4. Test: `curl http://localhost:11434/api/tags`

### Rebuild fails with "No text chunks created"
The PDF may contain scanned images only. Run OCR first:
```bash
pip install ocrmypdf
ocrmypdf handbook.pdf handbook.pdf
```

### Rebuild fails offline (embedding model missing)
Ensure the embedding model was downloaded at least once:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
```

### Cannot access from other devices on the LAN
- Confirm the app binds to `0.0.0.0`: the default `app.py` does this.
- Check firewall: `sudo ufw allow 5000`
- Find the Jetson IP: `hostname -I`

### Out-of-memory error on Jetson Nano
- Reduce `EMBED_BATCH` in `Config` to `8`.
- Reduce `BM25_POOL` to `5`.
- Build the index on a dev machine and copy artifacts via `scp`.

### Slow first response
The first query after startup is slower as Ollama loads the model weights into RAM. Subsequent queries are much faster.

---

## Screenshots

> *(Add screenshots after deployment)*

| Screen | Description |
|---|---|
| `docs/main_ui.png` | Full chat interface in browser |
| `docs/sidebar_status.png` | System status and resource monitor |
| `docs/mobile_view.png` | Mobile-responsive layout |
| `docs/pdf_upload.png` | PDF upload flow |
| `docs/rebuild_progress.png` | Index rebuild in progress |
| `docs/answer_citation.png` | Answer with page citation |


## Acknowledgements

- [Flask](https://flask.palletsprojects.com) — lightweight Python web framework
- [Ollama](https://ollama.com) — local LLM serving
- [sentence-transformers](https://www.sbert.net) — `all-MiniLM-L6-v2` embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — efficient vector search
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF text extraction
- [psutil](https://psutil.readthedocs.io) — system resource monitoring
