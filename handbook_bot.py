"""
Campus Handbook Chatbot — Jetson Nano Edition
==============================================
Architecture : Hybrid RAG (BM25 + Dense Retrieval + Re-ranking)
Target HW    : NVIDIA Jetson Nano (4 GB RAM)
LLM          : llama3.2:1b via Ollama (local)
Embeddings   : all-MiniLM-L6-v2  (sentence-transformers)
Vector Index : FAISS IndexFlatIP  (cosine similarity, L2-normalised)
Retrieval    : BM25 candidate pool  →  dense re-rank  →  top_k=1
Mode         : Fully Offline — zero external API calls

Pipeline
--------
PDF  →  page-aware 100-word chunks  →  BM25 index + FAISS index
Query →  BM25 pre-filter (top 10)  →  dense re-rank (top_k=1)
      →  prompt with page citation  →  llama3.2:1b  →  answer

Why Hybrid RAG over Naive RAG?
- Naive RAG (dense-only) fails on exact-keyword queries (acronyms, IDs).
- BM25 catches keyword matches; dense vectors catch semantic similarity.
- Combining both raises recall without adding a second heavy model.
- Re-ranking over a small candidate pool is fast enough for Jetson Nano.

Jetson Nano optimisations
- Embeddings computed in small batches to stay inside 4 GB RAM.
- FAISS IndexFlatIP: lightweight, no GPU training step required.
- Model loaded once; query loop holds it in RAM (no reload per query).
- Logging level configurable via LOG_LEVEL env var.
- Prebuilt index loaded on startup — no rebuild on device.
"""

# ─────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────
import os
import sys
import math
import time
import pickle
import hashlib
import logging
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────────
import fitz                              # PyMuPDF
import numpy as np
import faiss
import ollama
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer


# LOGGING SETUP

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handbook_bot")


# ═════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit only this section
# ═════════════════════════════════════════════════════════════════
class Config:
    # ── Files ──────────────────────────────────────────────────
    PDF_PATH     : str = "handbook.pdf"
    INDEX_PATH   : str = "handbook.index"
    CHUNKS_PATH  : str = "chunks.pkl"

    # ── Chunking ───────────────────────────────────────────────
    CHUNK_WORDS  : int = 100          # mandatory: 100-word chunks
    CHUNK_OVERLAP: int = 20           # sliding overlap keeps context

    # ── Embedding ──────────────────────────────────────────────
    EMBED_MODEL  : str = "all-MiniLM-L6-v2"
    EMBED_BATCH  : int = 16           # small batch → low RAM on Jetson

    # ── Retrieval ──────────────────────────────────────────────
    BM25_POOL    : int = 10           # BM25 candidates before dense re-rank
    TOP_K        : int = 1            # final chunks sent to LLM
    SCORE_THRESH : float = 0.25       # min cosine score to accept a chunk

    # ── LLM ────────────────────────────────────────────────────
    OLLAMA_MODEL : str = "llama3.2:1b"
    OLLAMA_OPTS  : dict = None        # set in __post_init__

    def __post_init__(self):
        self.OLLAMA_OPTS = {
            "num_ctx"    : 2048,   # keep context small for 4 GB RAM
            "temperature": 0.1,    # low temp → factual, deterministic
            "top_p"      : 0.9,
            "repeat_penalty": 1.1,
        }

# Singleton config
cfg = Config()
cfg.__post_init__()


# ═════════════════════════════════════════════════════════════════
# SECTION 1 — PDF UTILITIES
# ═════════════════════════════════════════════════════════════════

def compute_md5(path: str) -> str:
    """MD5 hash of a file — used to detect when the PDF has changed."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Open the PDF and extract (page_number, text) for every page.
    page_number is 1-indexed to match what the user sees in a PDF viewer.
    Raises FileNotFoundError / ValueError for bad inputs.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: '{pdf_path}'")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: '{pdf_path}'")

    log.info("Extracting text from '%s'", pdf_path)
    doc = fitz.open(pdf_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():          # skip blank / image-only pages
            pages.append((i, text))
    doc.close()
    log.info("Extracted %d non-empty pages", len(pages))
    return pages


# ═════════════════════════════════════════════════════════════════
# SECTION 2 — 100-WORD CHUNKING WITH SLIDING OVERLAP
# ═════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """Collapse whitespace, strip control characters."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_pages(
    pages: List[Tuple[int, str]],
    chunk_words: int = cfg.CHUNK_WORDS,
    overlap: int    = cfg.CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split every page into 100-word chunks with `overlap` word overlap.
    Each chunk dict carries:
        text  : str   — the chunk text
        page  : int   — source page number
        chunk_id : int — global chunk index (for BM25 lookup)

    Sliding overlap preserves sentence context across chunk boundaries,
    reducing the chance of splitting a key fact across two chunks.
    """
    chunks: List[Dict] = []
    chunk_id = 0
    step = max(1, chunk_words - overlap)

    for page_num, raw_text in pages:
        words = _clean(raw_text).split()
        if not words:
            continue
        for start in range(0, len(words), step):
            window = words[start : start + chunk_words]
            if len(window) < 10:          # skip tiny tail fragments
                continue
            chunks.append({
                "text"     : " ".join(window),
                "page"     : page_num,
                "chunk_id" : chunk_id,
            })
            chunk_id += 1

    log.info("Created %d chunks (%d-word, %d-word overlap)", len(chunks), chunk_words, overlap)
    return chunks


# ═════════════════════════════════════════════════════════════════
# SECTION 3 — BM25 SPARSE INDEX  (keyword pre-filter)
# ═════════════════════════════════════════════════════════════════

class BM25:
    """
    Lightweight BM25 implementation — no external library needed.
    BM25 scores documents by term frequency (TF) and inverse document
    frequency (IDF), penalising documents that are longer than average.

    Parameters follow the standard Okapi BM25:  k1=1.5, b=0.75
    """
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus_size = len(corpus)
        self.tokenized  = [self._tok(doc) for doc in corpus]
        self.avgdl      = sum(len(d) for d in self.tokenized) / max(1, self.corpus_size)

        # IDF for each unique term
        df: Dict[str, int] = {}
        for doc in self.tokenized:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.idf: Dict[str, float] = {
            t: math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1)
            for t, n in df.items()
        }

    @staticmethod
    def _tok(text: str) -> List[str]:
        """Lowercase, split on non-alphanumeric, drop single chars."""
        return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 1]

    def score(self, query: str, doc_idx: int) -> float:
        q_terms = self._tok(query)
        doc     = self.tokenized[doc_idx]
        dl      = len(doc)
        tf_map: Dict[str, int] = {}
        for term in doc:
            tf_map[term] = tf_map.get(term, 0) + 1
        score = 0.0
        for term in q_terms:
            if term not in self.idf:
                continue
            tf  = tf_map.get(term, 0)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * num / den
        return score

    def get_top_n(self, query: str, n: int) -> List[Tuple[int, float]]:
        """Return (doc_idx, bm25_score) sorted by descending score."""
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


# ═════════════════════════════════════════════════════════════════
# SECTION 4 — EMBEDDINGS + FAISS INDEX
# ═════════════════════════════════════════════════════════════════

def load_embedding_model() -> SentenceTransformer:
    """
    Load all-MiniLM-L6-v2.
    On Jetson Nano the model runs on CPU; sentence-transformers handles
    this automatically when CUDA is unavailable.
    """
    log.info("Loading embedding model '%s'", cfg.EMBED_MODEL)
    model = SentenceTransformer(cfg.EMBED_MODEL, local_files_only=True)
    log.info("Embedding model ready (dim=%d)", model.get_sentence_embedding_dimension())
    return model


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Encode texts in small batches (EMBED_BATCH=16) to control RAM.
    Returns an L2-normalised float32 array of shape (N, 384).
    L2 normalisation turns IndexFlatIP dot-product into cosine similarity.
    """
    log.info("Embedding %d texts (batch=%d)", len(texts), cfg.EMBED_BATCH)
    all_vecs = []
    for start in range(0, len(texts), cfg.EMBED_BATCH):
        batch = texts[start : start + cfg.EMBED_BATCH]
        vecs  = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs).astype("float32")

    # L2-normalise so that dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)   # avoid division by zero
    embeddings /= norms

    log.debug("Embeddings shape: %s", embeddings.shape)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (inner product = cosine, after normalisation).
    IndexFlatIP is brute-force but has zero memory overhead for quantisation
    and is perfectly fast for handbook-scale corpora on Jetson Nano.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("FAISS index built — %d vectors, dim=%d", index.ntotal, dim)
    return index


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Embed and L2-normalise a single query string."""
    vec  = model.encode([query], convert_to_numpy=True).astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ═════════════════════════════════════════════════════════════════
# SECTION 5 — PERSISTENCE  (save / load)
# ═════════════════════════════════════════════════════════════════

def save_artifacts(
    index : faiss.IndexFlatIP,
    chunks: List[Dict],
    pdf_hash: str,
) -> None:
    """Persist FAISS index and chunk metadata to disk."""
    faiss.write_index(index, cfg.INDEX_PATH)
    payload = {"chunks": chunks, "pdf_hash": pdf_hash, "version": 2}
    with open(cfg.CHUNKS_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    log.info("Saved index → '%s', chunks → '%s'", cfg.INDEX_PATH, cfg.CHUNKS_PATH)


def load_artifacts() -> Tuple[faiss.IndexFlatIP, List[Dict], str]:
    """
    Load FAISS index and chunk metadata from disk.
    Returns (index, chunks, pdf_hash).
    Supports both v1 (plain list) and v2 (dict) pickle formats.
    """
    log.info("Loading prebuilt index from disk")
    index = faiss.read_index(cfg.INDEX_PATH)

    with open(cfg.CHUNKS_PATH, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, list):            # legacy v1 format
        chunks   = payload
        pdf_hash = None
    else:                                    # v2 format
        chunks   = payload.get("chunks", [])
        pdf_hash = payload.get("pdf_hash")

    log.info("Loaded %d chunks, %d FAISS vectors", len(chunks), index.ntotal)
    return index, chunks, pdf_hash


def artifacts_exist() -> bool:
    return Path(cfg.INDEX_PATH).exists() and Path(cfg.CHUNKS_PATH).exists()


# ═════════════════════════════════════════════════════════════════
# SECTION 6 — HYBRID RETRIEVAL  (BM25 pool → dense re-rank)
# ═════════════════════════════════════════════════════════════════

def retrieve(
    query       : str,
    embed_model : SentenceTransformer,
    index       : faiss.IndexFlatIP,
    chunks      : List[Dict],
    bm25        : BM25,
    top_k       : int = cfg.TOP_K,
    bm25_pool   : int = cfg.BM25_POOL,
    score_thresh: float = cfg.SCORE_THRESH,
) -> List[Dict]:
    """
    Hybrid retrieval — two stages:

    Stage 1 — BM25 keyword pre-filter
        Scores all chunks by keyword overlap with the query.
        Returns top `bm25_pool` (default 10) candidates.
        This dramatically narrows the dense search space and
        rescues exact-match queries that dense vectors sometimes miss.

    Stage 2 — Dense cosine re-rank
        Embeds the query and the BM25 candidates, computes cosine
        similarity, returns the top `top_k` result(s).
        Dense vectors catch semantic / paraphrase matches.

    Score threshold:
        If the best cosine score is below SCORE_THRESH the function
        returns an empty list, signalling "not found in handbook",
        which prevents the LLM from hallucinating a vague answer.
    """
    if not chunks:
        log.warning("Chunk list is empty — no retrieval possible")
        return []

    # ── Stage 1: BM25 candidate pool ─────────────────────────
    bm25_hits = bm25.get_top_n(query, n=bm25_pool)
    candidate_indices = [idx for idx, _ in bm25_hits]
    candidate_chunks  = [chunks[i] for i in candidate_indices]

    log.debug("BM25 pool: %d candidates", len(candidate_chunks))

    # ── Stage 2: Dense re-rank over the candidate pool ────────
    cand_texts = [c["text"] for c in candidate_chunks]
    cand_embs  = embed_texts(cand_texts, embed_model)   # (pool, 384)

    q_emb      = embed_query(query, embed_model)         # (1, 384)
    scores     = (cand_embs @ q_emb.T).flatten()        # cosine scores

    # Sort by descending cosine score
    ranked = sorted(
        zip(scores, candidate_chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    # Apply score threshold
    results = [
        {**chunk, "score": float(score)}
        for score, chunk in ranked[:top_k]
        if score >= score_thresh
    ]

    if results:
        log.info(
            "Retrieved %d chunk(s) — top score=%.4f, page=%s",
            len(results), results[0]["score"], results[0]["page"],
        )
    else:
        log.info(
            "No chunk passed score threshold (%.2f). Best=%.4f",
            score_thresh, ranked[0][0] if ranked else 0,
        )

    return results


# ═════════════════════════════════════════════════════════════════
# SECTION 7 — PROMPT ENGINEERING
# ═════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise assistant for a college handbook.
Rules you MUST follow:
1. Answer ONLY from the context provided below.
2. If the context does not contain the answer, reply with exactly:
   "I could not find this information in the handbook."
3. Keep answers concise and factual.
4. Always cite the page number like: (Page N).
5. Never guess, infer, or use outside knowledge."""


def build_prompt(question: str, retrieved: List[Dict]) -> str:
    """
    Construct the RAG prompt.
    The system instruction is embedded inside the user turn because
    llama3.2:1b via Ollama's /api/chat already wraps it correctly.
    """
    if not retrieved:
        # No relevant chunk found — skip RAG, let the LLM use the rule
        context_text = "(No relevant section found in the handbook.)"
    else:
        blocks = [
            f"[Page {c['page']} | score={c['score']:.3f}]\n{c['text']}"
            for c in retrieved
        ]
        context_text = "\n\n".join(blocks)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"HANDBOOK CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )
    return prompt


# ═════════════════════════════════════════════════════════════════
# SECTION 8 — LLM CALL  (Ollama — local, offline)
# ═════════════════════════════════════════════════════════════════

def ask_llm(prompt: str) -> str:
    """
    Send the prompt to llama3.2:1b through Ollama.
    Uses Ollama's chat API (not raw generate) for reliable role handling.
    OLLAMA_OPTS (num_ctx=2048, temperature=0.1) are tuned for Jetson Nano:
      - Small context window keeps VRAM / RAM usage low.
      - Low temperature gives deterministic, factual answers.
    """
    try:
        t0 = time.perf_counter()
        response = ollama.chat(
            model=cfg.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=cfg.OLLAMA_OPTS,
        )
        elapsed = time.perf_counter() - t0
        answer  = response["message"]["content"].strip()
        log.info("LLM response in %.2fs (%d chars)", elapsed, len(answer))
        return answer
    except Exception as exc:
        log.error("Ollama error: %s", exc)
        return (
            "⚠  Could not reach the local Ollama model. "
            f"Ensure 'ollama serve' is running and '{cfg.OLLAMA_MODEL}' is pulled.\n"
            f"Details: {exc}"
        )


# ═════════════════════════════════════════════════════════════════
# SECTION 9 — SETUP  (load or build artifacts)
# ═════════════════════════════════════════════════════════════════

def setup() -> Tuple[SentenceTransformer, faiss.IndexFlatIP, List[Dict], BM25]:
    """
    Full initialisation sequence:
    1. Load embedding model.
    2. If prebuilt artifacts exist AND the PDF hasn't changed → load them.
    3. Otherwise, extract PDF → chunk → embed → build index → save.
    4. Build BM25 in RAM (fast, no persistence needed).
    Returns (embed_model, faiss_index, chunks, bm25).
    """
    embed_model = load_embedding_model()

    # ── Check if we can reuse prebuilt artifacts ──────────────
    rebuild = True
    if artifacts_exist():
        index, chunks, saved_hash = load_artifacts()
        if saved_hash and Path(cfg.PDF_PATH).exists():
            current_hash = compute_md5(cfg.PDF_PATH)
            if saved_hash == current_hash and chunks:
                log.info("Prebuilt index matches current PDF — skipping rebuild")
                rebuild = False
            else:
                log.info("PDF changed (hash mismatch) — rebuilding index")
        elif not Path(cfg.PDF_PATH).exists():
            # On Jetson the PDF may not be present; load index as-is
            if chunks:
                log.info("PDF absent — loading prebuilt index (read-only mode)")
                rebuild = False

    # ── Build from scratch ────────────────────────────────────
    if rebuild:
        if not Path(cfg.PDF_PATH).exists():
            raise FileNotFoundError(
                f"Cannot build index: PDF '{cfg.PDF_PATH}' not found.\n"
                "Build the index on a development machine and copy "
                "handbook.index + chunks.pkl to the Jetson."
            )
        pdf_hash = compute_md5(cfg.PDF_PATH)
        pages    = extract_pages(cfg.PDF_PATH)
        chunks   = chunk_pages(pages)

        if not chunks:
            raise RuntimeError(
                "No text chunks created. The PDF may be scanned images. "
                "Run OCR on it first (e.g. ocrmypdf)."
            )

        texts      = [c["text"] for c in chunks]
        embeddings = embed_texts(texts, embed_model)
        index      = build_faiss_index(embeddings)
        save_artifacts(index, chunks, pdf_hash)

    # ── BM25 always built in RAM (fast) ───────────────────────
    log.info("Building BM25 index in RAM")
    bm25 = BM25([c["text"] for c in chunks])
    log.info("BM25 ready")

    return embed_model, index, chunks, bm25


# ═════════════════════════════════════════════════════════════════
# SECTION 10 — SINGLE QUERY  (for scripted / batch use)
# ═════════════════════════════════════════════════════════════════

def query(
    question    : str,
    embed_model : SentenceTransformer,
    index       : faiss.IndexFlatIP,
    chunks      : List[Dict],
    bm25        : BM25,
    debug       : bool = False,
) -> str:
    """
    Run one query through the full Hybrid RAG pipeline.
    Returns the LLM answer string.
    Set debug=True to print retrieved chunk text to stdout.
    """
    log.info("Query: %s", question)

    retrieved = retrieve(question, embed_model, index, chunks, bm25)

    # ── Debug: print retrieved chunks ─────────────────────────
    if debug:
        if retrieved:
            print("\n── Retrieved Chunks ──────────────────────────")
            for i, c in enumerate(retrieved, 1):
                print(f"[{i}] Page {c['page']} | score={c['score']:.4f}")
                print(c["text"][:300])
                print("──────────────────────────────────────────────")
        else:
            print("\n[DEBUG] No chunks passed the score threshold.")

    prompt = build_prompt(question, retrieved)
    answer = ask_llm(prompt)
    return answer


# ═════════════════════════════════════════════════════════════════
# SECTION 11 — INTERACTIVE CLI LOOP
# ═════════════════════════════════════════════════════════════════

def chat_loop(
    embed_model : SentenceTransformer,
    index       : faiss.IndexFlatIP,
    chunks      : List[Dict],
    bm25        : BM25,
    debug       : bool = False,
) -> None:
    """
    Interactive terminal loop.
    Commands:
        quit / exit / q  — exit
        !debug           — toggle chunk debug output
        !info            — print index stats
    """
    print("\n" + "═" * 58)
    print("  Campus Handbook Chatbot  ·  Jetson Nano Edition")
    print(f"  Model: {cfg.OLLAMA_MODEL}  ·  Chunks: {len(chunks)}")
    print("  Type 'quit' to exit  ·  '!debug' to toggle debug")
    print("═" * 58 + "\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        # ── Built-in commands ─────────────────────────────────
        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if raw.lower() == "!debug":
            debug = not debug
            print(f"[System] Debug mode: {'ON' if debug else 'OFF'}")
            continue

        if raw.lower() == "!info":
            print(
                f"[System] Chunks={len(chunks)} | "
                f"FAISS vectors={index.ntotal} | "
                f"Model={cfg.OLLAMA_MODEL}"
            )
            continue

        # ── Normal query ──────────────────────────────────────
        answer = query(raw, embed_model, index, chunks, bm25, debug=debug)
        print(f"\nBot: {answer}\n")


# ═════════════════════════════════════════════════════════════════
# SECTION 12 — ENTRY POINT
# ═════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Campus Handbook Chatbot — Jetson Nano Offline RAG"
    )
    p.add_argument("--pdf",       default=cfg.PDF_PATH,   help="Path to handbook PDF")
    p.add_argument("--index",     default=cfg.INDEX_PATH, help="Path to FAISS index")
    p.add_argument("--chunks",    default=cfg.CHUNKS_PATH,help="Path to chunks pickle")
    p.add_argument("--top-k",     type=int, default=cfg.TOP_K, help="Chunks to retrieve")
    p.add_argument("--debug",     action="store_true",     help="Print retrieved chunks")
    p.add_argument("--rebuild",   action="store_true",     help="Force index rebuild")
    p.add_argument("--query",     type=str, default=None,  help="Single query then exit")
    p.add_argument("--log-level", default=LOG_LEVEL,       help="DEBUG/INFO/WARNING")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Apply CLI overrides to config
    cfg.PDF_PATH    = args.pdf
    cfg.INDEX_PATH  = args.index
    cfg.CHUNKS_PATH = args.chunks
    cfg.TOP_K       = args.top_k
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.rebuild:
        for p in (cfg.INDEX_PATH, cfg.CHUNKS_PATH):
            if Path(p).exists():
                Path(p).unlink()
                log.info("Deleted '%s' for forced rebuild", p)

    try:
        embed_model, index, chunks, bm25 = setup()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if args.query:
        # Non-interactive single-shot mode
        answer = query(args.query, embed_model, index, chunks, bm25, debug=args.debug)
        print(answer)
    else:
        chat_loop(embed_model, index, chunks, bm25, debug=args.debug)


if __name__ == "__main__":
    main()
