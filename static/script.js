/**
 * Campus Handbook Guide — script.js
 * Handles: chat UI, /ask, /upload, /rebuild, /status polling, /clear, export
 * Zero external dependencies — vanilla JS only (Jetson-friendly, fully offline)
 */

"use strict";

/* ════════════════════════════════════════════════════════════════
   DOM REFS
════════════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);

const chatMessages   = $("chatMessages");
const questionInput  = $("questionInput");
const sendBtn        = $("sendBtn");
const charCount      = $("charCount");
const lastRespTime   = $("lastResponseTime");
const welcomeCard    = $("welcomeCard");
const fileInput      = $("fileInput");
const uploadProgress = $("uploadProgress");
const progressFill   = $("progressFill");
const progressLabel  = $("progressLabel");
const rebuildBtn     = $("rebuildBtn");
const rebuildStatus  = $("rebuildStatus");
const rebuildLabel   = $("rebuildLabel");
const clearBtn       = $("clearBtn");
const exportBtn      = $("exportBtn");
const sidebar        = $("sidebar");
const sidebarOverlay = $("sidebarOverlay");
const hamburger      = $("hamburger");
const toastContainer = $("toastContainer");

/* Status elements */
const ollamaStatus  = $("ollamaStatus");
const embedStatus   = $("embedStatus");
const faissStatus   = $("faissStatus");
const chunksCount   = $("chunksCount");
const ramValue      = $("ramValue");
const ramBar        = $("ramBar");
const cpuValue      = $("cpuValue");
const cpuBar        = $("cpuBar");
const modelLabel    = $("modelLabel");
const pdfName       = $("pdfName");
const pdfSize       = $("pdfSize");
const systemDot     = $("systemDot");
const systemFooter  = $("systemFooter");
const mobileStatusDot = $("mobileStatusDot");

/* ════════════════════════════════════════════════════════════════
   APP STATE
════════════════════════════════════════════════════════════════ */
let isWaiting    = false;   // waiting for /ask response
let isRebuilding = false;   // rebuild in progress
let chatHistory  = [];      // local copy {role, text, citation, ts}
let statusTimer  = null;
let rebuildTimer = null;

/* ════════════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
════════════════════════════════════════════════════════════════ */
function toast(msg, type = "info", duration = 4000) {
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  toastContainer.appendChild(el);

  setTimeout(() => {
    el.style.animation = "fadeOut 0.3s ease forwards";
    setTimeout(() => el.remove(), 300);
  }, duration);
}

/* ════════════════════════════════════════════════════════════════
   SYSTEM STATUS POLLING  (every 4 s)
════════════════════════════════════════════════════════════════ */
async function pollStatus() {
  try {
    const res  = await fetch("/status");
    const data = await res.json();
    applyStatus(data);
  } catch (_) {
    // Server unreachable — don't crash
    setFooterState("error", "Server unreachable");
  }
}

function applyStatus(d) {
  /* Badge helper */
  const badge = (el, ok, loadingText, okText, errText) => {
    el.textContent = ok ? okText : (d.rebuilding ? loadingText : errText);
    el.className   = "status-badge " + (ok ? "ok" : (d.rebuilding ? "loading" : "error"));
  };

  badge(ollamaStatus, d.ollama_ok,  "Checking…", "Online",  "Offline");
  badge(embedStatus,  d.embed_ok,   "Loading…",  "Loaded",  "Not loaded");

  const faissOk = d.faiss_vectors > 0;
  badge(faissStatus, faissOk, "Building…", `${d.faiss_vectors} vecs`, "No index");

  chunksCount.textContent = d.chunks_count > 0 ? d.chunks_count : "—";

  /* PDF info */
  if (d.pdf_name) {
    pdfName.textContent = d.pdf_name;
    pdfSize.textContent = `${d.pdf_size_kb} KB`;
  } else {
    pdfName.textContent = "No PDF loaded";
    pdfSize.textContent = "—";
  }

  /* RAM / CPU */
  if (d.ram_pct !== null) {
    ramValue.textContent = `${d.ram_used_gb} / ${d.ram_total_gb} GB  (${d.ram_pct}%)`;
    ramBar.style.width   = `${Math.min(d.ram_pct, 100)}%`;
    // Colour shifts: green → orange → red
    const hue = d.ram_pct < 60 ? "#58a6ff" : d.ram_pct < 85 ? "#f0883e" : "#f85149";
    ramBar.style.background = hue;
  } else {
    ramValue.textContent = "psutil unavailable";
    ramBar.style.width   = "0%";
  }

  if (d.cpu_pct !== null) {
    cpuValue.textContent = `${d.cpu_pct}%`;
    cpuBar.style.width   = `${Math.min(d.cpu_pct, 100)}%`;
    const hue = d.cpu_pct < 60 ? "#3fb950" : d.cpu_pct < 85 ? "#f0883e" : "#f85149";
    cpuBar.style.background = hue;
  } else {
    cpuValue.textContent = "—";
  }

  /* Model label */
  modelLabel.textContent = `Model: ${d.ollama_model || "—"}`;

  /* Footer dot + text */
  if (d.rebuilding) {
    setFooterState("rebuilding", "Rebuilding index…");
    isRebuilding = true;
  } else if (d.ready) {
    setFooterState("ready", `Ready · ${d.chunks_count} chunks`);
    if (isRebuilding) {
      // Just finished rebuilding
      isRebuilding = false;
      hideRebuildStatus();
      toast(`Index ready — ${d.chunks_count} chunks indexed.`, "success");
      rebuildBtn.disabled = false;
    }
  } else if (d.error) {
    setFooterState("error", "Error — check logs");
    if (isRebuilding) {
      isRebuilding = false;
      hideRebuildStatus();
      toast(`Rebuild failed: ${d.error}`, "error", 7000);
      rebuildBtn.disabled = false;
    }
  } else {
    setFooterState("loading", "Initialising…");
  }
}

function setFooterState(state, text) {
  systemDot.className    = `footer-dot ${state}`;
  mobileStatusDot.className = `mobile-status-dot ${state}`;
  systemFooter.textContent = text;
}

function hideRebuildStatus() {
  rebuildStatus.classList.add("hidden");
  rebuildLabel.textContent = "Rebuilding…";
}

/* ════════════════════════════════════════════════════════════════
   CHAT — MESSAGE RENDERING
════════════════════════════════════════════════════════════════ */
function hideWelcome() {
  if (welcomeCard && !welcomeCard.classList.contains("hidden")) {
    welcomeCard.classList.add("hidden");
  }
}

function scrollBottom() {
  const win = $("chatWindow");
  win.scrollTo({ top: win.scrollHeight, behavior: "smooth" });
}

function addUserBubble(text) {
  const ts  = now();
  const wrap = document.createElement("div");
  wrap.className = "msg-wrap msg-user";
  wrap.innerHTML = `
    <div class="msg-bubble">${escHtml(text)}</div>
    <div class="msg-meta">${ts}</div>`;
  chatMessages.appendChild(wrap);
  scrollBottom();
}

function addBotBubble(text, citation) {
  const ts   = now();
  const wrap  = document.createElement("div");
  wrap.className = "msg-wrap msg-bot";
  wrap.innerHTML = `
    <div class="msg-bubble">${escHtml(text)}</div>
    ${citation ? `<div class="msg-citation">${escHtml(citation)}</div>` : ""}
    <div class="msg-meta">${ts}</div>`;
  chatMessages.appendChild(wrap);
  scrollBottom();
}

function addTypingIndicator() {
  const wrap = document.createElement("div");
  wrap.className = "msg-wrap msg-bot";
  wrap.id = "typingWrap";
  wrap.innerHTML = `
    <div class="typing-indicator">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>`;
  chatMessages.appendChild(wrap);
  scrollBottom();
  return wrap;
}

function removeTyping() {
  const el = $("typingWrap");
  if (el) el.remove();
}

function addSystemNote(text) {
  const el = document.createElement("div");
  el.className = "system-note";
  el.textContent = text;
  chatMessages.appendChild(el);
  scrollBottom();
}

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ════════════════════════════════════════════════════════════════
   SEND QUESTION
════════════════════════════════════════════════════════════════ */
async function sendQuestion() {
  const text = questionInput.value.trim();
  if (!text || isWaiting) return;

  hideWelcome();
  addUserBubble(text);
  chatHistory.push({ role: "user", text, citation: "", ts: now() });

  questionInput.value = "";
  autoResize();
  updateCharCount();

  isWaiting = true;
  sendBtn.disabled = true;
  const typingWrap = addTypingIndicator();

  try {
    const t0  = performance.now();
    const res = await fetch("/ask", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ question: text }),
    });

    const data = await res.json();
    removeTyping();

    if (!res.ok || data.error) {
      const err = data.error || `Server error (${res.status})`;
      addBotBubble(`⚠ ${err}`, "");
    } else {
      const elapsed = data.elapsed
        ? `${data.elapsed}s`
        : `${((performance.now() - t0) / 1000).toFixed(2)}s`;

      addBotBubble(data.answer, data.citation);
      lastRespTime.textContent = `Last response: ${elapsed}`;
      chatHistory.push({
        role: "assistant",
        text: data.answer,
        citation: data.citation,
        ts: now(),
      });
    }
  } catch (err) {
    removeTyping();
    addBotBubble(`⚠ Network error: ${err.message}`, "");
  } finally {
    isWaiting = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

/* ════════════════════════════════════════════════════════════════
   INPUT — auto-resize textarea + char counter
════════════════════════════════════════════════════════════════ */
function autoResize() {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 180) + "px";
}

function updateCharCount() {
  const len = questionInput.value.length;
  charCount.textContent = `${len} / 2000`;
  charCount.style.color = len > 1800 ? "var(--fg-error)" : "var(--fg-secondary)";
}

questionInput.addEventListener("input", () => {
  autoResize();
  updateCharCount();
});

questionInput.addEventListener("keydown", e => {
  // Enter sends; Shift+Enter inserts newline
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

sendBtn.addEventListener("click", sendQuestion);

/* ════════════════════════════════════════════════════════════════
   SUGGESTION CHIPS
════════════════════════════════════════════════════════════════ */
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    questionInput.value = chip.dataset.q;
    autoResize();
    updateCharCount();
    questionInput.focus();
  });
});

/* ════════════════════════════════════════════════════════════════
   PDF UPLOAD
════════════════════════════════════════════════════════════════ */
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  // Show progress UI
  uploadProgress.classList.remove("hidden");
  progressFill.style.width = "0%";
  progressLabel.textContent = `Uploading ${file.name}…`;
  $("uploadLabel").style.pointerEvents = "none";
  $("uploadLabel").style.opacity = "0.6";

  const formData = new FormData();
  formData.append("pdf", file);

  // Fake progress while XHR runs (true progress needs XHR, not fetch)
  let fakeProgress = 0;
  const fakeTimer = setInterval(() => {
    fakeProgress = Math.min(fakeProgress + 8, 85);
    progressFill.style.width = fakeProgress + "%";
  }, 120);

  try {
    const res  = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();

    clearInterval(fakeTimer);
    progressFill.style.width = "100%";

    if (!res.ok || data.error) {
      progressLabel.textContent = `Upload failed: ${data.error}`;
      toast(`Upload failed: ${data.error}`, "error");
    } else {
      progressLabel.textContent = `✓ ${data.filename} uploaded (${data.size_kb} KB)`;
      toast(data.message, "success");
      addSystemNote(`📄 PDF uploaded: ${data.filename}`);
    }

    setTimeout(() => {
      uploadProgress.classList.add("hidden");
      $("uploadLabel").style.pointerEvents = "";
      $("uploadLabel").style.opacity = "";
      fileInput.value = "";
    }, 2500);

  } catch (err) {
    clearInterval(fakeTimer);
    progressLabel.textContent = `Upload failed: ${err.message}`;
    toast(`Upload error: ${err.message}`, "error");
    setTimeout(() => {
      uploadProgress.classList.add("hidden");
      $("uploadLabel").style.pointerEvents = "";
      $("uploadLabel").style.opacity = "";
    }, 3000);
  }
});

/* ════════════════════════════════════════════════════════════════
   REBUILD INDEX
════════════════════════════════════════════════════════════════ */
rebuildBtn.addEventListener("click", async () => {
  if (isRebuilding) return;

  try {
    const res  = await fetch("/rebuild", { method: "POST" });
    const data = await res.json();

    if (!res.ok) {
      toast(data.error || "Rebuild request failed.", "error");
      return;
    }

    // Accepted (202)
    isRebuilding = true;
    rebuildBtn.disabled = true;
    rebuildStatus.classList.remove("hidden");
    rebuildLabel.textContent = "Building BM25 + FAISS index…";
    addSystemNote("🔄 Index rebuild started…");
    toast("Index rebuild started. This may take a minute.", "info");

    // Status polling will detect completion automatically
  } catch (err) {
    toast(`Rebuild request error: ${err.message}`, "error");
  }
});

/* ════════════════════════════════════════════════════════════════
   CLEAR CHAT
════════════════════════════════════════════════════════════════ */
clearBtn.addEventListener("click", async () => {
  if (!chatHistory.length && welcomeCard && !welcomeCard.classList.contains("hidden")) return;
  if (!confirm("Clear all chat messages?")) return;

  try {
    await fetch("/clear", { method: "POST" });
  } catch (_) { /* ignore — clear locally regardless */ }

  chatHistory = [];

  // Remove all messages except welcome card
  [...chatMessages.children].forEach(child => {
    if (child.id !== "welcomeCard") child.remove();
  });

  if (welcomeCard) welcomeCard.classList.remove("hidden");
  lastRespTime.textContent = "";
  toast("Chat cleared.", "info", 2000);
});

/* ════════════════════════════════════════════════════════════════
   EXPORT CHAT  (download as Markdown)
════════════════════════════════════════════════════════════════ */
exportBtn.addEventListener("click", () => {
  if (!chatHistory.length) {
    toast("Nothing to export yet.", "warn");
    return;
  }

  const lines = [
    "# Campus Handbook Guide — Chat Export",
    `Exported: ${new Date().toLocaleString()}`,
    "",
    "---",
    "",
  ];

  chatHistory.forEach(({ role, text, citation }) => {
    if (role === "user") {
      lines.push(`**You:** ${text}`, "");
    } else {
      lines.push(`**Assistant:** ${text}`);
      if (citation) lines.push(`*${citation}*`);
      lines.push("");
    }
  });

  const blob     = new Blob([lines.join("\n")], { type: "text/markdown" });
  const url      = URL.createObjectURL(blob);
  const a        = document.createElement("a");
  const ts       = new Date().toISOString().slice(0, 16).replace("T", "_").replace(":", "-");
  a.href         = url;
  a.download     = `chat_export_${ts}.md`;
  a.click();
  URL.revokeObjectURL(url);
  toast("Chat exported as Markdown.", "success", 2500);
});

/* ════════════════════════════════════════════════════════════════
   MOBILE SIDEBAR TOGGLE
════════════════════════════════════════════════════════════════ */
hamburger.addEventListener("click", () => {
  sidebar.classList.toggle("open");
  sidebarOverlay.classList.toggle("hidden");
});

sidebarOverlay.addEventListener("click", () => {
  sidebar.classList.remove("open");
  sidebarOverlay.classList.add("hidden");
});

/* ════════════════════════════════════════════════════════════════
   INIT
════════════════════════════════════════════════════════════════ */
(function init() {
  // Kick off status polling immediately, then every 4 seconds
  pollStatus();
  statusTimer = setInterval(pollStatus, 4000);

  // Focus input on load
  questionInput.focus();
})();
