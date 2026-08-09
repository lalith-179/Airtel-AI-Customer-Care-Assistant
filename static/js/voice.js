(() => {
  "use strict";

  // ------------------------------------------------------------------
  // DOM references
  // ------------------------------------------------------------------
  const messagesEl = document.getElementById("messages");
  const emptyStateEl = document.getElementById("emptyState");
  const conversationEl = document.getElementById("conversation");
  const composerForm = document.getElementById("composerForm");
  const textInput = document.getElementById("textInput");
  const sendBtn = document.getElementById("sendBtn");
  const micBtn = document.getElementById("micBtn");
  const voiceStage = document.getElementById("voiceStage");
  const stopRecordingBtn = document.getElementById("stopRecordingBtn");
  const waveformCanvas = document.getElementById("waveformCanvas");
  const canvasCtx = waveformCanvas.getContext("2d");
  const pipelineStatus = document.getElementById("pipelineStatus");
  const pipelineStatusText = document.getElementById("pipelineStatusText");
  const responseAudio = document.getElementById("responseAudio");
  const exampleChips = document.getElementById("exampleChips");

  const kbStatusPill = document.getElementById("kbStatusPill");
  const kbDot = document.getElementById("kbDot");
  const kbStatusText = document.getElementById("kbStatusText");
  const adminToggle = document.getElementById("adminToggle");
  const adminDrawer = document.getElementById("adminDrawer");

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  let conversationId = null;
  let socket = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let audioContext = null;
  let analyser = null;
  let micStream = null;
  let animationFrameId = null;

  // ------------------------------------------------------------------
  // Utilities
  // ------------------------------------------------------------------
  function ensureConversationId() {
    if (!conversationId) {
      conversationId = crypto.randomUUID ? crypto.randomUUID() : `local-${Date.now()}`;
    }
    return conversationId;
  }

  function hideEmptyState() {
    if (emptyStateEl) emptyStateEl.style.display = "none";
  }

  function scrollToBottom() {
    conversationEl.scrollTop = conversationEl.scrollHeight;
  }

  function setPipelineStatus(text) {
    if (!text) {
      pipelineStatus.hidden = true;
      return;
    }
    pipelineStatus.hidden = false;
    pipelineStatusText.textContent = text;
  }

  function addMessage({ role, text, meta }) {
    hideEmptyState();
    const row = document.createElement("div");
    row.className = `msg-row ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.textContent = text;
    row.appendChild(bubble);

    if (meta) {
      const metaEl = document.createElement("div");
      metaEl.className = "msg-meta";

      if (meta.language) {
        const langBadge = document.createElement("span");
        langBadge.className = "badge badge-lang";
        langBadge.textContent = meta.language.toUpperCase();
        metaEl.appendChild(langBadge);
      }

      if (typeof meta.grounded === "boolean") {
        const groundedBadge = document.createElement("span");
        groundedBadge.className = `badge ${meta.grounded ? "badge-grounded" : "badge-escalate"}`;
        groundedBadge.textContent = meta.grounded
          ? `grounded ${(meta.confidence * 100).toFixed(0)}%`
          : "low confidence";
        metaEl.appendChild(groundedBadge);
      }

      if (meta.should_escalate) {
        const escBadge = document.createElement("span");
        escBadge.className = "badge badge-escalate";
        escBadge.textContent = "escalate to support";
        metaEl.appendChild(escBadge);
      }

      if (meta.audioBase64) {
        const playBtn = document.createElement("button");
        playBtn.className = "play-btn";
        playBtn.setAttribute("aria-label", "Play response audio");
        playBtn.innerHTML = '<svg viewBox="0 0 24 24" width="12" height="12"><path fill="currentColor" d="M8 5v14l11-7z"/></svg>';
        playBtn.addEventListener("click", () => playAudioBase64(meta.audioBase64));
        metaEl.appendChild(playBtn);
      }

      row.appendChild(metaEl);

      if (meta.sources && meta.sources.length) {
        const details = document.createElement("details");
        details.className = "sources-block";
        const summary = document.createElement("summary");
        summary.textContent = `Sources (${meta.sources.length})`;
        details.appendChild(summary);
        const ul = document.createElement("ul");
        meta.sources.forEach((s) => {
          const li = document.createElement("li");
          li.textContent = s.title || s.url || "Untitled source";
          ul.appendChild(li);
        });
        details.appendChild(ul);
        row.appendChild(details);
      }
    }

    messagesEl.appendChild(row);
    scrollToBottom();
    return row;
  }

  function addTypingIndicator() {
    hideEmptyState();
    const row = document.createElement("div");
    row.className = "msg-row assistant";
    row.id = "typingIndicatorRow";
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.innerHTML = '<span class="typing-indicator"><span></span><span></span><span></span></span>';
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const row = document.getElementById("typingIndicatorRow");
    if (row) row.remove();
  }

  function playAudioBase64(base64) {
    if (!base64) return;
    const src = `data:audio/wav;base64,${base64}`;
    responseAudio.src = src;
    responseAudio.play().catch((err) => console.warn("Audio playback blocked:", err));
  }

  // ------------------------------------------------------------------
  // Status / admin drawer
  // ------------------------------------------------------------------
  async function loadStatus() {
    try {
      const resp = await fetch("/api/status");
      const data = await resp.json();

      const kbReady = data.knowledge_base && data.knowledge_base.ready;
      kbDot.className = `dot ${kbReady ? "dot-ready" : "dot-error"}`;
      kbStatusText.textContent = kbReady
        ? `Knowledge base ready · ${data.knowledge_base.chunk_count} chunks`
        : "Knowledge base not built";

      document.getElementById("adminOllama").textContent = data.ollama_available ? "Connected" : "Unavailable";
      document.getElementById("adminLlm").textContent = data.llm_model || "—";
      document.getElementById("adminEmbed").textContent = data.embedding_model || "—";
      document.getElementById("adminStt").textContent = data.stt_ready ? "faster-whisper (ready)" : "not loaded";
      document.getElementById("adminTts").textContent = data.tts_ready ? "Piper (ready)" : "not loaded";
      document.getElementById("adminCollection").textContent = data.knowledge_base.collection || "—";
      document.getElementById("adminChunks").textContent = data.knowledge_base.chunk_count ?? "0";
      document.getElementById("adminLastIndexed").textContent = data.knowledge_base.last_indexed || "unknown";
    } catch (err) {
      kbDot.className = "dot dot-error";
      kbStatusText.textContent = "Status unavailable";
      console.error("Failed to load status:", err);
    }
  }

  adminToggle.addEventListener("click", () => {
    adminDrawer.hidden = !adminDrawer.hidden;
  });
  kbStatusPill.addEventListener("click", () => {
    adminDrawer.hidden = !adminDrawer.hidden;
  });

  // ------------------------------------------------------------------
  // Text chat
  // ------------------------------------------------------------------
  async function sendTextMessage(text) {
    if (!text.trim()) return;
    addMessage({ role: "user", text });
    textInput.value = "";
    sendBtn.disabled = true;
    addTypingIndicator();
    setPipelineStatus("Thinking…");

    try {
      const resp = await fetch("/api/chat/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversation_id: ensureConversationId(),
          synthesize_speech: false,
        }),
      });
      const data = await resp.json();
      removeTypingIndicator();
      setPipelineStatus(null);

      if (!resp.ok) {
        addMessage({ role: "assistant", text: data.message || "Something went wrong. Please try again." });
        return;
      }

      conversationId = data.conversation_id || conversationId;
      addMessage({
        role: "assistant",
        text: data.response,
        meta: {
          language: data.language,
          grounded: data.grounded,
          confidence: data.confidence,
          should_escalate: data.should_escalate,
          sources: data.sources,
        },
      });
    } catch (err) {
      removeTypingIndicator();
      setPipelineStatus(null);
      addMessage({ role: "assistant", text: "Network error - please check your connection and try again." });
      console.error(err);
    } finally {
      sendBtn.disabled = false;
    }
  }

  composerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendTextMessage(textInput.value);
  });

  exampleChips.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    sendTextMessage(chip.dataset.text);
  });

  // ------------------------------------------------------------------
  // WebSocket (voice)
  // ------------------------------------------------------------------
  function getSocket() {
    if (!socket) {
      socket = io();

      socket.on("status", (payload) => {
        setPipelineStatus(payload.message || payload.stage);
      });

      socket.on("transcript", (payload) => {
        setPipelineStatus("Thinking…");
        addMessage({ role: "user", text: payload.text, meta: { language: payload.language } });
        addTypingIndicator();
      });

      socket.on("voice_response", (data) => {
        removeTypingIndicator();
        setPipelineStatus(null);
        conversationId = data.conversation_id || conversationId;
        addMessage({
          role: "assistant",
          text: data.response,
          meta: {
            language: data.language,
            grounded: data.grounded,
            confidence: data.confidence,
            should_escalate: data.should_escalate,
            sources: data.sources,
            audioBase64: data.audio_base64,
          },
        });
        if (data.audio_base64) {
          playAudioBase64(data.audio_base64);
        }
      });

      socket.on("error", (payload) => {
        removeTypingIndicator();
        setPipelineStatus(null);
        addMessage({ role: "assistant", text: payload.message || "Something went wrong." });
      });
    }
    return socket;
  }

  // ------------------------------------------------------------------
  // Microphone capture + live waveform (Web Audio API)
  // ------------------------------------------------------------------
  function drawIdle() {
    canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  }

  function drawWaveformFrame(dataArray, bufferLength) {
    const w = waveformCanvas.width;
    const h = waveformCanvas.height;
    canvasCtx.clearRect(0, 0, w, h);

    const barCount = 48;
    const step = Math.floor(bufferLength / barCount) || 1;
    const barWidth = w / barCount;
    const centerY = h / 2;

    for (let i = 0; i < barCount; i++) {
      const value = dataArray[i * step] / 255; // 0..1
      const barHeight = Math.max(3, value * h * 0.9);
      const x = i * barWidth;

      const gradient = canvasCtx.createLinearGradient(0, centerY - barHeight / 2, 0, centerY + barHeight / 2);
      gradient.addColorStop(0, "#ff2f5c");
      gradient.addColorStop(1, "#e4002b");
      canvasCtx.fillStyle = gradient;

      const radius = Math.min(3, barWidth / 2 - 1);
      roundRect(canvasCtx, x + 1.5, centerY - barHeight / 2, barWidth - 3, barHeight, radius);
      canvasCtx.fill();
    }

    animationFrameId = requestAnimationFrame(() => tickWaveform());
  }

  function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
  }

  let freqData = null;
  function tickWaveform() {
    if (!analyser) return;
    if (!freqData) freqData = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(freqData);
    drawWaveformFrame(freqData, freqData.length);
  }

  async function startRecording() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      addMessage({ role: "assistant", text: "Microphone permission was denied. Please allow microphone access to use voice input." });
      return;
    }

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(micStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    voiceStage.hidden = false;
    micBtn.classList.add("recording");
    setPipelineStatus("Listening…");
    tickWaveform();

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";
    mediaRecorder = new MediaRecorder(micStream, { mimeType });
    audioChunks = [];

    mediaRecorder.addEventListener("dataavailable", (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    });

    mediaRecorder.addEventListener("stop", handleRecordingStop);
    mediaRecorder.start();
  }

  function stopMicTracks() {
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (audioContext) {
      audioContext.close().catch(() => {});
      audioContext = null;
    }
    analyser = null;
    freqData = null;
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
    drawIdle();
  }

  async function handleRecordingStop() {
    stopMicTracks();
    voiceStage.hidden = true;
    micBtn.classList.remove("recording");

    const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
    if (blob.size === 0) {
      setPipelineStatus(null);
      return;
    }

    setPipelineStatus("Transcribing…");
    const base64 = await blobToBase64(blob);
    const sock = getSocket();
    sock.emit("voice_message", {
      conversation_id: ensureConversationId(),
      audio_base64: base64,
      mime_type: blob.type,
    });
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result;
        const base64 = result.split(",")[1] || "";
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    } else {
      stopMicTracks();
      voiceStage.hidden = true;
      micBtn.classList.remove("recording");
    }
  }

  micBtn.addEventListener("click", () => {
    if (micBtn.classList.contains("recording")) {
      stopRecording();
    } else {
      startRecording();
    }
  });
  stopRecordingBtn.addEventListener("click", stopRecording);

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  loadStatus();
  setInterval(loadStatus, 30000);
})();
