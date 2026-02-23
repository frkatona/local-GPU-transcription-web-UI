const state = {
  file: null,
  jobId: null,
  pollTimer: null,
  localAudioUrl: null,
  segments: [],
  lineElements: [],
  activeLineIndex: -1,
};

const fileInput = document.getElementById("fileInput");
const localAudioInput = document.getElementById("localAudioInput");
const localSrtInput = document.getElementById("localSrtInput");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const uploadForm = document.getElementById("uploadForm");
const modelSelect = document.getElementById("modelSelect");
const exportFolder = document.getElementById("exportFolder");
const startBtn = document.getElementById("startBtn");
const loadExistingBtn = document.getElementById("loadExistingBtn");
const localPairName = document.getElementById("localPairName");
const statusLabel = document.getElementById("statusLabel");
const statusPercent = document.getElementById("statusPercent");
const statusMessage = document.getElementById("statusMessage");
const progressBar = document.getElementById("progressBar");
const resultPanel = document.getElementById("resultPanel");
const downloadTxt = document.getElementById("downloadTxt");
const downloadSrt = document.getElementById("downloadSrt");
const metaStatus = document.getElementById("metaStatus");
const metaDevice = document.getElementById("metaDevice");
const metaLanguage = document.getElementById("metaLanguage");
const metaSegments = document.getElementById("metaSegments");
const audioPlayer = document.getElementById("audioPlayer");
const transcriptList = document.getElementById("transcriptList");
const currentTextBox = document.getElementById("currentTextBox");

const busyStates = new Set(["queued", "running"]);

const formatTimestamp = (seconds) => {
  const total = Math.floor(Number(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
};

const formatElapsed = (seconds) => {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value <= 0) {
    return null;
  }
  if (value < 60) {
    return `${value.toFixed(1)}s`;
  }
  const mins = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  return `${mins}m ${String(secs).padStart(2, "0")}s`;
};

const setProgress = (fraction) => {
  const clamped = Math.max(0, Math.min(1, Number(fraction) || 0));
  progressBar.style.width = `${clamped * 100}%`;
  statusPercent.textContent = `${Math.round(clamped * 100)}%`;
};

const setStatus = (label, message, progress = null) => {
  statusLabel.textContent = label;
  statusMessage.textContent = message;
  if (progress !== null) {
    setProgress(progress);
  }
};

const clearPolling = () => {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
};

const clearLocalAudioObjectUrl = () => {
  if (state.localAudioUrl) {
    URL.revokeObjectURL(state.localAudioUrl);
    state.localAudioUrl = null;
  }
};

const setFile = (file) => {
  state.file = file;
  fileName.textContent = file ? file.name : "No file selected";
};

const setBusy = (busy) => {
  startBtn.disabled = busy;
  loadExistingBtn.disabled = busy;
  startBtn.textContent = busy ? "Working..." : "Start Transcription";
};

const resetTranscript = () => {
  transcriptList.innerHTML = "";
  currentTextBox.value = "";
  state.segments = [];
  state.lineElements = [];
  state.activeLineIndex = -1;
};

const resetResults = () => {
  resultPanel.classList.add("hidden");
  downloadTxt.removeAttribute("href");
  downloadSrt.removeAttribute("href");
  metaStatus.textContent = "-";
  metaDevice.textContent = "-";
  metaLanguage.textContent = "-";
  metaSegments.textContent = "-";
  clearLocalAudioObjectUrl();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  resetTranscript();
};

const setActiveLine = (lineIndex, scrollToLine = true) => {
  if (lineIndex === state.activeLineIndex) {
    return;
  }

  if (state.activeLineIndex >= 0) {
    const prev = state.lineElements[state.activeLineIndex];
    if (prev) {
      prev.classList.remove("active");
    }
  }

  state.activeLineIndex = lineIndex;
  if (lineIndex < 0 || lineIndex >= state.lineElements.length) {
    currentTextBox.value = "";
    return;
  }

  const activeButton = state.lineElements[lineIndex];
  activeButton.classList.add("active");
  currentTextBox.value = state.segments[lineIndex].text;

  if (scrollToLine) {
    activeButton.scrollIntoView({ block: "nearest" });
  }
};

const findSegmentIndexAtTime = (seconds) => {
  for (let i = 0; i < state.segments.length; i += 1) {
    const current = state.segments[i];
    const next = state.segments[i + 1];
    const segmentStart = Number(current.start);
    const segmentEnd = Number.isFinite(Number(current.end))
      ? Number(current.end)
      : next
        ? Number(next.start)
        : Number.POSITIVE_INFINITY;
    if (seconds >= segmentStart && seconds < segmentEnd) {
      return i;
    }
  }
  return -1;
};

const normalizeSegments = (rawSegments) => {
  const normalized = rawSegments
    .map((segment) => ({
      start: Number(segment.start),
      end: Number(segment.end),
      text: String(segment.text ?? "").trim(),
    }))
    .filter((segment) => Number.isFinite(segment.start) && segment.text.length > 0)
    .sort((a, b) => a.start - b.start);

  return normalized.map((segment, i) => ({
    index: i + 1,
    start: segment.start,
    end: Number.isFinite(segment.end) ? segment.end : segment.start,
    text: segment.text,
  }));
};

const renderSegments = (rawSegments) => {
  resetTranscript();
  const segments = normalizeSegments(rawSegments);
  state.segments = segments;

  if (!segments.length) {
    transcriptList.textContent = "No segments found.";
    return;
  }

  const fragment = document.createDocumentFragment();
  for (let i = 0; i < segments.length; i += 1) {
    const segment = segments[i];
    const row = document.createElement("button");
    row.type = "button";
    row.className = "line";
    row.dataset.index = String(segment.index);
    row.dataset.start = String(segment.start);
    row.innerHTML = `<span class="line-time">${formatTimestamp(segment.start)}</span><span class="line-text">${segment.text}</span>`;
    row.addEventListener("click", () => {
      audioPlayer.currentTime = Number(segment.start);
      setActiveLine(i, true);
      void audioPlayer.play();
    });
    state.lineElements.push(row);
    fragment.appendChild(row);
  }
  transcriptList.appendChild(fragment);
};

const parseSrtTimestamp = (value) => {
  const normalized = value.trim().replace(",", ".");
  const parts = normalized.split(":");
  if (parts.length !== 3) {
    return NaN;
  }
  const hours = Number(parts[0]);
  const minutes = Number(parts[1]);
  const secParts = parts[2].split(".");
  const seconds = Number(secParts[0]);
  const millis = Number((secParts[1] || "0").padEnd(3, "0").slice(0, 3));
  return (hours * 3600) + (minutes * 60) + seconds + (millis / 1000);
};

const parseSrtText = (srtText) => {
  const clean = srtText.replace(/^\uFEFF/, "").trim();
  if (!clean) {
    return [];
  }

  const blocks = clean.split(/\r?\n\r?\n+/);
  const parsed = [];

  for (const block of blocks) {
    const lines = block.split(/\r?\n/).map((line) => line.trimEnd());
    if (!lines.length) {
      continue;
    }

    let cursor = 0;
    if (/^\d+$/.test(lines[0].trim())) {
      cursor = 1;
    }
    if (cursor >= lines.length) {
      continue;
    }

    const timingLine = lines[cursor].trim();
    const timingMatch = timingLine.match(
      /(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})/
    );
    if (!timingMatch) {
      continue;
    }

    const start = parseSrtTimestamp(timingMatch[1]);
    const end = parseSrtTimestamp(timingMatch[2]);
    const text = lines.slice(cursor + 1).join(" ").trim();
    if (!Number.isFinite(start) || !Number.isFinite(end) || !text) {
      continue;
    }

    parsed.push({
      index: parsed.length + 1,
      start,
      end,
      text,
    });
  }

  return parsed;
};

const applyJobMeta = (job) => {
  metaStatus.textContent = job.status;
  metaDevice.textContent = `${job.device || "-"}${job.compute_type ? ` (${job.compute_type})` : ""}`;
  metaLanguage.textContent = job.language
    ? `${job.language} (${(Number(job.language_probability) * 100).toFixed(1)}%)`
    : "-";
  metaSegments.textContent = String(job.segment_count ?? "-");
};

const loadSegmentsFromServer = async (segmentsUrl) => {
  const response = await fetch(segmentsUrl);
  if (!response.ok) {
    throw new Error("Failed to load transcript segments.");
  }
  const segments = await response.json();
  renderSegments(segments);
};

const handleCompleted = async (job) => {
  applyJobMeta(job);
  resultPanel.classList.remove("hidden");
  if (job.downloads) {
    downloadTxt.href = job.downloads.txt;
    downloadSrt.href = job.downloads.srt;
  }
  clearLocalAudioObjectUrl();
  if (job.audio_url) {
    audioPlayer.src = job.audio_url;
  }
  if (job.segments_url) {
    await loadSegmentsFromServer(job.segments_url);
  }
};

const pollJob = async () => {
  if (!state.jobId) {
    return;
  }
  try {
    const response = await fetch(`/api/jobs/${state.jobId}`);
    if (!response.ok) {
      throw new Error("Failed to fetch job status.");
    }
    const job = await response.json();
    setStatus(job.status.toUpperCase(), job.message || job.status, job.progress);

    if (busyStates.has(job.status)) {
      return;
    }

    clearPolling();
    setBusy(false);
    if (job.status === "completed") {
      const elapsedFromServer = Number(job.transcription_seconds);
      const fallbackElapsed = Number(job.updated_at) - Number(job.created_at);
      const elapsed = formatElapsed(Number.isFinite(elapsedFromServer) ? elapsedFromServer : fallbackElapsed);
      const completionMessage = elapsed
        ? `Transcription completed in ${elapsed}.`
        : "Transcription completed.";
      setStatus("COMPLETED", completionMessage, 1);
      await handleCompleted(job);
    } else {
      setStatus("FAILED", job.error || "Transcription failed.", 1);
      metaStatus.textContent = "failed";
      resultPanel.classList.remove("hidden");
    }
  } catch (error) {
    clearPolling();
    setBusy(false);
    setStatus("ERROR", error instanceof Error ? error.message : "Unexpected error.", 1);
  }
};

const loadLocalPair = async () => {
  const audioFile = localAudioInput.files?.[0];
  const srtFile = localSrtInput.files?.[0];
  if (!audioFile || !srtFile) {
    setStatus("IDLE", "Choose both an audio file and an SRT file.", 0);
    return;
  }

  clearPolling();
  state.jobId = null;
  resetResults();
  setBusy(true);
  setStatus("LOADING", "Loading local audio and SRT...", 0.2);

  try {
    const srtText = await srtFile.text();
    const parsed = parseSrtText(srtText);
    if (!parsed.length) {
      throw new Error("The selected SRT file has no valid subtitle entries.");
    }

    const localAudioUrl = URL.createObjectURL(audioFile);
    state.localAudioUrl = localAudioUrl;
    audioPlayer.src = localAudioUrl;

    renderSegments(parsed);
    resultPanel.classList.remove("hidden");
    metaStatus.textContent = "local";
    metaDevice.textContent = "browser";
    metaLanguage.textContent = "-";
    metaSegments.textContent = String(parsed.length);
    downloadTxt.removeAttribute("href");
    downloadSrt.removeAttribute("href");
    localPairName.textContent = `${audioFile.name} + ${srtFile.name}`;
    setStatus("READY", "Loaded local audio and SRT for playback.", 1);
  } catch (error) {
    setStatus("ERROR", error instanceof Error ? error.message : "Failed to load local files.", 1);
  } finally {
    setBusy(false);
  }
};

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => setFile(fileInput.files?.[0] ?? null));

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    fileInput.files = event.dataTransfer.files;
    setFile(file);
  }
});

audioPlayer.addEventListener("timeupdate", () => {
  if (!state.segments.length) {
    return;
  }
  const activeIndex = findSegmentIndexAtTime(audioPlayer.currentTime);
  setActiveLine(activeIndex, true);
});

loadExistingBtn.addEventListener("click", () => {
  localAudioInput.value = "";
  localSrtInput.value = "";
  localAudioInput.click();
});

localAudioInput.addEventListener("change", () => {
  if (localAudioInput.files?.[0]) {
    localSrtInput.click();
  }
});

localSrtInput.addEventListener("change", () => {
  void loadLocalPair();
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.file) {
    setStatus("IDLE", "Select an audio file first.", 0);
    return;
  }

  clearPolling();
  state.jobId = null;
  resetResults();
  setBusy(true);
  setStatus("UPLOADING", "Uploading audio...", 0.02);

  const formData = new FormData();
  formData.append("file", state.file);
  formData.append("model", modelSelect.value);
  formData.append("export_folder", exportFolder.value || "default");

  try {
    const response = await fetch("/api/jobs", { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error("Failed to submit transcription job.");
    }
    const payload = await response.json();
    state.jobId = payload.job_id;
    setStatus("QUEUED", "Job queued.", 0.05);
    await pollJob();
    state.pollTimer = window.setInterval(() => {
      void pollJob();
    }, 900);
  } catch (error) {
    setBusy(false);
    setStatus("ERROR", error instanceof Error ? error.message : "Unexpected error.", 1);
  }
});
