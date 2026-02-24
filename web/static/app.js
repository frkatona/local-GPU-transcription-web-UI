const state = {
  file: null,
  uiMode: "file",
  jobId: null,
  pollTimer: null,
  localAudioUrl: null,
  segments: [],
  lineElements: [],
  activeLineIndex: -1,
  batchBusy: false,
  livePending: false,
  livePreloadPending: false,
  liveStatus: "idle",
  liveSocket: null,
  liveDevices: {
    mic: [],
    system: [],
  },
  liveMetrics: {
    inputLevelRms: 0,
    inputLevelPeak: 0,
    inputLevelDbfs: -90,
    inputLevelUpdatedAt: null,
    nextBufferUpdateSeconds: null,
    bufferIntervalSeconds: null,
    bufferQueueDepth: 0,
    receivedAtMs: 0,
  },
  diagnosticsTimer: null,
  localDownloadUrls: {
    txt: null,
    srt: null,
  },
};

const fileInput = document.getElementById("fileInput");
const localAudioInput = document.getElementById("localAudioInput");
const localSrtInput = document.getElementById("localSrtInput");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const uploadForm = document.getElementById("uploadForm");
const fileModeBtn = document.getElementById("fileModeBtn");
const liveModeBtn = document.getElementById("liveModeBtn");
const fileModePanel = document.getElementById("fileModePanel");
const liveModePanel = document.getElementById("liveModePanel");
const modelSelect = document.getElementById("modelSelect");
const exportFolder = document.getElementById("exportFolder");
const startBtn = document.getElementById("startBtn");
const loadExistingBtn = document.getElementById("loadExistingBtn");
const localPairName = document.getElementById("localPairName");
const liveSource = document.getElementById("liveSource");
const liveCaptureMode = document.getElementById("liveCaptureMode");
const liveDevice = document.getElementById("liveDevice");
const liveMode = document.getElementById("liveMode");
const liveModel = document.getElementById("liveModel");
const liveLanguage = document.getElementById("liveLanguage");
const livePreloadBtn = document.getElementById("livePreloadBtn");
const livePreloadStatus = document.getElementById("livePreloadStatus");
const liveToggleBtn = document.getElementById("liveToggleBtn");
const liveLevelBar = document.getElementById("liveLevelBar");
const liveLevelDbfs = document.getElementById("liveLevelDbfs");
const bufferCountdownLabel = document.getElementById("bufferCountdownLabel");
const bufferCountdownBar = document.getElementById("bufferCountdownBar");
const statusLabel = document.getElementById("statusLabel");
const statusPercent = document.getElementById("statusPercent");
const statusMessage = document.getElementById("statusMessage");
const progressBar = document.getElementById("progressBar");
const resultPanel = document.getElementById("resultPanel");
const downloadTxt = document.getElementById("downloadTxt");
const downloadSrt = document.getElementById("downloadSrt");
const downloadAudio = document.getElementById("downloadAudio");
const downloadAudio16k = document.getElementById("downloadAudio16k");
const downloadDiagnostics = document.getElementById("downloadDiagnostics");
const metaStatus = document.getElementById("metaStatus");
const metaDevice = document.getElementById("metaDevice");
const metaLanguage = document.getElementById("metaLanguage");
const metaSegments = document.getElementById("metaSegments");
const audioPlayer = document.getElementById("audioPlayer");
const transcriptList = document.getElementById("transcriptList");
const currentTextBox = document.getElementById("currentTextBox");

const liveDebugDownloadAnchors = [downloadAudio, downloadAudio16k, downloadDiagnostics].filter(Boolean);

const busyStates = new Set(["queued", "running"]);
const liveActiveStates = new Set(["starting", "running", "stopping"]);
const defaultBufferedIntervalSeconds = 60;

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

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));

const interpolate = (start, end, ratio) => start + ((end - start) * ratio);

const liveLevelColor = (peak) => {
  const level = clamp01(peak);
  const hue = interpolate(150, 8, level);
  return `hsl(${hue.toFixed(1)}, 72%, 48%)`;
};

const formatSecondsLabel = (seconds) => {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) {
    return null;
  }
  if (value < 10) {
    return `${value.toFixed(1)}s`;
  }
  return `${Math.round(value)}s`;
};

const getBufferedCountdownRemaining = () => {
  const baseRemaining = state.liveMetrics.nextBufferUpdateSeconds;
  if (!Number.isFinite(baseRemaining)) {
    return null;
  }
  const ageSeconds = state.liveMetrics.receivedAtMs > 0
    ? (performance.now() - state.liveMetrics.receivedAtMs) / 1000
    : 0;
  return Math.max(0, baseRemaining - ageSeconds);
};

const renderLiveDiagnostics = () => {
  const peak = clamp01(state.liveMetrics.inputLevelPeak);
  const visualPeak = Math.pow(peak, 0.65);
  liveLevelBar.style.width = `${(visualPeak * 100).toFixed(1)}%`;
  liveLevelBar.style.backgroundColor = liveLevelColor(peak);

  const dbfs = Number(state.liveMetrics.inputLevelDbfs);
  if (Number.isFinite(dbfs)) {
    liveLevelDbfs.textContent = `${dbfs.toFixed(1)} dBFS`;
  } else {
    liveLevelDbfs.textContent = "-90 dBFS";
  }

  const bufferedModeActive = liveCaptureMode.value === "buffered_hq"
    && (state.liveStatus === "starting" || state.liveStatus === "running" || state.liveStatus === "stopping");
  if (!bufferedModeActive) {
    bufferCountdownLabel.textContent = "N/A";
    bufferCountdownBar.style.width = "0%";
    return;
  }

  const remaining = getBufferedCountdownRemaining();
  const intervalRaw = Number(state.liveMetrics.bufferIntervalSeconds);
  const interval = Number.isFinite(intervalRaw) && intervalRaw > 0
    ? intervalRaw
    : defaultBufferedIntervalSeconds;
  const ratio = remaining === null ? 0 : (1 - Math.min(1, remaining / interval));
  bufferCountdownBar.style.width = `${Math.max(0, ratio * 100).toFixed(1)}%`;

  if (remaining === null) {
    bufferCountdownLabel.textContent = "Preparing...";
    return;
  }

  const queueDepth = Number(state.liveMetrics.bufferQueueDepth);
  const base = formatSecondsLabel(remaining) || "0s";
  bufferCountdownLabel.textContent = queueDepth > 0
    ? `${base} (queue ${queueDepth})`
    : base;
};

const applyLiveMetrics = (metrics) => {
  if (!metrics || typeof metrics !== "object") {
    return;
  }
  const hasOwn = (key) => Object.prototype.hasOwnProperty.call(metrics, key);

  if (hasOwn("input_level_rms")) {
    const value = Number(metrics.input_level_rms);
    if (Number.isFinite(value)) {
      state.liveMetrics.inputLevelRms = Math.max(0, value);
    }
  }
  if (hasOwn("input_level_peak")) {
    const value = Number(metrics.input_level_peak);
    if (Number.isFinite(value)) {
      state.liveMetrics.inputLevelPeak = clamp01(value);
    }
  }
  if (hasOwn("input_level_dbfs")) {
    const value = Number(metrics.input_level_dbfs);
    if (Number.isFinite(value)) {
      state.liveMetrics.inputLevelDbfs = value;
    }
  }
  if (hasOwn("input_level_updated_at")) {
    const value = Number(metrics.input_level_updated_at);
    state.liveMetrics.inputLevelUpdatedAt = Number.isFinite(value) ? value : null;
  }
  if (hasOwn("next_buffer_update_seconds")) {
    const value = Number(metrics.next_buffer_update_seconds);
    state.liveMetrics.nextBufferUpdateSeconds = Number.isFinite(value) ? Math.max(0, value) : null;
  }
  if (hasOwn("buffer_interval_seconds")) {
    const value = Number(metrics.buffer_interval_seconds);
    state.liveMetrics.bufferIntervalSeconds = Number.isFinite(value) && value > 0 ? value : null;
  }
  if (hasOwn("buffer_queue_depth")) {
    const value = Number(metrics.buffer_queue_depth);
    state.liveMetrics.bufferQueueDepth = Number.isFinite(value) ? Math.max(0, Math.round(value)) : 0;
  }
  state.liveMetrics.receivedAtMs = performance.now();
  renderLiveDiagnostics();
};

const ensureDiagnosticsTicker = () => {
  if (state.diagnosticsTimer) {
    return;
  }
  state.diagnosticsTimer = window.setInterval(() => {
    renderLiveDiagnostics();
  }, 150);
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

const revokeLocalDownloads = () => {
  if (state.localDownloadUrls.txt) {
    URL.revokeObjectURL(state.localDownloadUrls.txt);
    state.localDownloadUrls.txt = null;
  }
  if (state.localDownloadUrls.srt) {
    URL.revokeObjectURL(state.localDownloadUrls.srt);
    state.localDownloadUrls.srt = null;
  }
};

const clearDownloadLinks = () => {
  revokeLocalDownloads();
  downloadTxt.removeAttribute("href");
  downloadSrt.removeAttribute("href");
  downloadTxt.removeAttribute("download");
  downloadSrt.removeAttribute("download");
  for (const anchor of liveDebugDownloadAnchors) {
    anchor.removeAttribute("href");
    anchor.removeAttribute("download");
    anchor.classList.add("hidden");
  }
};

const setServerDownloads = (downloads) => {
  clearDownloadLinks();
  if (!downloads) {
    return;
  }
  if (downloads.txt) {
    downloadTxt.href = downloads.txt;
  }
  if (downloads.srt) {
    downloadSrt.href = downloads.srt;
  }
  if (downloads.audio && downloadAudio) {
    downloadAudio.href = downloads.audio;
    downloadAudio.classList.remove("hidden");
  }
  if (downloads.audio_16k && downloadAudio16k) {
    downloadAudio16k.href = downloads.audio_16k;
    downloadAudio16k.classList.remove("hidden");
  }
  if (downloads.diagnostics && downloadDiagnostics) {
    downloadDiagnostics.href = downloads.diagnostics;
    downloadDiagnostics.classList.remove("hidden");
  }
};

const setLivePlayerAudio = (audioUrl, sessionId = null) => {
  if (!audioUrl) {
    return;
  }
  if (audioPlayer.src && audioPlayer.src.includes(audioUrl)) {
    return;
  }
  const stamp = Date.now();
  const sessionPart = sessionId ? `&session=${encodeURIComponent(sessionId)}` : "";
  const separator = audioUrl.includes("?") ? "&" : "?";
  const resolved = `${audioUrl}${separator}ts=${stamp}${sessionPart}`;
  clearLocalAudioObjectUrl();
  audioPlayer.src = resolved;
};

const buildTxtFromSegments = (segments, title = "transcript") => {
  const lines = ["# Transcript", `# Source: ${title}`, ""];
  for (const segment of segments) {
    lines.push(`[${formatTimestamp(segment.start)}] ${segment.text}`);
  }
  return `${lines.join("\n").trimEnd()}\n`;
};

const buildSrtFromSegments = (segments) => {
  const toSrtTime = (seconds) => {
    const totalMs = Math.max(0, Math.round(Number(seconds) * 1000));
    const hours = Math.floor(totalMs / 3_600_000);
    const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
    const secs = Math.floor((totalMs % 60_000) / 1000);
    const ms = totalMs % 1000;
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
  };

  const lines = [];
  for (let i = 0; i < segments.length; i += 1) {
    const segment = segments[i];
    lines.push(String(i + 1));
    lines.push(`${toSrtTime(segment.start)} --> ${toSrtTime(segment.end)}`);
    lines.push(segment.text);
    lines.push("");
  }
  return `${lines.join("\n").trimEnd()}\n`;
};

const setLocalDownloadsFromSegments = (segments, stem = "transcript") => {
  clearDownloadLinks();
  if (!segments.length) {
    return;
  }

  const txtBlob = new Blob([buildTxtFromSegments(segments, stem)], { type: "text/plain;charset=utf-8" });
  const srtBlob = new Blob([buildSrtFromSegments(segments)], { type: "application/x-subrip;charset=utf-8" });
  const txtUrl = URL.createObjectURL(txtBlob);
  const srtUrl = URL.createObjectURL(srtBlob);

  state.localDownloadUrls.txt = txtUrl;
  state.localDownloadUrls.srt = srtUrl;

  downloadTxt.href = txtUrl;
  downloadSrt.href = srtUrl;
  downloadTxt.download = `${stem}.txt`;
  downloadSrt.download = `${stem}.srt`;
};

const setFile = (file) => {
  state.file = file;
  fileName.textContent = file ? file.name : "No file selected";
};

const isLiveActive = () => liveActiveStates.has(state.liveStatus);

const setUIMode = (mode, options = {}) => {
  const nextMode = mode === "live" ? "live" : "file";
  const force = Boolean(options.force);
  if (!force) {
    if (state.livePending || state.livePreloadPending) {
      return false;
    }
    if (isLiveActive() && nextMode !== "live") {
      return false;
    }
    if (state.batchBusy && nextMode !== "file") {
      return false;
    }
  }

  state.uiMode = nextMode;
  const isLiveMode = state.uiMode === "live";
  fileModePanel.classList.toggle("hidden", isLiveMode);
  liveModePanel.classList.toggle("hidden", !isLiveMode);
  fileModeBtn.classList.toggle("active", !isLiveMode);
  liveModeBtn.classList.toggle("active", isLiveMode);
  return true;
};

const devicesForSource = (source) => {
  if (source === "system") {
    return state.liveDevices.system || [];
  }
  return state.liveDevices.mic || [];
};

const populateLiveDeviceOptions = (source, preferredId = null) => {
  const devices = devicesForSource(source);
  liveDevice.innerHTML = "";

  if (!devices.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = source === "system" ? "No system device found" : "No microphone found";
    liveDevice.appendChild(option);
    liveDevice.disabled = true;
    return;
  }

  const targetId = preferredId
    || devices.find((device) => device.default)?.id
    || devices[0].id;

  for (const device of devices) {
    const option = document.createElement("option");
    option.value = String(device.id);
    option.textContent = device.label;
    if (String(device.id) === String(targetId)) {
      option.selected = true;
    }
    liveDevice.appendChild(option);
  }
  liveDevice.disabled = false;
};

const refreshControls = () => {
  const lockMethodSwitch = state.livePending || state.livePreloadPending;
  fileModeBtn.disabled = lockMethodSwitch || isLiveActive();
  liveModeBtn.disabled = lockMethodSwitch || state.batchBusy;

  const lockFileActions = state.batchBusy || isLiveActive();
  startBtn.disabled = lockFileActions;
  loadExistingBtn.disabled = lockFileActions;
  dropZone.disabled = lockFileActions;

  const lockLiveButton = state.batchBusy || state.livePending || state.liveStatus === "stopping";
  liveToggleBtn.disabled = lockLiveButton;

  const lockLiveSelectors = state.batchBusy || isLiveActive() || state.livePending;
  liveSource.disabled = lockLiveSelectors;
  liveCaptureMode.disabled = lockLiveSelectors;
  liveDevice.disabled = lockLiveSelectors || !devicesForSource(liveSource.value).length;
  liveMode.disabled = lockLiveSelectors;
  liveModel.disabled = lockLiveSelectors;
  liveLanguage.disabled = lockLiveSelectors;
  livePreloadBtn.disabled = lockLiveSelectors || state.livePreloadPending;
  livePreloadBtn.textContent = state.livePreloadPending ? "Pre-loading..." : "Pre-load Selected Model";

  if (state.batchBusy) {
    startBtn.textContent = "Working...";
  } else {
    startBtn.textContent = "Start Transcription";
  }

  if (isLiveActive()) {
    liveToggleBtn.textContent = state.liveStatus === "stopping" ? "Stopping..." : "Stop Live Transcription";
    liveToggleBtn.classList.add("running");
  } else {
    liveToggleBtn.textContent = "Start Live Transcription";
    liveToggleBtn.classList.remove("running");
  }

  renderLiveDiagnostics();
};

const setBatchBusy = (busy) => {
  state.batchBusy = busy;
  refreshControls();
};

const setLivePending = (pending) => {
  state.livePending = pending;
  refreshControls();
};

const setLivePreloadPending = (pending) => {
  state.livePreloadPending = pending;
  refreshControls();
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
  metaStatus.textContent = "-";
  metaDevice.textContent = "-";
  metaLanguage.textContent = "-";
  metaSegments.textContent = "-";
  clearLocalAudioObjectUrl();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  clearDownloadLinks();
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
    const start = Number(current.start);
    const end = Number.isFinite(Number(current.end))
      ? Number(current.end)
      : next
        ? Number(next.start)
        : Number.POSITIVE_INFINITY;
    if (seconds >= start && seconds < end) {
      return i;
    }
  }
  return -1;
};

const normalizeSegments = (rawSegments) => rawSegments
  .map((segment) => ({
    start: Number(segment.start),
    end: Number(segment.end),
    text: String(segment.text ?? "").trim(),
  }))
  .filter((segment) => Number.isFinite(segment.start) && segment.text.length > 0)
  .sort((a, b) => a.start - b.start)
  .map((segment, index) => ({
    index: index + 1,
    start: segment.start,
    end: Number.isFinite(segment.end) ? segment.end : segment.start,
    text: segment.text,
  }));

const buildSegmentRow = (segment, rowIndex) => {
  const row = document.createElement("button");
  row.type = "button";
  row.className = "line";
  row.dataset.index = String(segment.index);
  row.dataset.start = String(segment.start);
  row.innerHTML = `<span class="line-time">${formatTimestamp(segment.start)}</span><span class="line-text">${segment.text}</span>`;
  row.addEventListener("click", () => {
    if (audioPlayer.src) {
      audioPlayer.currentTime = Number(segment.start);
      void audioPlayer.play();
    }
    setActiveLine(rowIndex, true);
  });
  return row;
};

const renderSegments = (rawSegments) => {
  resetTranscript();
  const segments = normalizeSegments(rawSegments);
  state.segments = segments;

  if (!segments.length) {
    transcriptList.textContent = "No segments found.";
    return;
  }

  transcriptList.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < segments.length; i += 1) {
    const row = buildSegmentRow(segments[i], i);
    state.lineElements.push(row);
    fragment.appendChild(row);
  }
  transcriptList.appendChild(fragment);
};

const appendSegment = (rawSegment) => {
  const normalized = normalizeSegments([rawSegment])[0];
  if (!normalized) {
    return;
  }
  if (transcriptList.textContent === "No segments found.") {
    transcriptList.innerHTML = "";
  }

  normalized.index = state.segments.length + 1;
  state.segments.push(normalized);
  const row = buildSegmentRow(normalized, state.segments.length - 1);
  state.lineElements.push(row);
  transcriptList.appendChild(row);
  setActiveLine(state.segments.length - 1, true);
};

const parseSrtTimestamp = (value) => {
  const normalized = value.trim().replace(",", ".");
  const parts = normalized.split(":");
  if (parts.length !== 3) {
    return Number.NaN;
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
  setServerDownloads(job.downloads || null);
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
    setBatchBusy(false);
    if (job.status === "completed") {
      const elapsedFromServer = Number(job.transcription_seconds);
      const fallbackElapsed = Number(job.updated_at) - Number(job.created_at);
      const elapsed = formatElapsed(Number.isFinite(elapsedFromServer) ? elapsedFromServer : fallbackElapsed);
      const completionMessage = elapsed ? `Transcription completed in ${elapsed}.` : "Transcription completed.";
      setStatus("COMPLETED", completionMessage, 1);
      await handleCompleted(job);
    } else {
      setStatus("FAILED", job.error || "Transcription failed.", 1);
      metaStatus.textContent = "failed";
      resultPanel.classList.remove("hidden");
    }
  } catch (error) {
    clearPolling();
    setBatchBusy(false);
    setStatus("ERROR", error instanceof Error ? error.message : "Unexpected error.", 1);
  }
};

const getErrorDetail = async (response, fallback) => {
  try {
    const payload = await response.json();
    return payload?.detail || fallback;
  } catch {
    return fallback;
  }
};

const applyLiveState = (liveState, preferStatusMessage = true) => {
  const prevStatus = state.liveStatus;
  state.liveStatus = String(liveState.status || "idle");
  if (isLiveActive()) {
    setUIMode("live", { force: true });
  }

  if (liveState.source && liveState.source !== liveSource.value) {
    liveSource.value = liveState.source;
  }
  if (liveState.model && liveState.model !== liveModel.value) {
    liveModel.value = liveState.model;
  }
  if (Object.prototype.hasOwnProperty.call(liveState, "language")) {
    const stateLanguage = liveState.language || "auto";
    const matchingOption = Array.from(liveLanguage.options).find((opt) => opt.value === stateLanguage);
    liveLanguage.value = matchingOption ? matchingOption.value : "auto";
  }
  if (liveState.capture_mode && liveState.capture_mode !== liveCaptureMode.value) {
    liveCaptureMode.value = liveState.capture_mode;
  }
  if (liveState.device_id !== undefined) {
    populateLiveDeviceOptions(liveSource.value, liveState.device_id || null);
  }
  applyLiveMetrics(liveState);

  setLivePending(false);
  refreshControls();

  if (!state.batchBusy) {
    if (state.liveStatus === "running") {
      const sourceLabel = liveState.source === "system" ? "system audio" : "microphone";
      const modeLabel = liveState.capture_mode === "buffered_hq" ? "buffered HQ" : "low latency";
      if (preferStatusMessage) {
        setStatus("LIVE", `Listening to ${sourceLabel} (${modeLabel})...`, 1);
      }
    } else if (state.liveStatus === "starting") {
      if (preferStatusMessage) {
        setStatus("LIVE", "Starting live transcription...", 1);
      }
    } else if (state.liveStatus === "stopping") {
      if (preferStatusMessage) {
        setStatus("LIVE", "Stopping live transcription...", 1);
      }
    } else if (state.liveStatus === "error") {
      if (preferStatusMessage) {
        setStatus("LIVE ERROR", liveState.error || liveState.message || "Live transcription failed.", 1);
      }
    } else if (state.liveStatus === "idle" && prevStatus !== "idle") {
      if (preferStatusMessage) {
        setStatus("READY", liveState.message || "Live transcription is idle.", 1);
      }
    }
  }

  if (liveState.segment_count !== undefined) {
    metaSegments.textContent = String(liveState.segment_count);
  }

  if (state.liveStatus === "running" || state.liveStatus === "starting" || state.liveStatus === "stopping") {
    resultPanel.classList.remove("hidden");
    metaStatus.textContent = `live-${state.liveStatus}`;
    metaDevice.textContent = liveState.device_label || (liveState.source === "system" ? "system loopback" : "microphone");
    metaLanguage.textContent = "-";
  } else if (state.liveStatus === "idle" && Number(liveState.segment_count || 0) > 0) {
    resultPanel.classList.remove("hidden");
    metaStatus.textContent = "live-idle";
    metaDevice.textContent = liveState.device_label || (liveState.source === "system" ? "system loopback" : "microphone");
    metaLanguage.textContent = "-";
  } else if (state.liveStatus === "error") {
    metaStatus.textContent = "live-error";
  }

  if (liveState.downloads) {
    setServerDownloads(liveState.downloads);
    if (liveState.downloads.audio) {
      setLivePlayerAudio(liveState.downloads.audio, liveState.session_id || null);
    }
  } else if (state.liveStatus === "running" || state.liveStatus === "starting") {
    clearDownloadLinks();
  }
};

const fetchLiveSegments = async () => {
  const response = await fetch("/api/live/segments");
  if (!response.ok) {
    return;
  }
  const segments = await response.json();
  renderSegments(segments);
  if (segments.length) {
    resultPanel.classList.remove("hidden");
    setActiveLine(segments.length - 1, true);
  }
};

const fetchLiveState = async () => {
  const response = await fetch("/api/live/state");
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  if (payload?.state) {
    applyLiveState(payload.state, false);
    if (Number(payload.state.segment_count || 0) > 0) {
      await fetchLiveSegments();
    }
  }
};

const loadLiveDevices = async () => {
  try {
    const response = await fetch("/api/live/devices");
    if (!response.ok) {
      throw new Error("Failed to load live devices.");
    }
    const payload = await response.json();
    state.liveDevices.mic = Array.isArray(payload.mic) ? payload.mic : [];
    state.liveDevices.system = Array.isArray(payload.system) ? payload.system : [];
    populateLiveDeviceOptions(liveSource.value);
    refreshControls();
  } catch {
    state.liveDevices.mic = [];
    state.liveDevices.system = [];
    populateLiveDeviceOptions(liveSource.value);
    refreshControls();
  }
};

const handleLiveSegmentMessage = (segment, liveState) => {
  if (state.batchBusy) {
    return;
  }
  resultPanel.classList.remove("hidden");
  appendSegment(segment);
  metaStatus.textContent = "live-running";
  metaDevice.textContent = liveState?.device_label
    || (liveSource.value === "system" ? "system loopback" : "microphone");
  metaLanguage.textContent = "-";
  metaSegments.textContent = String(state.segments.length);
  clearDownloadLinks();
  if (liveState) {
    applyLiveState(liveState, false);
  }
};

const connectLiveSocket = () => {
  if (state.liveSocket && (state.liveSocket.readyState === WebSocket.OPEN || state.liveSocket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${wsProtocol}://${window.location.host}/ws/live`);
  state.liveSocket = ws;

  ws.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "live_state" && payload.state) {
        applyLiveState(payload.state, true);
      } else if (payload.type === "live_metrics" && payload.metrics) {
        applyLiveMetrics(payload.metrics);
      } else if (payload.type === "live_snapshot" && Array.isArray(payload.segments) && !state.batchBusy) {
        if (payload.segments.length) {
          resultPanel.classList.remove("hidden");
          renderSegments(payload.segments);
        }
      } else if (payload.type === "segment" && payload.segment) {
        handleLiveSegmentMessage(payload.segment, payload.state);
      }
    } catch {
      // Ignore malformed websocket events.
    }
  });

  ws.addEventListener("close", () => {
    state.liveSocket = null;
    if (isLiveActive() && !state.batchBusy) {
      setStatus("LIVE", "Live stream connection lost. Refreshing state...", 1);
      window.setTimeout(() => {
        void fetchLiveState();
        connectLiveSocket();
      }, 1200);
    }
  });
};

const preloadSelectedLiveModel = async () => {
  if (state.batchBusy || isLiveActive() || state.livePending) {
    return;
  }

  setLivePreloadPending(true);
  livePreloadStatus.textContent = `Pre-loading ${liveModel.value}...`;
  try {
    const response = await fetch("/api/models/preload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: liveModel.value }),
    });
    if (!response.ok) {
      const detail = await getErrorDetail(response, "Failed to preload model.");
      throw new Error(detail);
    }

    const payload = await response.json();
    const loadSeconds = Number(payload.load_seconds);
    const loadLabel = Number.isFinite(loadSeconds)
      ? (loadSeconds < 1 ? `${Math.round(loadSeconds * 1000)}ms` : `${loadSeconds.toFixed(1)}s`)
      : null;
    const cachePrefix = payload.cached_before ? "Already cached" : "Loaded";
    livePreloadStatus.textContent = `${cachePrefix}: ${payload.model} on ${payload.device} (${payload.compute_type})${loadLabel ? ` in ${loadLabel}` : ""}`;
    if (!state.batchBusy) {
      setStatus("READY", `Model ${payload.model} pre-loaded.`, 1);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to preload model.";
    livePreloadStatus.textContent = message;
    if (!state.batchBusy) {
      setStatus("ERROR", message, 1);
    }
  } finally {
    setLivePreloadPending(false);
  }
};

const startLive = async () => {
  if (state.batchBusy) {
    return;
  }
  setUIMode("live", { force: true });
  if (!liveDevice.value) {
    setStatus("ERROR", "Select a live capture device first.", 1);
    return;
  }
  setLivePending(true);
  const modeLabel = liveCaptureMode.value === "buffered_hq" ? "buffered HQ" : "low latency";
  setStatus("LIVE", `Starting ${modeLabel} transcription...`, 1);

  if (liveMode.value === "new") {
    clearPolling();
    state.jobId = null;
    resetResults();
    resultPanel.classList.remove("hidden");
  }

  connectLiveSocket();

  const response = await fetch("/api/live/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source: liveSource.value,
      mode: liveMode.value,
      model: liveModel.value,
      language: liveLanguage.value,
      device_id: liveDevice.value || null,
      capture_mode: liveCaptureMode.value,
    }),
  });

  if (!response.ok) {
    const detail = await getErrorDetail(response, "Failed to start live transcription.");
    setLivePending(false);
    setStatus("ERROR", detail, 1);
    throw new Error(detail);
  }

  const payload = await response.json();
  if (payload?.state) {
    applyLiveState(payload.state, true);
    if (payload.state.mode === "append" && Number(payload.state.segment_count || 0) > 0) {
      await fetchLiveSegments();
    }
  }
};

const stopLive = async () => {
  setLivePending(true);
  const response = await fetch("/api/live/stop", { method: "POST" });
  if (!response.ok) {
    const detail = await getErrorDetail(response, "Failed to stop live transcription.");
    setLivePending(false);
    setStatus("ERROR", detail, 1);
    throw new Error(detail);
  }
  const payload = await response.json();
  if (payload?.state) {
    applyLiveState(payload.state, true);
  }
  setLivePending(false);
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
  setBatchBusy(true);
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
    localPairName.textContent = `${audioFile.name} + ${srtFile.name}`;

    const stem = srtFile.name.replace(/\.[^/.]+$/, "") || "local-transcript";
    setLocalDownloadsFromSegments(parsed, stem);
    setStatus("READY", "Loaded local audio and SRT for playback.", 1);
  } catch (error) {
    setStatus("ERROR", error instanceof Error ? error.message : "Failed to load local files.", 1);
  } finally {
    setBatchBusy(false);
  }
};

dropZone.addEventListener("click", () => {
  if (!startBtn.disabled) {
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => setFile(fileInput.files?.[0] ?? null));

dropZone.addEventListener("dragover", (event) => {
  if (startBtn.disabled) {
    return;
  }
  event.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", (event) => {
  if (startBtn.disabled) {
    return;
  }
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = event.dataTransfer?.files?.[0];
  if (file) {
    fileInput.files = event.dataTransfer.files;
    setFile(file);
  }
});

audioPlayer.addEventListener("timeupdate", () => {
  if (!state.segments.length || !audioPlayer.src) {
    return;
  }
  const activeIndex = findSegmentIndexAtTime(audioPlayer.currentTime);
  setActiveLine(activeIndex, true);
});

loadExistingBtn.addEventListener("click", () => {
  if (isLiveActive()) {
    setStatus("LIVE", "Stop live transcription before loading local files.", 1);
    return;
  }
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

fileModeBtn.addEventListener("click", () => {
  setUIMode("file");
  refreshControls();
});

liveModeBtn.addEventListener("click", () => {
  setUIMode("live");
  refreshControls();
});

liveSource.addEventListener("change", () => {
  populateLiveDeviceOptions(liveSource.value);
  refreshControls();
});

liveModel.addEventListener("change", () => {
  if (!state.livePreloadPending) {
    livePreloadStatus.textContent = "No live model pre-loaded in this session";
  }
});

livePreloadBtn.addEventListener("click", async () => {
  await preloadSelectedLiveModel();
});

liveToggleBtn.addEventListener("click", async () => {
  try {
    if (isLiveActive()) {
      await stopLive();
    } else {
      await startLive();
    }
  } catch {
    // Status already updated in helpers.
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isLiveActive()) {
    setStatus("LIVE", "Stop live transcription before uploading a file.", 1);
    return;
  }
  if (!state.file) {
    setStatus("IDLE", "Select an audio file first.", 0);
    return;
  }

  clearPolling();
  state.jobId = null;
  resetResults();
  setBatchBusy(true);
  setStatus("UPLOADING", "Uploading audio...", 0.02);

  const formData = new FormData();
  formData.append("file", state.file);
  formData.append("model", modelSelect.value);
  formData.append("export_folder", exportFolder.value || "default");

  try {
    const response = await fetch("/api/jobs", { method: "POST", body: formData });
    if (!response.ok) {
      const detail = await getErrorDetail(response, "Failed to submit transcription job.");
      throw new Error(detail);
    }
    const payload = await response.json();
    state.jobId = payload.job_id;
    setStatus("QUEUED", "Job queued.", 0.05);
    await pollJob();
    state.pollTimer = window.setInterval(() => {
      void pollJob();
    }, 900);
  } catch (error) {
    setBatchBusy(false);
    setStatus("ERROR", error instanceof Error ? error.message : "Unexpected error.", 1);
  }
});

connectLiveSocket();
void loadLiveDevices().then(() => fetchLiveState());
ensureDiagnosticsTicker();
setUIMode("file", { force: true });
renderLiveDiagnostics();
refreshControls();
