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
    markdown: null,
  },
  settingsMenuOpen: false,
  transcriptView: "timeline",
  verboseDescriptions: true,
  statusRawLabel: "Idle",
  statusRawMessage: "Drop an audio file to begin.",
  localPairText: "",
  livePreloadStatusMode: "default",
  livePreloadText: "",
  metaStatusRaw: "-",
  metaSegmentCountRaw: null,
  metaSpeakerCountRaw: null,
  downloadOptionCount: 0,
  downloadsMenuOpen: false,
  outputStem: "transcript",
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
const settingsMenuBtn = document.getElementById("settingsMenuBtn");
const settingsOverlay = document.getElementById("settingsOverlay");
const settingsBackdrop = document.getElementById("settingsBackdrop");
const settingsPanel = document.getElementById("settingsPanel");
const settingsPanelTitle = document.getElementById("settingsPanelTitle");
const settingsCloseBtn = document.getElementById("settingsCloseBtn");
const modelSelect = document.getElementById("modelSelect");
const exportFolder = document.getElementById("exportFolder");
const themeSelect = document.getElementById("themeSelect");
const verboseDescriptionsToggle = document.getElementById("verboseDescriptionsToggle");
const diarizeToggle = document.getElementById("diarizeToggle");
const diarizationToken = document.getElementById("diarizationToken");
const diarizeSpeakers = document.getElementById("diarizeSpeakers");
const startBtn = document.getElementById("startBtn");
const loadExistingBtn = document.getElementById("loadExistingBtn");
const localPairName = document.getElementById("localPairName");
const liveSource = document.getElementById("liveSource");
const liveCaptureMode = document.getElementById("liveCaptureMode");
const liveDevice = document.getElementById("liveDevice");
const liveMode = document.getElementById("liveMode");
const liveModel = document.getElementById("liveModel");
const liveLanguage = document.getElementById("liveLanguage");
const fileSettingsSection = document.getElementById("fileSettingsSection");
const liveSettingsSection = document.getElementById("liveSettingsSection");
const liveThemeSelect = document.getElementById("liveThemeSelect");
const liveVerboseDescriptionsToggle = document.getElementById("liveVerboseDescriptionsToggle");
const liveDiarizeToggle = document.getElementById("liveDiarizeToggle");
const liveDiarizationToken = document.getElementById("liveDiarizationToken");
const liveDiarizeSpeakers = document.getElementById("liveDiarizeSpeakers");
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
const downloadsMenu = document.getElementById("downloadsMenu");
const downloadsTrigger = document.getElementById("downloadsTrigger");
const downloadsList = document.getElementById("downloadsList");
const downloadsTriggerLabel = document.getElementById("downloadsTriggerLabel");
const downloadsTriggerHint = document.getElementById("downloadsTriggerHint");
const downloadTxt = document.getElementById("downloadTxt");
const downloadSrt = document.getElementById("downloadSrt");
const downloadMarkdown = document.getElementById("downloadMarkdown");
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
const readingViewBtn = document.getElementById("readingViewBtn");
const readingViewHint = document.getElementById("readingViewHint");
const readingView = document.getElementById("readingView");
const methodSwitch = document.querySelector(".method-switch");
const heroEyebrow = document.querySelector(".eyebrow");
const heroTitle = document.querySelector(".hero h1");
const heroSubtitle = document.querySelector(".subtitle");
const dropTitleLabel = document.querySelector(".drop-title");
const dropSubtitleLabel = document.querySelector(".drop-subtitle");
const fileThemeLabel = themeSelect?.closest(".field")?.querySelector("span") || null;
const liveThemeLabel = liveThemeSelect?.closest(".field")?.querySelector("span") || null;
const verboseDescriptionsLabel = verboseDescriptionsToggle?.closest("label")?.querySelector("span") || null;
const liveVerboseDescriptionsLabel = liveVerboseDescriptionsToggle?.closest("label")?.querySelector("span") || null;
const modelLabel = modelSelect?.closest(".field")?.querySelector("span") || null;
const exportFolderLabel = exportFolder?.closest(".field")?.querySelector("span") || null;
const diarizeLabel = diarizeToggle?.closest("label")?.querySelector("span") || null;
const tokenLabel = diarizationToken?.closest(".field")?.querySelector("span") || null;
const expectedSpeakersLabel = diarizeSpeakers?.closest(".field")?.querySelector("span") || null;
const liveSourceLabel = liveSource?.closest(".field")?.querySelector("span") || null;
const liveCaptureModeLabel = liveCaptureMode?.closest(".field")?.querySelector("span") || null;
const liveDeviceLabel = liveDevice?.closest(".field")?.querySelector("span") || null;
const liveModeLabel = liveMode?.closest(".field")?.querySelector("span") || null;
const liveModelLabel = liveModel?.closest(".field")?.querySelector("span") || null;
const liveLanguageLabel = liveLanguage?.closest(".field")?.querySelector("span") || null;
const liveDiarizeLabel = liveDiarizeToggle?.closest("label")?.querySelector("span") || null;
const liveTokenLabel = liveDiarizationToken?.closest(".field")?.querySelector("span") || null;
const liveExpectedSpeakersLabel = liveDiarizeSpeakers?.closest(".field")?.querySelector("span") || null;
const modelWarmupLabel = livePreloadBtn?.closest(".field")?.querySelector("span") || null;
const resultHeadings = resultPanel?.querySelectorAll("h2") || [];
const outputsHeading = resultHeadings[0] || null;
const playerHeading = resultHeadings[1] || null;
const metaLabels = document.querySelectorAll(".meta-item span");
const metaStatusLabel = metaLabels[0] || null;
const metaDeviceLabel = metaLabels[1] || null;
const metaLanguageLabel = metaLabels[2] || null;
const metaSegmentsLabel = metaLabels[3] || null;
const currentTextLabel = document.querySelector(".now-playing span");
const liveMetricLabels = document.querySelectorAll(".live-metric-top span");
const inputLevelLabel = liveMetricLabels[0] || null;
const nextBufferLabel = liveMetricLabels[1] || null;

const busyStates = new Set(["queued", "running"]);
const liveActiveStates = new Set(["starting", "running", "stopping"]);
const defaultBufferedIntervalSeconds = 60;
const defaultTheme = "sand";
const defaultVerboseDescriptions = true;
const validThemes = new Set(["sand", "ocean", "ember", "evergreen"]);
const themeSelects = [themeSelect].filter(Boolean);
const verboseToggleInputs = [verboseDescriptionsToggle].filter(Boolean);
const themeSwatches = Array.from(document.querySelectorAll(".theme-swatch"));

const uiText = {
  verbose: {
    eyebrow: "Local GPU-First Transcription",
    title: "Speech-to-Text Meeting Helper",
    subtitle: "Upload audio, track progress, preview transcript, and download TXT/SRT outputs.",
    methodSwitchAria: "Transcription method",
    fileModeBtn: "File Transcription",
    liveModeBtn: "Live Transcription",
    dropTitle: "Drop audio here",
    dropSubtitle: "or click to choose a file",
    noFileSelected: "No file selected",
    modelLabel: "Model",
    exportFolderLabel: "Export Folder",
    fileMenuLabel: "Settings",
    liveMenuLabel: "Settings",
    pageThemeLabel: "Page Theme",
    verboseDescriptionsLabel: "Verbose descriptions",
    diarizeLabel: "Label speakers",
    hfTokenLabel: "Hugging Face Token",
    expectedSpeakersLabel: "Expected Speakers",
    autoDetectPlaceholder: "Auto detect",
    startTranscription: "Start Transcription",
    working: "Working...",
    loadPairBtn: "Upload Existing Audio + SRT",
    noLocalPair: "No local audio/SRT pair loaded",
    liveSourceLabel: "Live Source",
    captureModeLabel: "Capture Mode",
    liveDeviceLabel: "Live Device",
    onStartLabel: "On Start",
    liveModelLabel: "Live Model",
    liveLanguageLabel: "Live Language",
    liveDiarizeLabel: "Label speakers (buffered HQ only)",
    liveExpectedSpeakersLabel: "Live Expected Speakers",
    modelWarmupLabel: "Model Warmup",
    preloadSelectedModel: "Pre-load Selected Model",
    preloading: "Pre-loading...",
    noLivePreload: "No live model pre-loaded in this session",
    startLive: "Start Live Transcription",
    stopLive: "Stop Live Transcription",
    stopping: "Stopping...",
    inputLevelLabel: "Input Level",
    nextBufferLabel: "Next Buffer Update",
    outputsHeading: "Outputs",
    downloadsTriggerLabel: "Downloads",
    downloadsHintEmpty: "No files yet",
    playerHeading: "Player + Transcript",
    currentTextLabel: "Current Text",
    metaStatusLabel: "Status",
    metaDeviceLabel: "Device",
    metaLanguageLabel: "Language",
    metaSegmentsLabel: "Segments",
    downloadTxt: "Download TXT",
    downloadSrt: "Download SRT",
    downloadMarkdown: "Download Markdown",
    downloadAudio: "Captured WAV",
    downloadAudio16k: "Whisper 16k WAV",
    downloadDiagnostics: "Diagnostics JSON",
    transcriptToggleReading: "Reading View",
    transcriptToggleTimeline: "Timeline View",
    readingViewHint: "Short paragraphs without timestamps",
    readingViewBackHint: "Return to the timestamped transcript",
    transcriptEmpty: "No segments found.",
    noSystemDevice: "No system device found",
    noMicDevice: "No microphone found",
    speakerUnit: "speakers",
    idleMessage: "Drop an audio file to begin.",
  },
  compact: {
    eyebrow: "Local GPU STT",
    title: "Meeting Helper",
    subtitle: "Upload, transcribe, export.",
    methodSwitchAria: "Mode",
    fileModeBtn: "Files",
    liveModeBtn: "Live",
    dropTitle: "Drop audio",
    dropSubtitle: "or browse",
    noFileSelected: "No file",
    modelLabel: "Model",
    exportFolderLabel: "Folder",
    fileMenuLabel: "Menu",
    liveMenuLabel: "Menu",
    pageThemeLabel: "Theme",
    verboseDescriptionsLabel: "Verbose text",
    diarizeLabel: "Speaker labels",
    hfTokenLabel: "HF Token",
    expectedSpeakersLabel: "# Speakers",
    autoDetectPlaceholder: "Auto",
    startTranscription: "Transcribe",
    working: "Working...",
    loadPairBtn: "Load Audio + SRT",
    noLocalPair: "No pair loaded",
    liveSourceLabel: "Source",
    captureModeLabel: "Mode",
    liveDeviceLabel: "Device",
    onStartLabel: "Start",
    liveModelLabel: "Model",
    liveLanguageLabel: "Lang",
    liveDiarizeLabel: "Speaker labels (HQ)",
    liveExpectedSpeakersLabel: "# Speakers",
    modelWarmupLabel: "Warmup",
    preloadSelectedModel: "Pre-load",
    preloading: "Loading...",
    noLivePreload: "Not preloaded",
    startLive: "Start Live",
    stopLive: "Stop Live",
    stopping: "Stopping...",
    inputLevelLabel: "Input",
    nextBufferLabel: "Buffer",
    outputsHeading: "Files",
    downloadsTriggerLabel: "Files",
    downloadsHintEmpty: "No files",
    playerHeading: "Player",
    currentTextLabel: "Now",
    metaStatusLabel: "State",
    metaDeviceLabel: "Device",
    metaLanguageLabel: "Lang",
    metaSegmentsLabel: "Segs",
    downloadTxt: "TXT",
    downloadSrt: "SRT",
    downloadMarkdown: "MD",
    downloadAudio: "WAV",
    downloadAudio16k: "16k",
    downloadDiagnostics: "JSON",
    transcriptToggleReading: "Chunks",
    transcriptToggleTimeline: "Timeline",
    readingViewHint: "Short paras",
    readingViewBackHint: "Show times",
    transcriptEmpty: "No text.",
    noSystemDevice: "No system device",
    noMicDevice: "No mic found",
    speakerUnit: "spk",
    idleMessage: "Drop audio to start.",
  },
};

const selectText = {
  fileModel: {
    small: { verbose: "small (recommended)", compact: "small" },
    tiny: { verbose: "tiny", compact: "tiny" },
    base: { verbose: "base", compact: "base" },
    medium: { verbose: "medium", compact: "medium" },
    "large-v3": { verbose: "large-v3", compact: "large-v3" },
  },
  liveSource: {
    system: { verbose: "System Audio (Loopback)", compact: "System" },
    mic: { verbose: "Microphone", compact: "Mic" },
  },
  liveCaptureMode: {
    low_latency: { verbose: "Low Latency (faster text)", compact: "Fast" },
    buffered_hq: { verbose: "Buffered HQ (best quality)", compact: "HQ" },
  },
  liveMode: {
    append: { verbose: "Continue existing text", compact: "Append" },
    new: { verbose: "Start new transcript", compact: "New" },
  },
  liveModel: {
    small: { verbose: "small (recommended)", compact: "small" },
    tiny: { verbose: "tiny (lowest quality)", compact: "tiny" },
    base: { verbose: "base", compact: "base" },
    medium: { verbose: "medium", compact: "medium" },
    "large-v3": { verbose: "large-v3 (best quality)", compact: "large-v3" },
  },
  liveLanguage: {
    en: { verbose: "English (recommended)", compact: "English" },
    auto: { verbose: "Auto Detect", compact: "Auto" },
  },
};

const compactStatusLabelMap = {
  IDLE: "IDLE",
  READY: "OK",
  QUEUED: "QUEUE",
  RUNNING: "RUN",
  UPLOADING: "UP",
  LOADING: "LOAD",
  COMPLETED: "DONE",
  FAILED: "FAIL",
  ERROR: "ERR",
  LIVE: "LIVE",
  "LIVE ERROR": "L-ERR",
};

const compactStatusMessageMap = {
  "Drop an audio file to begin.": uiText.compact.idleMessage,
  "Failed to fetch job status.": "Status fetch failed.",
  "Failed to load transcript segments.": "Segment load failed.",
  "Starting live transcription...": "Starting live...",
  "Stopping live transcription...": "Stopping live...",
  "Live transcription failed.": "Live failed.",
  "Live transcription is idle.": "Live idle.",
  "Live stream connection lost. Refreshing state...": "Live reconnecting...",
  "Select a live capture device first.": "Select device.",
  "Live diarization requires Buffered HQ mode.": "Diarization needs HQ.",
  "Live expected speakers must be an integer from 1 to 20.": "Speakers: 1-20.",
  "Choose both an audio file and an SRT file.": "Pick audio + SRT.",
  "Loading local audio and SRT...": "Loading pair...",
  "Loaded local audio and SRT for playback.": "Pair loaded.",
  "Failed to load local files.": "Load failed.",
  "Stop live transcription before loading local files.": "Stop live first.",
  "Stop live transcription before uploading a file.": "Stop live first.",
  "Select an audio file first.": "Pick audio first.",
  "Expected speakers must be an integer from 1 to 20.": "Speakers: 1-20.",
  "Uploading audio...": "Uploading...",
  "Job queued.": "Queued.",
  "Queued": "Queued",
  "Loading model": "Loading model...",
  "Analyzing audio": "Analyzing...",
  "Transcribing": "Transcribing...",
  "Running speaker diarization": "Finding speakers...",
  "Finalizing outputs": "Wrapping up...",
  "No transcript segments were produced.": "No transcript.",
  "No segments found.": uiText.compact.transcriptEmpty,
  "Preparing...": "Prep...",
  "Failed to preload model.": "Preload failed.",
};

const applyTheme = (theme, options = {}) => {
  const persist = options.persist !== false;
  const nextTheme = validThemes.has(theme) ? theme : defaultTheme;
  document.documentElement.dataset.theme = nextTheme;
  for (const select of themeSelects) {
    if (select.value !== nextTheme) {
      select.value = nextTheme;
    }
  }
  for (const swatch of themeSwatches) {
    const isActive = swatch.getAttribute("data-theme-choice") === nextTheme;
    swatch.classList.toggle("active", isActive);
    swatch.setAttribute("aria-pressed", isActive ? "true" : "false");
  }
  if (!persist) {
    return;
  }
  try {
    window.localStorage.setItem("meetingHelperTheme", nextTheme);
  } catch {
    // Ignore storage failures and still apply the theme for this session.
  }
};

const loadSavedTheme = () => {
  try {
    return window.localStorage.getItem("meetingHelperTheme") || defaultTheme;
  } catch {
    return defaultTheme;
  }
};

const loadSavedVerboseDescriptions = () => {
  try {
    const value = window.localStorage.getItem("meetingHelperVerboseDescriptions");
    if (value === null) {
      return defaultVerboseDescriptions;
    }
    return !["0", "false", "off", "no"].includes(String(value).trim().toLowerCase());
  } catch {
    return defaultVerboseDescriptions;
  }
};

const copyFor = (key) => {
  const mode = state.verboseDescriptions ? "verbose" : "compact";
  return uiText[mode][key] ?? uiText.verbose[key] ?? "";
};

const setSelectCopy = (select, labels) => {
  if (!select) {
    return;
  }
  const mode = state.verboseDescriptions ? "verbose" : "compact";
  for (const [value, variants] of Object.entries(labels)) {
    const option = Array.from(select.options).find((item) => item.value === value);
    if (!option) {
      continue;
    }
    option.textContent = variants[mode] ?? variants.verbose ?? option.textContent;
  }
};

const renderLiveMetricLabels = () => {
  if (inputLevelLabel) {
    const selectedSourceLabel = liveSource?.selectedOptions?.[0]?.textContent?.trim()
      || liveSource?.options?.[liveSource.selectedIndex]?.textContent?.trim()
      || copyFor("liveSourceLabel");
    inputLevelLabel.textContent = `${copyFor("inputLevelLabel")} (${selectedSourceLabel})`;
  }
  if (nextBufferLabel) {
    nextBufferLabel.textContent = copyFor("nextBufferLabel");
  }
};

const compactStatusLabel = (label) => {
  const normalized = String(label || "").trim();
  if (!normalized) {
    return "";
  }
  const upper = normalized.toUpperCase();
  return compactStatusLabelMap[upper] || upper;
};

const shortenLivePhrase = (value) => String(value || "")
  .replace(/system audio/gi, "system")
  .replace(/microphone/gi, "mic")
  .replace(/buffered hq/gi, "HQ")
  .replace(/low latency/gi, "fast")
  .replace(/diarization/gi, "speakers");

const compactStatusMessage = (message) => {
  const text = String(message || "");
  if (!text) {
    return "";
  }
  if (compactStatusMessageMap[text]) {
    return compactStatusMessageMap[text];
  }

  let match = text.match(/^Transcribing segment (\d+)$/i);
  if (match) {
    return `Seg ${match[1]}...`;
  }

  match = text.match(/^Listening to (.+) \((.+)\)(.*)\.\.\.$/i);
  if (match) {
    const source = shortenLivePhrase(match[1]);
    const mode = shortenLivePhrase(match[2]);
    const suffix = String(match[3] || "")
      .replace(/\+ diarization/gi, "+ speakers")
      .replace(/\(diarization failed\)/gi, "(speaker fail)");
    return `Listening: ${source} (${mode})${suffix}`;
  }

  match = text.match(/^Starting (.+?) transcription\.\.\.$/i);
  if (match) {
    return `Starting ${shortenLivePhrase(match[1])}...`;
  }

  match = text.match(/^Model (.+) pre-loaded\.$/i);
  if (match) {
    return `${match[1]} ready.`;
  }

  match = text.match(/^Already cached:\s*(.+)$/i);
  if (match) {
    return `Cached: ${match[1]}`;
  }

  match = text.match(/^Loaded:\s*(.+)$/i);
  if (match) {
    return `Loaded: ${match[1]}`;
  }

  match = text.match(/^Transcription completed in (.+)\. Speaker labels added\.$/i);
  if (match) {
    return `Done in ${match[1]} + speakers.`;
  }

  match = text.match(/^Transcription completed in (.+)\. Diarization failed; transcript generated without speaker labels\.$/i);
  if (match) {
    return `Done in ${match[1]} (no speakers).`;
  }

  match = text.match(/^Transcription completed in (.+)\.$/i);
  if (match) {
    return `Done in ${match[1]}.`;
  }

  if (/^Transcription completed\. Speaker labels added\.$/i.test(text)) {
    return "Done + speakers.";
  }
  if (/^Transcription completed\. Diarization failed; transcript generated without speaker labels\.$/i.test(text)) {
    return "Done (no speakers).";
  }
  if (/^Transcription completed\.$/i.test(text)) {
    return "Done.";
  }

  return text;
};

const setLivePreloadState = (mode, text = "") => {
  state.livePreloadStatusMode = mode;
  state.livePreloadText = text;
  if (livePreloadStatus) {
    if (mode === "default") {
      livePreloadStatus.textContent = copyFor("noLivePreload");
    } else if (mode === "loading") {
      livePreloadStatus.textContent = state.verboseDescriptions
        ? `Pre-loading ${liveModel.value}...`
        : `Loading ${liveModel.value}...`;
    } else {
      livePreloadStatus.textContent = state.verboseDescriptions ? text : compactStatusMessage(text);
    }
  }
};

const setLocalPairText = (text = "") => {
  state.localPairText = text;
  if (localPairName) {
    localPairName.textContent = text || copyFor("noLocalPair");
  }
};

const allDownloadAnchors = [
  downloadTxt,
  downloadSrt,
  downloadMarkdown,
  downloadAudio,
  downloadAudio16k,
  downloadDiagnostics,
].filter(Boolean);

const formatDownloadsHint = (count) => {
  const safeCount = Math.max(0, Number(count) || 0);
  if (safeCount <= 0) {
    return copyFor("downloadsHintEmpty");
  }
  if (state.verboseDescriptions) {
    return `Click to choose (${safeCount})`;
  }
  return `Choose (${safeCount})`;
};

const updateDownloadsMenu = () => {
  const availableCount = allDownloadAnchors.filter((anchor) => !anchor.classList.contains("hidden")).length;
  state.downloadOptionCount = availableCount;
  if (downloadsTriggerLabel) {
    downloadsTriggerLabel.textContent = copyFor("downloadsTriggerLabel");
  }
  if (downloadsTriggerHint) {
    downloadsTriggerHint.textContent = formatDownloadsHint(availableCount);
  }
  if (downloadsTrigger) {
    downloadsTrigger.setAttribute("aria-disabled", availableCount > 0 ? "false" : "true");
  }
  if (availableCount <= 0) {
    setDownloadsMenuOpen(false);
  }
};

const setDownloadsMenuOpen = (open) => {
  const nextOpen = Boolean(open) && state.downloadOptionCount > 0;
  state.downloadsMenuOpen = nextOpen;
  if (downloadsTrigger) {
    downloadsTrigger.setAttribute("aria-expanded", nextOpen ? "true" : "false");
  }
  if (downloadsList) {
    downloadsList.classList.toggle("hidden", !nextOpen);
  }
};

const renderStatus = () => {
  statusLabel.textContent = state.verboseDescriptions
    ? state.statusRawLabel
    : compactStatusLabel(state.statusRawLabel);
  statusMessage.textContent = state.verboseDescriptions
    ? state.statusRawMessage
    : compactStatusMessage(state.statusRawMessage);
};

const setMetaStatusValue = (value) => {
  const raw = String(value ?? "-");
  state.metaStatusRaw = raw;
  if (state.verboseDescriptions) {
    metaStatus.textContent = raw;
    return;
  }
  const compactMetaStatusMap = {
    completed: "done",
    failed: "fail",
    local: "local",
    "live-running": "live-run",
    "live-starting": "live-start",
    "live-stopping": "live-stop",
    "live-idle": "live-idle",
    "live-error": "live-err",
  };
  metaStatus.textContent = compactMetaStatusMap[raw] || raw;
};

const renderCopyMode = () => {
  if (methodSwitch) {
    methodSwitch.setAttribute("aria-label", copyFor("methodSwitchAria"));
  }
  if (heroEyebrow) {
    heroEyebrow.textContent = copyFor("eyebrow");
  }
  if (heroTitle) {
    heroTitle.textContent = copyFor("title");
  }
  if (heroSubtitle) {
    heroSubtitle.textContent = copyFor("subtitle");
  }
  fileModeBtn.textContent = copyFor("fileModeBtn");
  liveModeBtn.textContent = copyFor("liveModeBtn");
  if (dropTitleLabel) {
    dropTitleLabel.textContent = copyFor("dropTitle");
  }
  if (dropSubtitleLabel) {
    dropSubtitleLabel.textContent = copyFor("dropSubtitle");
  }
  if (!state.file) {
    fileName.textContent = copyFor("noFileSelected");
  }

  if (modelLabel) {
    modelLabel.textContent = copyFor("modelLabel");
  }
  if (exportFolderLabel) {
    exportFolderLabel.textContent = copyFor("exportFolderLabel");
  }
  const menuCopyKey = state.uiMode === "live" ? "liveMenuLabel" : "fileMenuLabel";
  if (settingsMenuBtn) {
    const menuLabel = copyFor(menuCopyKey);
    settingsMenuBtn.setAttribute("aria-label", menuLabel);
    settingsMenuBtn.setAttribute("title", menuLabel);
  }
  if (settingsPanelTitle) {
    settingsPanelTitle.textContent = copyFor(menuCopyKey);
  }
  if (fileThemeLabel) {
    fileThemeLabel.textContent = copyFor("pageThemeLabel");
  }
  if (liveThemeLabel) {
    liveThemeLabel.textContent = copyFor("pageThemeLabel");
  }
  if (verboseDescriptionsLabel) {
    verboseDescriptionsLabel.textContent = copyFor("verboseDescriptionsLabel");
  }
  if (liveVerboseDescriptionsLabel) {
    liveVerboseDescriptionsLabel.textContent = copyFor("verboseDescriptionsLabel");
  }
  if (diarizeLabel) {
    diarizeLabel.textContent = copyFor("diarizeLabel");
  }
  if (tokenLabel) {
    tokenLabel.textContent = copyFor("hfTokenLabel");
  }
  if (expectedSpeakersLabel) {
    expectedSpeakersLabel.textContent = copyFor("expectedSpeakersLabel");
  }
  if (liveSourceLabel) {
    liveSourceLabel.textContent = copyFor("liveSourceLabel");
  }
  if (liveCaptureModeLabel) {
    liveCaptureModeLabel.textContent = copyFor("captureModeLabel");
  }
  if (liveDeviceLabel) {
    liveDeviceLabel.textContent = copyFor("liveDeviceLabel");
  }
  if (liveModeLabel) {
    liveModeLabel.textContent = copyFor("onStartLabel");
  }
  if (liveModelLabel) {
    liveModelLabel.textContent = copyFor("liveModelLabel");
  }
  if (liveLanguageLabel) {
    liveLanguageLabel.textContent = copyFor("liveLanguageLabel");
  }
  if (liveDiarizeLabel) {
    liveDiarizeLabel.textContent = copyFor("liveDiarizeLabel");
  }
  if (liveTokenLabel) {
    liveTokenLabel.textContent = copyFor("hfTokenLabel");
  }
  if (liveExpectedSpeakersLabel) {
    liveExpectedSpeakersLabel.textContent = copyFor("liveExpectedSpeakersLabel");
  }
  if (modelWarmupLabel) {
    modelWarmupLabel.textContent = copyFor("modelWarmupLabel");
  }
  if (outputsHeading) {
    outputsHeading.textContent = copyFor("outputsHeading");
  }
  if (playerHeading) {
    playerHeading.textContent = copyFor("playerHeading");
  }
  if (currentTextLabel) {
    currentTextLabel.textContent = copyFor("currentTextLabel");
  }
  if (metaStatusLabel) {
    metaStatusLabel.textContent = copyFor("metaStatusLabel");
  }
  if (metaDeviceLabel) {
    metaDeviceLabel.textContent = copyFor("metaDeviceLabel");
  }
  if (metaLanguageLabel) {
    metaLanguageLabel.textContent = copyFor("metaLanguageLabel");
  }
  if (metaSegmentsLabel) {
    metaSegmentsLabel.textContent = copyFor("metaSegmentsLabel");
  }

  diarizeSpeakers.placeholder = copyFor("autoDetectPlaceholder");
  liveDiarizeSpeakers.placeholder = copyFor("autoDetectPlaceholder");
  loadExistingBtn.textContent = copyFor("loadPairBtn");
  downloadTxt.textContent = copyFor("downloadTxt");
  downloadSrt.textContent = copyFor("downloadSrt");
  downloadMarkdown.textContent = copyFor("downloadMarkdown");
  downloadAudio.textContent = copyFor("downloadAudio");
  downloadAudio16k.textContent = copyFor("downloadAudio16k");
  downloadDiagnostics.textContent = copyFor("downloadDiagnostics");

  setSelectCopy(modelSelect, selectText.fileModel);
  setSelectCopy(liveSource, selectText.liveSource);
  setSelectCopy(liveCaptureMode, selectText.liveCaptureMode);
  setSelectCopy(liveMode, selectText.liveMode);
  setSelectCopy(liveModel, selectText.liveModel);
  setSelectCopy(liveLanguage, selectText.liveLanguage);
  renderLiveMetricLabels();

  setLocalPairText(state.localPairText);
  setLivePreloadState(state.livePreloadStatusMode, state.livePreloadText);
  setMetaStatusValue(state.metaStatusRaw);
  setMetaSegments(state.metaSegmentCountRaw, state.metaSpeakerCountRaw);
  renderReadingView();
  renderTranscriptMode();
  updateDownloadsMenu();
  renderStatus();

  if (!state.segments.length && transcriptList.textContent) {
    if (
      transcriptList.textContent === uiText.verbose.transcriptEmpty
      || transcriptList.textContent === uiText.compact.transcriptEmpty
    ) {
      transcriptList.textContent = copyFor("transcriptEmpty");
    }
  }
};

const applyDescriptionMode = (verbose, options = {}) => {
  const persist = options.persist !== false;
  state.verboseDescriptions = Boolean(verbose);
  for (const toggle of verboseToggleInputs) {
    toggle.checked = state.verboseDescriptions;
  }
  renderCopyMode();
  if (liveDevice) {
    populateLiveDeviceOptions(liveSource.value, liveDevice.value || null);
  }
  refreshControls();
  if (!persist) {
    return;
  }
  try {
    window.localStorage.setItem("meetingHelperVerboseDescriptions", state.verboseDescriptions ? "true" : "false");
  } catch {
    // Ignore storage failures and still update this session.
  }
};

const formatTimestamp = (seconds) => {
  const total = Math.floor(Number(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
};

const formatSegmentText = (segment) => {
  const text = String(segment.text ?? "").trim();
  const speaker = String(segment.speaker ?? "").trim();
  if (!speaker) {
    return text;
  }
  return `${speaker}: ${text}`;
};

const sanitizeOutputStem = (value) => String(value || "")
  .trim()
  .replace(/\.[^/.\\]+$/, "")
  .replace(/[<>:"/\\|?*\u0000-\u001F]+/g, "-")
  .replace(/\s+/g, "-")
  .replace(/-+/g, "-")
  .replace(/^[-.]+|[-.]+$/g, "");

const setOutputStem = (value) => {
  const fallback = state.uiMode === "live" ? "live-transcript" : "transcript";
  state.outputStem = sanitizeOutputStem(value) || fallback;
};

const getOutputStem = () => {
  if (state.outputStem) {
    return state.outputStem;
  }
  if (state.uiMode === "live") {
    return "live-transcript";
  }
  if (state.file?.name) {
    return sanitizeOutputStem(state.file.name) || "transcript";
  }
  return "transcript";
};

const collapseWhitespace = (value) => String(value || "").replace(/\s+/g, " ").trim();

const sentenceBoundaryPattern = /[.!?]+["')\]]*$/;

const sentenceEnds = (value) => sentenceBoundaryPattern.test(collapseWhitespace(value));

const splitSentenceTokens = (value) => {
  const text = collapseWhitespace(value);
  if (!text) {
    return [];
  }

  const tokens = [];
  const boundaryPattern = /[.!?]+["')\]]*(?=\s+)/g;
  let start = 0;
  let match = boundaryPattern.exec(text);
  while (match) {
    const end = match.index + match[0].length;
    const token = text.slice(start, end).trim();
    if (token) {
      tokens.push(token);
    }
    start = end;
    while (text[start] === " ") {
      start += 1;
    }
    match = boundaryPattern.exec(text);
  }

  const remainder = text.slice(start).trim();
  if (remainder) {
    tokens.push(remainder);
  }
  return tokens;
};

const getMinuteStart = (seconds) => {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) {
    return null;
  }
  return Math.floor(value / 60) * 60;
};

const buildReadingParagraphItems = (segments) => {
  const paragraphs = [];
  let currentParts = [];
  let currentChars = 0;
  let currentSpeaker = null;
  let currentStart = null;
  let currentTokenCount = 0;

  const softCharLimit = 420;
  const hardCharLimit = 900;
  const softTokenLimit = 8;

  const flush = () => {
    if (!currentParts.length) {
      return;
    }
    paragraphs.push({
      start: currentStart,
      text: currentParts.join(" ").trim(),
    });
    currentParts = [];
    currentChars = 0;
    currentSpeaker = null;
    currentStart = null;
    currentTokenCount = 0;
  };

  for (const segment of segments) {
    const tokens = splitSentenceTokens(segment.text);
    if (!tokens.length) {
      continue;
    }

    const speaker = collapseWhitespace(segment.speaker);
    const speakerChanged = currentSpeaker !== null && speaker !== currentSpeaker;
    if (speakerChanged) {
      flush();
    }

    for (const token of tokens) {
      let piece = token;
      if (speaker) {
        piece = currentParts.length ? token : `${speaker}: ${token}`;
      }

      const projectedChars = currentChars + piece.length + (currentParts.length ? 1 : 0);
      const canBreakCleanly = currentParts.length && sentenceEnds(currentParts[currentParts.length - 1]);
      if (
        currentParts.length
        && (
          (canBreakCleanly && (projectedChars > softCharLimit || currentTokenCount >= softTokenLimit))
          || projectedChars > hardCharLimit
        )
      ) {
        flush();
        piece = speaker ? `${speaker}: ${token}` : token;
      }

      if (!currentParts.length) {
        currentSpeaker = speaker || "";
        currentStart = Number.isFinite(Number(segment.start)) ? Number(segment.start) : null;
      }

      currentParts.push(piece);
      currentChars += piece.length + (currentParts.length > 1 ? 1 : 0);
      currentTokenCount += 1;

      if (sentenceEnds(token) && currentChars >= 300) {
        flush();
      }
    }
  }

  flush();
  return paragraphs;
};

const buildReadingParagraphs = (segments) => buildReadingParagraphItems(segments)
  .map((paragraph) => paragraph.text);

const buildMarkdownFromSegments = (segments, title = "transcript") => {
  const paragraphs = buildReadingParagraphItems(segments);
  const heading = sanitizeOutputStem(title) || "transcript";
  const lines = [`# ${heading.replace(/-/g, " ")}`, ""];
  if (!paragraphs.length) {
    lines.push(copyFor("transcriptEmpty"));
  } else {
    let currentMinuteStart = null;
    for (const paragraph of paragraphs) {
      const minuteStart = getMinuteStart(paragraph.start);
      if (minuteStart !== null && minuteStart !== currentMinuteStart) {
        currentMinuteStart = minuteStart;
        lines.push(`## ${formatTimestamp(minuteStart)}`, "");
      }
      lines.push(paragraph.text, "");
    }
  }
  return `${lines.join("\n").trimEnd()}\n`;
};

const renderReadingView = () => {
  if (!readingView) {
    return;
  }

  const paragraphs = buildReadingParagraphs(state.segments);
  if (!paragraphs.length) {
    readingView.textContent = copyFor("transcriptEmpty");
    return;
  }

  readingView.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const paragraph of paragraphs) {
    const node = document.createElement("p");
    node.className = "reading-paragraph";
    node.textContent = paragraph;
    fragment.appendChild(node);
  }
  readingView.appendChild(fragment);
};

const renderTranscriptMode = () => {
  const readingMode = state.transcriptView === "reading";
  if (transcriptList) {
    transcriptList.classList.toggle("hidden", readingMode);
  }
  if (readingView) {
    readingView.classList.toggle("hidden", !readingMode);
  }
  if (readingViewBtn) {
    const hasSegments = state.segments.length > 0;
    readingViewBtn.disabled = !hasSegments;
    readingViewBtn.classList.toggle("active", readingMode);
    readingViewBtn.setAttribute("aria-pressed", readingMode ? "true" : "false");
    readingViewBtn.textContent = copyFor(readingMode ? "transcriptToggleTimeline" : "transcriptToggleReading");
  }
  if (readingViewHint) {
    readingViewHint.textContent = copyFor(readingMode ? "readingViewBackHint" : "readingViewHint");
  }
};

const setTranscriptView = (mode) => {
  state.transcriptView = mode === "reading" ? "reading" : "timeline";
  renderReadingView();
  renderTranscriptMode();
};

const syncMarkdownDownload = (stem = getOutputStem()) => {
  if (state.localDownloadUrls.markdown) {
    URL.revokeObjectURL(state.localDownloadUrls.markdown);
    state.localDownloadUrls.markdown = null;
  }

  if (!downloadMarkdown) {
    return;
  }

  downloadMarkdown.removeAttribute("href");
  downloadMarkdown.removeAttribute("download");
  downloadMarkdown.classList.add("hidden");

  if (!state.segments.length) {
    updateDownloadsMenu();
    return;
  }

  const resolvedStem = sanitizeOutputStem(stem) || getOutputStem();
  const markdownBlob = new Blob(
    [buildMarkdownFromSegments(state.segments, resolvedStem)],
    { type: "text/markdown;charset=utf-8" },
  );
  const markdownUrl = URL.createObjectURL(markdownBlob);

  state.localDownloadUrls.markdown = markdownUrl;
  downloadMarkdown.href = markdownUrl;
  downloadMarkdown.download = `${resolvedStem}.md`;
  downloadMarkdown.classList.remove("hidden");
  updateDownloadsMenu();
};

const setMetaSegments = (segmentCount, speakerCount = null) => {
  if (segmentCount === null || segmentCount === undefined || segmentCount === "-") {
    state.metaSegmentCountRaw = null;
    state.metaSpeakerCountRaw = null;
    metaSegments.textContent = "-";
    return;
  }
  const count = Number(segmentCount);
  if (!Number.isFinite(count)) {
    state.metaSegmentCountRaw = null;
    state.metaSpeakerCountRaw = null;
    metaSegments.textContent = "-";
    return;
  }
  const safeCount = Number.isFinite(count) ? Math.max(0, Math.trunc(count)) : 0;
  const speakers = Number(speakerCount);
  state.metaSegmentCountRaw = safeCount;
  state.metaSpeakerCountRaw = Number.isFinite(speakers) ? Math.trunc(speakers) : null;
  if (Number.isFinite(speakers) && speakers > 0) {
    metaSegments.textContent = `${safeCount} (${Math.trunc(speakers)} ${copyFor("speakerUnit")})`;
    return;
  }
  metaSegments.textContent = String(safeCount);
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
    bufferCountdownLabel.textContent = state.verboseDescriptions ? "Preparing..." : "Prep...";
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
  state.statusRawLabel = String(label || "");
  state.statusRawMessage = String(message || "");
  renderStatus();
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
  if (state.localDownloadUrls.markdown) {
    URL.revokeObjectURL(state.localDownloadUrls.markdown);
    state.localDownloadUrls.markdown = null;
  }
};

const clearDownloadLinks = () => {
  revokeLocalDownloads();
  for (const anchor of allDownloadAnchors) {
    anchor.removeAttribute("href");
    anchor.removeAttribute("download");
    anchor.classList.add("hidden");
  }
  updateDownloadsMenu();
};

const setServerDownloads = (downloads) => {
  clearDownloadLinks();
  if (!downloads) {
    syncMarkdownDownload();
    return;
  }
  if (downloads.txt) {
    downloadTxt.href = downloads.txt;
    downloadTxt.classList.remove("hidden");
  }
  if (downloads.srt) {
    downloadSrt.href = downloads.srt;
    downloadSrt.classList.remove("hidden");
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
  syncMarkdownDownload();
  updateDownloadsMenu();
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
    lines.push(`[${formatTimestamp(segment.start)}] ${formatSegmentText(segment)}`);
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
    lines.push(formatSegmentText(segment));
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
  downloadTxt.classList.remove("hidden");
  downloadSrt.classList.remove("hidden");
  syncMarkdownDownload(stem);
  updateDownloadsMenu();
};

const setFile = (file) => {
  state.file = file;
  if (file?.name) {
    setOutputStem(file.name);
  }
  fileName.textContent = file ? file.name : copyFor("noFileSelected");
};

const isLiveActive = () => liveActiveStates.has(state.liveStatus);

const syncSettingsSections = () => {
  const isLiveMode = state.uiMode === "live";
  if (fileSettingsSection) {
    fileSettingsSection.classList.toggle("hidden", isLiveMode);
  }
  if (liveSettingsSection) {
    liveSettingsSection.classList.toggle("hidden", !isLiveMode);
  }
};

const setSettingsMenuOpen = (open, options = {}) => {
  const returnFocus = options.returnFocus !== false;
  state.settingsMenuOpen = Boolean(open);
  if (settingsOverlay) {
    settingsOverlay.classList.toggle("open", state.settingsMenuOpen);
    settingsOverlay.setAttribute("aria-hidden", state.settingsMenuOpen ? "false" : "true");
  }
  if (settingsMenuBtn) {
    settingsMenuBtn.setAttribute("aria-expanded", state.settingsMenuOpen ? "true" : "false");
  }
  document.body.classList.toggle("settings-open", state.settingsMenuOpen);
  if (state.settingsMenuOpen) {
    window.setTimeout(() => {
      settingsCloseBtn?.focus();
    }, 0);
  } else if (returnFocus) {
    settingsMenuBtn?.focus();
  }
};


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
  syncSettingsSections();
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
    option.textContent = source === "system" ? copyFor("noSystemDevice") : copyFor("noMicDevice");
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
  modelSelect.disabled = lockFileActions;
  exportFolder.disabled = lockFileActions;
  startBtn.disabled = lockFileActions;
  loadExistingBtn.disabled = lockFileActions;
  dropZone.disabled = lockFileActions;
  if (diarizeToggle) {
    diarizeToggle.disabled = lockFileActions;
  }
  if (diarizationToken) {
    diarizationToken.disabled = lockFileActions || !(diarizeToggle && diarizeToggle.checked);
  }
  if (diarizeSpeakers) {
    diarizeSpeakers.disabled = lockFileActions || !(diarizeToggle && diarizeToggle.checked);
  }

  const lockLiveButton = state.batchBusy || state.livePending || state.liveStatus === "stopping";
  liveToggleBtn.disabled = lockLiveButton;

  const lockLiveSelectors = state.batchBusy || isLiveActive() || state.livePending;
  const liveDiarizationEligible = liveCaptureMode.value === "buffered_hq";
  liveSource.disabled = lockLiveSelectors;
  liveCaptureMode.disabled = lockLiveSelectors;
  liveDevice.disabled = lockLiveSelectors || !devicesForSource(liveSource.value).length;
  liveMode.disabled = lockLiveSelectors;
  liveModel.disabled = lockLiveSelectors;
  liveLanguage.disabled = lockLiveSelectors;
  if (liveDiarizeToggle) {
    if (!liveDiarizationEligible && !lockLiveSelectors) {
      liveDiarizeToggle.checked = false;
    }
    liveDiarizeToggle.disabled = lockLiveSelectors || !liveDiarizationEligible;
  }
  if (liveDiarizationToken) {
    const tokenEnabled = liveDiarizationEligible && liveDiarizeToggle && liveDiarizeToggle.checked;
    liveDiarizationToken.disabled = lockLiveSelectors || !tokenEnabled;
  }
  if (liveDiarizeSpeakers) {
    const speakersEnabled = liveDiarizationEligible && liveDiarizeToggle && liveDiarizeToggle.checked;
    if (!speakersEnabled && !lockLiveSelectors) {
      liveDiarizeSpeakers.value = "";
    }
    liveDiarizeSpeakers.disabled = lockLiveSelectors || !speakersEnabled;
  }
  livePreloadBtn.disabled = lockLiveSelectors || state.livePreloadPending;
  livePreloadBtn.textContent = state.livePreloadPending ? copyFor("preloading") : copyFor("preloadSelectedModel");

  if (state.batchBusy) {
    startBtn.textContent = copyFor("working");
  } else {
    startBtn.textContent = copyFor("startTranscription");
  }

  if (isLiveActive()) {
    liveToggleBtn.textContent = state.liveStatus === "stopping" ? copyFor("stopping") : copyFor("stopLive");
    liveToggleBtn.classList.add("running");
  } else {
    liveToggleBtn.textContent = copyFor("startLive");
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
  if (readingView) {
    readingView.textContent = copyFor("transcriptEmpty");
  }
  currentTextBox.value = "";
  state.segments = [];
  state.lineElements = [];
  state.activeLineIndex = -1;
  renderTranscriptMode();
};

const resetResults = () => {
  resultPanel.classList.add("hidden");
  setMetaStatusValue("-");
  metaDevice.textContent = "-";
  metaLanguage.textContent = "-";
  setMetaSegments(null, null);
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
  currentTextBox.value = formatSegmentText(state.segments[lineIndex]);

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
    speaker: String(segment.speaker ?? "").trim(),
  }))
  .filter((segment) => Number.isFinite(segment.start) && segment.text.length > 0)
  .sort((a, b) => a.start - b.start)
  .map((segment, index) => ({
    index: index + 1,
    start: segment.start,
    end: Number.isFinite(segment.end) ? segment.end : segment.start,
    text: segment.text,
    speaker: segment.speaker,
  }));

const buildSegmentRow = (segment, rowIndex) => {
  const row = document.createElement("button");
  row.type = "button";
  row.className = "line";
  row.dataset.index = String(segment.index);
  row.dataset.start = String(segment.start);

  const timeLabel = document.createElement("span");
  timeLabel.className = "line-time";
  timeLabel.textContent = formatTimestamp(segment.start);

  const body = document.createElement("span");
  body.className = "line-meta";

  if (segment.speaker) {
    const speakerTag = document.createElement("span");
    speakerTag.className = "speaker-tag";
    speakerTag.textContent = segment.speaker;
    body.appendChild(speakerTag);
  }

  const textLabel = document.createElement("span");
  textLabel.className = "line-text";
  textLabel.textContent = segment.text;
  body.appendChild(textLabel);

  row.appendChild(timeLabel);
  row.appendChild(body);
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
  renderReadingView();
  syncMarkdownDownload();
  renderTranscriptMode();

  if (!segments.length) {
    transcriptList.textContent = copyFor("transcriptEmpty");
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
  renderTranscriptMode();
};

const appendSegment = (rawSegment) => {
  const normalized = normalizeSegments([rawSegment])[0];
  if (!normalized) {
    return;
  }
  if (
    transcriptList.textContent === uiText.verbose.transcriptEmpty
    || transcriptList.textContent === uiText.compact.transcriptEmpty
  ) {
    transcriptList.innerHTML = "";
  }

  normalized.index = state.segments.length + 1;
  state.segments.push(normalized);
  const row = buildSegmentRow(normalized, state.segments.length - 1);
  state.lineElements.push(row);
  transcriptList.appendChild(row);
  renderReadingView();
  syncMarkdownDownload();
  renderTranscriptMode();
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

    let speaker = "";
    let cleanText = text;
    const speakerMatch = text.match(/^(Speaker\s+\d+)\s*:\s*(.+)$/i);
    if (speakerMatch) {
      speaker = speakerMatch[1].trim();
      cleanText = speakerMatch[2].trim();
    }
    if (!cleanText) {
      continue;
    }

    parsed.push({
      index: parsed.length + 1,
      start,
      end,
      text: cleanText,
      speaker,
    });
  }

  return parsed;
};

const applyJobMeta = (job) => {
  setMetaStatusValue(job.status);
  metaDevice.textContent = `${job.device || "-"}${job.compute_type ? ` (${job.compute_type})` : ""}`;
  metaLanguage.textContent = job.language
    ? `${job.language} (${(Number(job.language_probability) * 100).toFixed(1)}%)`
    : "-";
  setMetaSegments(job.segment_count, job.speaker_count);
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
      let completionMessage = elapsed ? `Transcription completed in ${elapsed}.` : "Transcription completed.";
      if (job.diarize && job.diarization_status === "completed" && Number(job.speaker_count) > 0) {
        completionMessage = `${completionMessage} Speaker labels added.`;
      } else if (job.diarize && job.diarization_status === "failed") {
        completionMessage = `${completionMessage} Diarization failed; transcript generated without speaker labels.`;
      }
      setStatus("COMPLETED", completionMessage, 1);
      await handleCompleted(job);
    } else {
      setStatus("FAILED", job.error || "Transcription failed.", 1);
      setMetaStatusValue("failed");
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
  if (state.liveStatus === "running" || state.liveStatus === "starting" || Number(liveState.segment_count || 0) > 0) {
    setOutputStem("live-transcript");
  }
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
  if (Object.prototype.hasOwnProperty.call(liveState, "diarize") && liveDiarizeToggle) {
    liveDiarizeToggle.checked = Boolean(liveState.diarize);
  }
  if (Object.prototype.hasOwnProperty.call(liveState, "diarization_speakers") && liveDiarizeSpeakers) {
    const value = Number(liveState.diarization_speakers);
    liveDiarizeSpeakers.value = Number.isFinite(value) && value > 0 ? String(Math.trunc(value)) : "";
  }
  if (liveState.device_id !== undefined) {
    populateLiveDeviceOptions(liveSource.value, liveState.device_id || null);
  }
  renderLiveMetricLabels();
  applyLiveMetrics(liveState);

  setLivePending(false);
  refreshControls();

  if (!state.batchBusy) {
    if (state.liveStatus === "running") {
      const sourceLabel = liveState.source === "system" ? "system audio" : "microphone";
      const modeLabel = liveState.capture_mode === "buffered_hq" ? "buffered HQ" : "low latency";
      const diarizationEnabled = Boolean(liveState.diarize);
      const diarizationStatus = String(liveState.diarization_status || "");
      const diarizationSuffix = diarizationEnabled
        ? (
          diarizationStatus === "running"
            ? " + diarization"
            : diarizationStatus === "failed"
              ? " (diarization failed)"
              : ""
        )
        : "";
      if (preferStatusMessage) {
        setStatus("LIVE", `Listening to ${sourceLabel} (${modeLabel})${diarizationSuffix}...`, 1);
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
    setMetaSegments(liveState.segment_count, liveState.speaker_count);
  }

  if (state.liveStatus === "running" || state.liveStatus === "starting" || state.liveStatus === "stopping") {
    resultPanel.classList.remove("hidden");
    setMetaStatusValue(`live-${state.liveStatus}`);
    metaDevice.textContent = liveState.device_label || (liveState.source === "system" ? "system loopback" : "microphone");
    metaLanguage.textContent = "-";
  } else if (state.liveStatus === "idle" && Number(liveState.segment_count || 0) > 0) {
    resultPanel.classList.remove("hidden");
    setMetaStatusValue("live-idle");
    metaDevice.textContent = liveState.device_label || (liveState.source === "system" ? "system loopback" : "microphone");
    metaLanguage.textContent = "-";
  } else if (state.liveStatus === "error") {
    setMetaStatusValue("live-error");
  }

  if (liveState.downloads) {
    setServerDownloads(liveState.downloads);
    if (liveState.downloads.audio) {
      setLivePlayerAudio(liveState.downloads.audio, liveState.session_id || null);
    }
  } else if (state.liveStatus === "running" || state.liveStatus === "starting") {
    clearDownloadLinks();
    syncMarkdownDownload("live-transcript");
  }
};

const fetchLiveSegments = async () => {
  const response = await fetch("/api/live/segments");
  if (!response.ok) {
    return;
  }
  setOutputStem("live-transcript");
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
  setOutputStem("live-transcript");
  resultPanel.classList.remove("hidden");
  appendSegment(segment);
  setMetaStatusValue("live-running");
  metaDevice.textContent = liveState?.device_label
    || (liveSource.value === "system" ? "system loopback" : "microphone");
  metaLanguage.textContent = "-";
  setMetaSegments(state.segments.length, liveState?.speaker_count);
  clearDownloadLinks();
  syncMarkdownDownload("live-transcript");
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
          setOutputStem("live-transcript");
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
  setLivePreloadState("loading");
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
    setLivePreloadState(
      "success",
      `${cachePrefix}: ${payload.model} on ${payload.device} (${payload.compute_type})${loadLabel ? ` in ${loadLabel}` : ""}`,
    );
    if (!state.batchBusy) {
      setStatus("READY", `Model ${payload.model} pre-loaded.`, 1);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to preload model.";
    setLivePreloadState("error", message);
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
  setOutputStem("live-transcript");
  setUIMode("live", { force: true });
  if (!liveDevice.value) {
    setStatus("ERROR", "Select a live capture device first.", 1);
    return;
  }

  const liveDiarizeEnabled = Boolean(liveDiarizeToggle && liveDiarizeToggle.checked);
  const liveToken = liveDiarizeEnabled && liveDiarizationToken
    ? liveDiarizationToken.value.trim()
    : "";
  let liveDiarizationSpeakers = null;
  if (liveDiarizeEnabled && liveCaptureMode.value !== "buffered_hq") {
    setStatus("ERROR", "Live diarization requires Buffered HQ mode.", 1);
    return;
  }
  if (liveDiarizeEnabled && liveDiarizeSpeakers) {
    const trimmed = liveDiarizeSpeakers.value.trim();
    if (trimmed) {
      const parsed = Number(trimmed);
      if (!Number.isInteger(parsed) || parsed < 1 || parsed > 20) {
        setStatus("ERROR", "Live expected speakers must be an integer from 1 to 20.", 1);
        return;
      }
      liveDiarizationSpeakers = parsed;
    }
  }

  setLivePending(true);
  const modeLabel = liveCaptureMode.value === "buffered_hq" ? "buffered HQ" : "low latency";
  const diarizationLabel = liveDiarizeEnabled ? " + diarization" : "";
  setStatus("LIVE", `Starting ${modeLabel}${diarizationLabel} transcription...`, 1);

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
      diarize: liveDiarizeEnabled,
      huggingface_token: liveDiarizeEnabled && liveToken ? liveToken : null,
      diarization_speakers: liveDiarizeEnabled ? liveDiarizationSpeakers : null,
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
    setMetaStatusValue("local");
    metaDevice.textContent = "browser";
    metaLanguage.textContent = "-";
    setMetaSegments(parsed.length);
    setLocalPairText(`${audioFile.name} + ${srtFile.name}`);

    const stem = srtFile.name.replace(/\.[^/.]+$/, "") || "local-transcript";
    setOutputStem(stem);
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
  renderLiveMetricLabels();
  refreshControls();
});

liveCaptureMode.addEventListener("change", () => {
  refreshControls();
});

liveModel.addEventListener("change", () => {
  if (!state.livePreloadPending) {
    setLivePreloadState("default");
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

if (settingsMenuBtn) {
  settingsMenuBtn.addEventListener("click", () => {
    setSettingsMenuOpen(!state.settingsMenuOpen, { returnFocus: !state.settingsMenuOpen });
  });
}

if (settingsBackdrop) {
  settingsBackdrop.addEventListener("click", () => {
    setSettingsMenuOpen(false);
  });
}

if (settingsCloseBtn) {
  settingsCloseBtn.addEventListener("click", () => {
    setSettingsMenuOpen(false);
  });
}

if (settingsPanel) {
  settingsPanel.addEventListener("click", (event) => {
    event.stopPropagation();
  });
}

for (const select of themeSelects) {
  select.addEventListener("change", () => {
    applyTheme(select.value);
  });
}

for (const swatch of themeSwatches) {
  swatch.addEventListener("click", (event) => {
    event.preventDefault();
    const nextTheme = swatch.getAttribute("data-theme-choice") || defaultTheme;
    applyTheme(nextTheme);
  });
}

if (downloadsTrigger) {
  downloadsTrigger.addEventListener("click", (event) => {
    event.preventDefault();
    if (state.downloadOptionCount <= 0) {
      return;
    }
    setDownloadsMenuOpen(!state.downloadsMenuOpen);
  });
}

for (const anchor of allDownloadAnchors) {
  anchor.addEventListener("click", () => {
    setDownloadsMenuOpen(false);
  });
}

for (const toggle of verboseToggleInputs) {
  toggle.addEventListener("change", () => {
    applyDescriptionMode(toggle.checked);
  });
}

if (readingViewBtn) {
  readingViewBtn.addEventListener("click", () => {
    if (!state.segments.length) {
      return;
    }
    setTranscriptView(state.transcriptView === "reading" ? "timeline" : "reading");
  });
}

if (diarizeToggle) {
  diarizeToggle.addEventListener("change", () => {
    refreshControls();
  });
}

if (liveDiarizeToggle) {
  liveDiarizeToggle.addEventListener("change", () => {
    refreshControls();
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (state.downloadsMenuOpen) {
      setDownloadsMenuOpen(false);
    }
    if (state.settingsMenuOpen) {
      setSettingsMenuOpen(false);
    }
  }
});

document.addEventListener("click", (event) => {
  if (!state.downloadsMenuOpen || !downloadsMenu) {
    return;
  }
  if (downloadsMenu.contains(event.target)) {
    return;
  }
  setDownloadsMenuOpen(false);
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
  if (diarizeToggle && diarizeToggle.checked && diarizeSpeakers) {
    const trimmed = diarizeSpeakers.value.trim();
    if (trimmed) {
      const value = Number(trimmed);
      if (!Number.isInteger(value) || value < 1 || value > 20) {
        setStatus("IDLE", "Expected speakers must be an integer from 1 to 20.", 0);
        return;
      }
    }
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
  const diarizeEnabled = Boolean(diarizeToggle && diarizeToggle.checked);
  const huggingfaceToken = diarizeEnabled && diarizationToken
    ? diarizationToken.value.trim()
    : "";
  formData.append("diarize", diarizeEnabled ? "true" : "false");
  if (diarizeEnabled && huggingfaceToken) {
    formData.append("huggingface_token", huggingfaceToken);
  }
  if (diarizeEnabled && diarizeSpeakers) {
    const trimmed = diarizeSpeakers.value.trim();
    if (trimmed) {
      formData.append("diarization_speakers", trimmed);
    }
  }

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
applyTheme(loadSavedTheme(), { persist: false });
applyDescriptionMode(loadSavedVerboseDescriptions(), { persist: false });
renderLiveDiagnostics();
refreshControls();
