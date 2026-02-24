# Speech-to-Text Meeting Helper

Local speech-to-text app built with FastAPI + `faster-whisper` and a browser UI.

![progress bar](readme_images/STT-progressbar.gif)

## Features

- File transcription jobs (`/api/jobs`)
  - drag-and-drop upload
  - model selection (`tiny`, `base`, `small`, `medium`, `large-v3`)
  - TXT + SRT outputs
  - synced player/transcript view
- Live transcription (`/api/live/*`, `/ws/live`)
  - microphone or Windows system loopback capture
  - `low_latency` and `buffered_hq` capture modes
  - websocket segment streaming
  - append-to-current vs new session
- Live diagnostics and debug artifacts
  - input level meter (`dBFS`) + buffered countdown + queue depth
  - captured input WAV download (`/api/live/download/audio`)
  - 16 kHz Whisper-input WAV download (`/api/live/download/audio_16k`)
  - per-session diagnostics JSON download (`/api/live/download/diagnostics`)
- Local playback mode (no server transcription)
  - `Upload Existing Audio + SRT`

GPU-first model loading behavior:
- tries `cuda` + `float16` first
- falls back to `cpu` + `int8`

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## File Workflow

1. Drop/select an audio file.
2. Pick model + export folder.
3. Click `Start Transcription`.
4. Download TXT/SRT and use the synced player/transcript panel.

## Live Workflow

1. Select live source (`Microphone` or `System Audio (Loopback)`).
2. Select capture mode (`Buffered HQ` or `Low Latency`).
3. Select device, model, language, and start mode (`append` or `new`).
4. Click `Start Live Transcription`.
5. Stop capture and download artifacts as needed:
   - transcript TXT/SRT
   - captured source WAV
   - 16 kHz Whisper-input WAV
   - diagnostics JSON

## Quick API Checks

```powershell
irm http://127.0.0.1:8000/api/live/state | ConvertTo-Json -Depth 6
```

## Data Paths

- uploads: `data/uploads`
- file outputs: `data/outputs/<folder>`
- live captures + diagnostics: `data/live_captures`
