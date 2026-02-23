# Speech-to-Text Meeting Helper

Local speech-to-text model with simple browser UI and player/transcript reader 

![progress bar](readme_images/STT-progressbar.gif)

The transcript can be downloaded as a .txt or .srt for synchronized captioning.  Uses `faster-whisper` with GPU-first fallback behavior with the smallest model taking ~1 min to transcribe a ~1 hour mp3 on my RTX 3080

GPU-first fallback behavior:
- tries `cuda` + `float16` first
- falls back to `cpu` + `int8` automatically

Model times on RTX 3080 for a 1 hour mp3:
| model  | time |
| ------------- | ------------- |
| tiny  | 0 min 34 s |
| small  | 0 min 57 s |
| base  | 0 min 39 s |
| medium  | 2 min 27 s |
| large-v3 | 7 min 15 s |

Note that most of the time is from loading the model — reusing the same model for multiple files results in comparable times across all models.


I'm not seeing where the actual models are stored during a cursory search, but after experimenting with all 5 models (ending with large-v3), the size of my project is still under 600 MB.


It is unclear why the base model is slightly slower than the tiny model, but the difference is too small for me to want to explore.  All models seemed to report .srt content with a desync of a few seconds.

I wanted to compare the models in quality, but both the tiny and large-v3 models seem to have done a perfect job with a recording of strongly-accented English speaker who was using several theoretical chemistry terms.  I'm not sure what the returns are supposed to be...maybe the heavier model is more tolerant of background noise or overlapping speakers?


Outputs:
- `*.transcript.txt` with `[HH:MM:SS]` line timestamps
- `*.transcript.srt` for subtitle/caption workflows

## Features

- CLI transcription (`transcribe.py`)
- Local web UI (`app.py`) with:
  - drag-and-drop upload
  - model selection
  - export folder selection (inside `data/outputs/<folder>`)
  - live status + progress bar (includes total transcription time on completion)
  - download buttons for TXT/SRT
  - audio player + clickable transcript timestamps
  - local `Upload Existing Audio + SRT` flow (no server transcription needed)

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## CLI Usage

```powershell
.\.venv\Scripts\python.exe .\transcribe.py <audio_file> [--model <model_name>]
```

Example:

```powershell
.\.venv\Scripts\python.exe .\transcribe.py "Meeting.mp3" --model small
```

### CLI Flags

- `audio_file` (required): input audio file path
- `--model` (optional, default `small`): `tiny`, `base`, `small`, `medium`, `large-v3`

## Web UI Usage

Start server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Workflow:
1. Drop/select audio file.
2. Pick model and export folder.
3. Click `Start Transcription`.
4. Track progress, then preview transcript and download TXT/SRT.

Alternative local playback workflow:
1. Click `Upload Existing Audio + SRT`.
2. Select an audio file, then select an `.srt` file.
3. Use the player and auto-updating transcript text box.

## Using `.srt` with VLC for closed-captioned listening:

- open audio file in VLC and then drag-and-drop the corresponding `.srt` file onto the VLC window to load the captions
  - alternatively, you can place the `.srt` file in the same directory as the audio file with the same base name (e.g., `Meeting.mp3` and `Meeting.srt`), and VLC should automatically load the captions when you play the audio 
- In VLC, go to `Audio -> Visualizer ->` and select a visualizer (e.g., `Spectrum`)

## Expansion Ideas (Simple -> More Advanced)

1. Batch processing for folders of recordings.
2. Additional outputs: `.vtt`, JSON metadata, word-level timings.
3. Optional diarization (speaker labels).
4. Real-time microphone streaming with partial/final transcript states.
5. Session modes for live transcription: append-to-current vs new session.
6. Post-processing: summaries, action items, decision extraction.
