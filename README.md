# Speech-to-Text Meeting Helper

Local speech-to-text for meeting audio using `faster-whisper`.
It generates:
- `*.transcript.txt` (readable transcript with `[HH:MM:SS]` timestamps)
- `*.transcript.srt` (subtitle file with start/end timestamps)

The script tries GPU first (`cuda`, `float16`) and falls back to CPU (`int8`) if needed.

## Quick Start

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install faster-whisper
.\.venv\Scripts\python.exe .\transcribe.py "Meeting.mp3"
```

Outputs:
- `Meeting.transcript.txt`
- `Meeting.transcript.srt`

## Usage

```powershell
.\.venv\Scripts\python.exe .\transcribe.py <audio_file> [--model <model_name>]
```

Example:

```powershell
.\.venv\Scripts\python.exe .\transcribe.py "Adri Meeting.mp3" --model small
```

## Command-Line Flags

- `audio_file` (required): path to audio file (for example `.mp3`)
- `--model` (optional, default: `small`): Whisper model size/name
  - common values: `tiny`, `base`, `small`, `medium`, `large-v3`

See full help:

```powershell
.\.venv\Scripts\python.exe .\transcribe.py --help
```

## Best Practices

- Start with `--model small` for speed, then move to `medium`/`large-v3` only when needed.
- Use clear audio when possible (less overlap/noise gives better results).
- Keep original recordings unchanged; treat transcripts as derived files.
- Manually review names, acronyms, and technical terms before sharing.
- Use `.srt` timestamps to quickly jump to audio sections during review.

## Recommended Use of `.srt`

- Use `.srt` as your primary review/edit artifact because each line has exact start/end time.
- Load it in subtitle/video tools (`VLC`, `Subtitle Edit`, `DaVinci Resolve`, `Premiere`) for fast correction.
- Keep timestamps when turning transcripts into notes, action items, or citations.
- If you publish clips, `.srt` can be used directly as closed captions.

## Expansion Ideas (Simple -> More Advanced)

1. Add `--output-dir` so generated files go to a chosen folder.
2. Add batch mode (`--input-dir`) to transcribe many files at once.
3. Add flags for language, VAD toggle, beam size, and chunk settings.
4. Export additional formats (`.vtt`, JSON with word-level timing).
5. Add optional speaker diarization pipeline.
6. Add post-processing: punctuation cleanup, summary, action-item extraction.
7. Add a tiny local UI (drag/drop audio + progress + downloads).

