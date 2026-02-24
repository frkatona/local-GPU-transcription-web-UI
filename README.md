# Speech-to-Text Meeting Helper

Local speech-to-text run on the GPU with simple browser UI and player/transcript reader.  Able to transcribe pre-recorded audio files or live audio streamed on the computer.

### Transcription of pre-existing local audio file

![local transcription](readme_images/file-transcription.gif)

### Live transcription of background podcast audio

![live transcription](readme_images/live-transcription.gif)

The transcript can be downloaded as a .txt or .srt for synchronized captioning.  Uses GPU-first fallback behavior with the smallest model taking ~1 min to transcribe a ~1 hour mp3 on my RTX 3080

Built with `faster-whisper` and `FastAPI` for the backend and `React` for the frontend.

---

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

Both the tiny and large-v3 models seem to have done a perfect job with a recording of strongly-accented English speaker who was using several theoretical chemistry terms.  I'm not sure what the returns are supposed to be...maybe the heavier model is more tolerant of background noise or overlapping speakers?

---

## Setup

### Install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Start server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### Open web UI:

```text
http://127.0.0.1:8000
```

## Expansion Ideas

 -  diarization
 -  microphone + system audio
 -  post-processing: summaries, action items, etc.