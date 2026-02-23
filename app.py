from __future__ import annotations

import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import av
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"

for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    job_id: str
    original_filename: str
    model: str
    export_folder: str
    audio_path: str
    status: str
    progress: float
    message: str
    created_at: float
    updated_at: float
    txt_path: str | None = None
    srt_path: str | None = None
    segments_path: str | None = None
    device: str | None = None
    compute_type: str | None = None
    language: str | None = None
    language_probability: float | None = None
    segment_count: int = 0
    transcription_seconds: float | None = None
    error: str | None = None


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)
model_cache: dict[tuple[str, str, str], WhisperModel] = {}
model_cache_lock = threading.Lock()

app = FastAPI(title="Speech-to-Text Meeting Helper")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def safe_token(value: str, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        return default
    return cleaned[:80]


def format_hhmmss(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1_000
    ms = total_ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def load_model(model_name: str) -> tuple[WhisperModel, tuple[str, str]]:
    attempts = [("cuda", "float16"), ("cpu", "int8")]
    last_error: Exception | None = None

    for device, compute_type in attempts:
        cache_key = (model_name, device, compute_type)
        with model_cache_lock:
            cached_model = model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model, (device, compute_type)

        try:
            loaded_model = WhisperModel(model_name, device=device, compute_type=compute_type)
            with model_cache_lock:
                model_cache[cache_key] = loaded_model
            return loaded_model, (device, compute_type)
        except Exception as exc:  # pragma: no cover - environment specific
            last_error = exc

    raise RuntimeError(f"Failed to initialize model '{model_name}'. Last error: {last_error}")


def get_audio_duration_seconds(audio_path: Path) -> float:
    container = av.open(str(audio_path))
    try:
        if container.streams.audio:
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration) / 1_000_000.0
    finally:
        container.close()
    return 0.0


def update_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def build_job_response(job: Job) -> dict[str, Any]:
    payload = asdict(job)
    if job.status == "completed":
        payload["downloads"] = {
            "txt": f"/api/jobs/{job.job_id}/download/txt",
            "srt": f"/api/jobs/{job.job_id}/download/srt",
        }
        payload["audio_url"] = f"/api/jobs/{job.job_id}/audio"
        payload["segments_url"] = f"/api/jobs/{job.job_id}/segments"
    return payload


def transcribe_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs[job_id]

    audio_path = Path(job.audio_path)
    export_folder = safe_token(job.export_folder, "default")
    output_folder = OUTPUT_DIR / export_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    file_stem = safe_token(Path(job.original_filename).stem, "audio")
    base_name = f"{file_stem}.{job_id[:8]}"
    txt_path = output_folder / f"{base_name}.transcript.txt"
    srt_path = output_folder / f"{base_name}.transcript.srt"
    segments_path = output_folder / f"{base_name}.segments.json"
    transcription_start = time.time()

    try:
        update_job(job_id, status="running", progress=0.02, message="Loading model")

        model, selected = load_model(job.model)
        update_job(
            job_id,
            device=selected[0],
            compute_type=selected[1],
            message="Analyzing audio",
            progress=0.08,
        )

        total_duration = get_audio_duration_seconds(audio_path)

        segments_iter, info = model.transcribe(
            str(audio_path),
            beam_size=1,
            best_of=1,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        update_job(job_id, message="Transcribing", progress=0.12)
        segments_list: list[dict[str, Any]] = []
        segment_index = 0

        with txt_path.open("w", encoding="utf-8") as txt_file, srt_path.open(
            "w", encoding="utf-8"
        ) as srt_file:
            txt_file.write("# Transcription\n")
            txt_file.write(f"# File: {job.original_filename}\n")
            txt_file.write(f"# Model: {job.model}\n")
            txt_file.write(f"# Device: {selected[0]}\n")
            txt_file.write(f"# Compute type: {selected[1]}\n")
            txt_file.write(
                f"# Language: {info.language} (probability={info.language_probability:.3f})\n\n"
            )

            for segment in segments_iter:
                text = segment.text.strip()
                if not text:
                    continue
                segment_index += 1
                start = float(segment.start)
                end = float(segment.end)

                txt_file.write(f"[{format_hhmmss(start)}] {text}\n")
                srt_file.write(f"{segment_index}\n")
                srt_file.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
                srt_file.write(text + "\n\n")

                segments_list.append(
                    {
                        "index": segment_index,
                        "start": start,
                        "end": end,
                        "text": text,
                    }
                )

                if total_duration > 0:
                    fraction = min(end / total_duration, 1.0)
                    progress = min(0.12 + (fraction * 0.86), 0.99)
                else:
                    progress = min(0.12 + (segment_index * 0.0025), 0.99)

                update_job(
                    job_id,
                    progress=progress,
                    message=f"Transcribing segment {segment_index}",
                    segment_count=segment_index,
                )

        with segments_path.open("w", encoding="utf-8") as segment_file:
            json.dump(segments_list, segment_file, ensure_ascii=False, indent=2)

        update_job(
            job_id,
            status="completed",
            message="Completed",
            progress=1.0,
            txt_path=str(txt_path),
            srt_path=str(srt_path),
            segments_path=str(segments_path),
            language=info.language,
            language_probability=float(info.language_probability),
            segment_count=segment_index,
            transcription_seconds=time.time() - transcription_start,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            message="Failed",
            progress=1.0,
            transcription_seconds=time.time() - transcription_start,
            error=str(exc),
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form("small"),
    export_folder: str = Form("default"),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    audio_name = safe_token(Path(file.filename).name, "audio.mp3")
    job_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{job_id}-{audio_name}"

    with upload_path.open("wb") as out_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
    await file.close()

    now = time.time()
    job = Job(
        job_id=job_id,
        original_filename=file.filename,
        model=model,
        export_folder=export_folder or "default",
        audio_path=str(upload_path),
        status="queued",
        progress=0.0,
        message="Queued",
        created_at=now,
        updated_at=now,
    )

    with jobs_lock:
        jobs[job_id] = job

    executor.submit(transcribe_job, job_id)
    return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return build_job_response(job)


@app.get("/api/jobs/{job_id}/download/{kind}")
def download_artifact(job_id: str, kind: str) -> FileResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            raise HTTPException(status_code=409, detail="Job not completed")
        if kind == "txt":
            path = Path(job.txt_path) if job.txt_path else None
            media_type = "text/plain"
        elif kind == "srt":
            path = Path(job.srt_path) if job.srt_path else None
            media_type = "application/x-subrip"
        else:
            raise HTTPException(status_code=404, detail="Unknown artifact type")

    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact missing")

    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/jobs/{job_id}/audio")
def get_audio(job_id: str) -> FileResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        audio_path = Path(job.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio missing")
    return FileResponse(audio_path, filename=audio_path.name)


@app.get("/api/jobs/{job_id}/segments")
def get_segments(job_id: str) -> Any:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            raise HTTPException(status_code=409, detail="Job not completed")
        segments_path = Path(job.segments_path) if job.segments_path else None

    if not segments_path or not segments_path.exists():
        raise HTTPException(status_code=404, detail="Segments missing")
    return json.loads(segments_path.read_text(encoding="utf-8"))


@app.get("/api/models")
def list_models() -> dict[str, list[str]]:
    return {"models": ["tiny", "base", "small", "medium", "large-v3"]}
