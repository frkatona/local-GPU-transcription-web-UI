from __future__ import annotations

import asyncio
import ctypes
import json
import os
import queue
import re
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import av
import numpy as np
import sounddevice as sd
import soundcard as sc
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel

# Compat patch for libraries still using binary np.fromstring() on NumPy 2+.
_ORIGINAL_NP_FROMSTRING = np.fromstring


def _binary_fromstring_compat(
    string: Any, dtype: Any = float, count: int = -1, sep: str = "", *, like: Any = None
) -> np.ndarray:
    if sep == "":
        try:
            buffer_view = memoryview(string)
            # Keep binary fromstring semantics safe for mutable buffers by copying.
            return np.frombuffer(buffer_view, dtype=dtype, count=count).copy()
        except (TypeError, ValueError):
            pass
    if like is None:
        return _ORIGINAL_NP_FROMSTRING(string, dtype=dtype, count=count, sep=sep)
    return _ORIGINAL_NP_FROMSTRING(string, dtype=dtype, count=count, sep=sep, like=like)


np.fromstring = _binary_fromstring_compat  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
LIVE_CAPTURE_DIR = DATA_DIR / "live_captures"

for directory in (UPLOAD_DIR, OUTPUT_DIR, LIVE_CAPTURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LIVE_TARGET_SAMPLE_RATE = 16_000
LIVE_LOW_LATENCY_CHUNK_SECONDS = 4.0
LIVE_LOW_LATENCY_FLUSH_SECONDS = 0.35
LIVE_BUFFERED_CHUNK_SECONDS = 60.0
LIVE_BUFFERED_OVERLAP_SECONDS = 6.0
LIVE_BUFFERED_FLUSH_SECONDS = 8.0
LIVE_LEVEL_EVENT_INTERVAL_SECONDS = 0.12
LIVE_BUFFER_METRICS_INTERVAL_SECONDS = 0.3
LIVE_LEVEL_FLOOR_DBFS = -90.0
LIVE_USE_VAD_FILTER = False

LIVE_DEFAULT_MODEL = "small"
LIVE_DEFAULT_LANGUAGE = "en"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DIARIZATION_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN")


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
    diarize: bool = False
    diarization_speakers: int | None = None
    diarization_status: str = "disabled"
    diarization_error: str | None = None
    diarization_seconds: float | None = None
    speaker_count: int = 0
    transcription_seconds: float | None = None
    error: str | None = None


class LiveStartRequest(BaseModel):
    source: str = "system"
    mode: str = "new"
    model: str = LIVE_DEFAULT_MODEL
    language: str | None = LIVE_DEFAULT_LANGUAGE
    device_id: str | None = None
    capture_mode: str = "low_latency"
    diarize: bool = False
    diarization_speakers: int | None = None


class ModelPreloadRequest(BaseModel):
    model: str = "small"


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)
model_cache: dict[tuple[str, str, str], WhisperModel] = {}
model_cache_lock = threading.Lock()
diarization_pipeline_cache: dict[tuple[str, str], Any] = {}
diarization_pipeline_lock = threading.Lock()

live_state_lock = threading.Lock()
live_segments: list[dict[str, Any]] = []
live_state: dict[str, Any] = {
    "status": "idle",
    "source": "system",
    "mode": "new",
    "model": LIVE_DEFAULT_MODEL,
    "language": LIVE_DEFAULT_LANGUAGE,
    "capture_mode": "low_latency",
    "diarize": False,
    "diarization_speakers": None,
    "diarization_status": "disabled",
    "diarization_error": None,
    "diarization_seconds": None,
    "speaker_count": 0,
    "device_id": None,
    "device_label": None,
    "message": "Idle",
    "error": None,
    "session_id": None,
    "segment_count": 0,
    "started_at": None,
    "updated_at": time.time(),
    "input_level_rms": 0.0,
    "input_level_peak": 0.0,
    "input_level_dbfs": LIVE_LEVEL_FLOOR_DBFS,
    "input_level_updated_at": None,
    "next_buffer_update_seconds": None,
    "buffer_interval_seconds": None,
    "buffer_queue_depth": 0,
}
live_thread: threading.Thread | None = None
live_stop_event: threading.Event | None = None
live_audio_meta: dict[str, Any] = {
    "path": None,
    "sample_rate": None,
    "duration_seconds": 0.0,
    "session_id": None,
}
live_asr_audio_meta: dict[str, Any] = {
    "path": None,
    "sample_rate": None,
    "duration_seconds": 0.0,
    "session_id": None,
}
live_diagnostics_meta: dict[str, Any] = {
    "path": None,
    "session_id": None,
}

live_clients: set[WebSocket] = set()
live_main_loop: asyncio.AbstractEventLoop | None = None

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


def is_model_cached(model_name: str) -> bool:
    with model_cache_lock:
        return any(key[0] == model_name for key in model_cache)


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


def parse_form_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value '{value}'.")


def get_diarization_token() -> str | None:
    for env_key in DIARIZATION_TOKEN_ENV_VARS:
        token = os.environ.get(env_key)
        if token and token.strip():
            return token.strip()
    return None


def load_diarization_pipeline() -> tuple[Any, str]:
    token = get_diarization_token()
    if not token:
        token_keys = ", ".join(DIARIZATION_TOKEN_ENV_VARS)
        raise RuntimeError(f"Diarization requires a Hugging Face token in one of: {token_keys}.")

    cache_key = (DIARIZATION_MODEL, token)
    with diarization_pipeline_lock:
        cached = diarization_pipeline_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Install it to enable diarization."
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    except Exception as exc:
        raise RuntimeError(f"Failed to load diarization model '{DIARIZATION_MODEL}': {exc}") from exc

    device_label = "cpu"
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        device_label = str(device)
    except Exception:
        device_label = "cpu"

    loaded = (pipeline, device_label)
    with diarization_pipeline_lock:
        diarization_pipeline_cache[cache_key] = loaded
    return loaded


def _annotation_to_diarization_segments(annotation: Any) -> list[dict[str, Any]]:
    diarization_segments: list[dict[str, Any]] = []
    for turn, _, speaker_id in annotation.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        if end <= start:
            continue
        diarization_segments.append(
            {
                "start": start,
                "end": end,
                "speaker_id": str(speaker_id),
            }
        )
    diarization_segments.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return diarization_segments


def run_speaker_diarization(audio_path: Path, diarization_speakers: int | None) -> list[dict[str, Any]]:
    pipeline, _ = load_diarization_pipeline()
    kwargs: dict[str, Any] = {}
    if diarization_speakers is not None:
        kwargs["num_speakers"] = int(diarization_speakers)

    try:
        annotation = pipeline(str(audio_path), **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Diarization failed: {exc}") from exc
    return _annotation_to_diarization_segments(annotation)


def run_speaker_diarization_on_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    diarization_speakers: int | None,
    pipeline: Any | None = None,
) -> list[dict[str, Any]]:
    if waveform.size == 0:
        return []

    diarization_pipeline = pipeline
    if diarization_pipeline is None:
        diarization_pipeline, _ = load_diarization_pipeline()

    kwargs: dict[str, Any] = {}
    if diarization_speakers is not None:
        kwargs["num_speakers"] = int(diarization_speakers)

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for in-memory diarization.") from exc

    mono = waveform.astype(np.float32, copy=False).reshape(1, -1)
    payload = {"waveform": torch.from_numpy(mono.copy()), "sample_rate": int(sample_rate)}

    try:
        annotation = diarization_pipeline(payload, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Diarization failed: {exc}") from exc
    return _annotation_to_diarization_segments(annotation)


def interval_overlap_seconds(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def format_segment_text(text: str, speaker: str | None = None) -> str:
    cleaned = text.strip()
    speaker_label = (speaker or "").strip()
    if speaker_label:
        return f"{speaker_label}: {cleaned}"
    return cleaned


def assign_speakers_to_segments(
    segments: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
    speaker_label_map: dict[str, str] | None = None,
) -> int:
    if not segments or not diarization_segments:
        return len(speaker_label_map or {})

    if speaker_label_map is None:
        speaker_label_map = {}

    for segment in segments:
        start = float(segment["start"])
        end = max(float(segment["end"]), start)
        midpoint = (start + end) / 2.0
        best_speaker_id: str | None = None
        best_overlap = 0.0
        nearest_distance = float("inf")

        for diar_item in diarization_segments:
            diar_start = float(diar_item["start"])
            diar_end = float(diar_item["end"])
            overlap = interval_overlap_seconds(start, end, diar_start, diar_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker_id = str(diar_item["speaker_id"])
            if best_overlap <= 0.0:
                diar_midpoint = (diar_start + diar_end) / 2.0
                distance = abs(midpoint - diar_midpoint)
                if distance < nearest_distance:
                    nearest_distance = distance
                    best_speaker_id = str(diar_item["speaker_id"])

        if not best_speaker_id:
            continue

        if best_speaker_id not in speaker_label_map:
            speaker_label_map[best_speaker_id] = f"Speaker {len(speaker_label_map) + 1}"

        segment["speaker_id"] = best_speaker_id
        segment["speaker"] = speaker_label_map[best_speaker_id]

    return len(speaker_label_map)


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


def has_active_batch_jobs() -> bool:
    with jobs_lock:
        return any(job.status in {"queued", "running"} for job in jobs.values())


def is_live_active() -> bool:
    with live_state_lock:
        return live_state["status"] in {"starting", "running", "stopping"}


def get_live_segments_copy() -> list[dict[str, Any]]:
    with live_state_lock:
        return [dict(segment) for segment in live_segments]


def clear_live_audio_artifact(delete_file: bool = False) -> None:
    with live_state_lock:
        current_path = live_audio_meta.get("path")
        live_audio_meta.update(
            {
                "path": None,
                "sample_rate": None,
                "duration_seconds": 0.0,
                "session_id": None,
            }
        )

    if delete_file and current_path:
        try:
            path = Path(str(current_path))
            if path.exists():
                path.unlink()
        except Exception:
            pass


def set_live_audio_artifact(path: Path, sample_rate: int, duration_seconds: float, session_id: str) -> None:
    with live_state_lock:
        live_audio_meta.update(
            {
                "path": str(path),
                "sample_rate": int(sample_rate),
                "duration_seconds": float(max(0.0, duration_seconds)),
                "session_id": session_id,
            }
        )


def get_live_audio_artifact_copy() -> dict[str, Any]:
    with live_state_lock:
        return dict(live_audio_meta)


def clear_live_asr_audio_artifact(delete_file: bool = False) -> None:
    with live_state_lock:
        current_path = live_asr_audio_meta.get("path")
        live_asr_audio_meta.update(
            {
                "path": None,
                "sample_rate": None,
                "duration_seconds": 0.0,
                "session_id": None,
            }
        )

    if delete_file and current_path:
        try:
            path = Path(str(current_path))
            if path.exists():
                path.unlink()
        except Exception:
            pass


def set_live_asr_audio_artifact(path: Path, sample_rate: int, duration_seconds: float, session_id: str) -> None:
    with live_state_lock:
        live_asr_audio_meta.update(
            {
                "path": str(path),
                "sample_rate": int(sample_rate),
                "duration_seconds": float(max(0.0, duration_seconds)),
                "session_id": session_id,
            }
        )


def get_live_asr_audio_artifact_copy() -> dict[str, Any]:
    with live_state_lock:
        return dict(live_asr_audio_meta)


def clear_live_diagnostics_artifact(delete_file: bool = False) -> None:
    with live_state_lock:
        current_path = live_diagnostics_meta.get("path")
        live_diagnostics_meta.update(
            {
                "path": None,
                "session_id": None,
            }
        )

    if delete_file and current_path:
        try:
            path = Path(str(current_path))
            if path.exists():
                path.unlink()
        except Exception:
            pass


def set_live_diagnostics_artifact(path: Path, session_id: str) -> None:
    with live_state_lock:
        live_diagnostics_meta.update(
            {
                "path": str(path),
                "session_id": session_id,
            }
        )


def get_live_diagnostics_artifact_copy() -> dict[str, Any]:
    with live_state_lock:
        return dict(live_diagnostics_meta)


def build_live_metrics_response_locked() -> dict[str, Any]:
    return {
        "input_level_rms": float(live_state.get("input_level_rms", 0.0) or 0.0),
        "input_level_peak": float(live_state.get("input_level_peak", 0.0) or 0.0),
        "input_level_dbfs": float(live_state.get("input_level_dbfs", LIVE_LEVEL_FLOOR_DBFS)),
        "input_level_updated_at": live_state.get("input_level_updated_at"),
        "next_buffer_update_seconds": live_state.get("next_buffer_update_seconds"),
        "buffer_interval_seconds": live_state.get("buffer_interval_seconds"),
        "buffer_queue_depth": int(live_state.get("buffer_queue_depth", 0) or 0),
    }


def build_live_state_response_locked() -> dict[str, Any]:
    state_copy = dict(live_state)
    segment_count = len(live_segments)
    speaker_count = len(
        {str(item.get("speaker")) for item in live_segments if str(item.get("speaker") or "").strip()}
    )
    state_copy["segment_count"] = segment_count
    state_copy["speaker_count"] = speaker_count
    downloads: dict[str, str] = {}
    if segment_count > 0:
        downloads["txt"] = "/api/live/download/txt"
        downloads["srt"] = "/api/live/download/srt"

    audio_path = live_audio_meta.get("path")
    if audio_path:
        path = Path(str(audio_path))
        if path.exists():
            downloads["audio"] = "/api/live/download/audio"
            state_copy["audio_duration_seconds"] = float(live_audio_meta.get("duration_seconds") or 0.0)
            state_copy["audio_sample_rate"] = live_audio_meta.get("sample_rate")
        else:
            live_audio_meta.update(
                {"path": None, "sample_rate": None, "duration_seconds": 0.0, "session_id": None}
            )

    asr_audio_path = live_asr_audio_meta.get("path")
    if asr_audio_path:
        path = Path(str(asr_audio_path))
        if path.exists():
            downloads["audio_16k"] = "/api/live/download/audio_16k"
            state_copy["audio_16k_duration_seconds"] = float(
                live_asr_audio_meta.get("duration_seconds") or 0.0
            )
            state_copy["audio_16k_sample_rate"] = live_asr_audio_meta.get("sample_rate")
        else:
            live_asr_audio_meta.update(
                {"path": None, "sample_rate": None, "duration_seconds": 0.0, "session_id": None}
            )

    diagnostics_path = live_diagnostics_meta.get("path")
    if diagnostics_path:
        path = Path(str(diagnostics_path))
        if path.exists():
            downloads["diagnostics"] = "/api/live/download/diagnostics"
        else:
            live_diagnostics_meta.update({"path": None, "session_id": None})

    if downloads:
        state_copy["downloads"] = downloads
    return state_copy


def build_live_state_response() -> dict[str, Any]:
    with live_state_lock:
        return build_live_state_response_locked()


def update_live_state(**updates: Any) -> dict[str, Any]:
    with live_state_lock:
        live_state.update(updates)
        live_state["updated_at"] = time.time()
        live_state["segment_count"] = len(live_segments)
        live_state["speaker_count"] = len(
            {str(item.get("speaker")) for item in live_segments if str(item.get("speaker") or "").strip()}
        )
        return dict(live_state)


def update_live_metrics(**updates: Any) -> dict[str, Any]:
    with live_state_lock:
        live_state.update(updates)
        if {"input_level_rms", "input_level_peak", "input_level_dbfs"} & set(updates.keys()):
            live_state["input_level_updated_at"] = time.time()
        return build_live_metrics_response_locked()


def append_live_segment(
    start: float,
    end: float,
    text: str,
    speaker: str | None = None,
    speaker_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with live_state_lock:
        segment = {
            "index": len(live_segments) + 1,
            "start": float(start),
            "end": float(max(end, start)),
            "text": text,
        }
        speaker_label = (speaker or "").strip()
        speaker_identifier = (speaker_id or "").strip()
        if speaker_label:
            segment["speaker"] = speaker_label
        if speaker_identifier:
            segment["speaker_id"] = speaker_identifier
        live_segments.append(segment)
        live_state["segment_count"] = len(live_segments)
        live_state["speaker_count"] = len(
            {str(item.get("speaker")) for item in live_segments if str(item.get("speaker") or "").strip()}
        )
        live_state["updated_at"] = time.time()
        return dict(segment), dict(live_state)


def build_txt_content(segments: list[dict[str, Any]], title: str) -> str:
    lines = ["# Live Transcription", f"# Session: {title}", ""]
    for segment in segments:
        lines.append(
            f"[{format_hhmmss(float(segment['start']))}] "
            f"{format_segment_text(str(segment['text']), str(segment.get('speaker') or ''))}"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_srt_content(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(str(index))
        blocks.append(
            f"{format_srt_time(float(segment['start']))} --> {format_srt_time(float(segment['end']))}"
        )
        blocks.append(format_segment_text(str(segment["text"]), str(segment.get("speaker") or "")))
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if audio.size == 0:
        return np.empty((0,), dtype=np.float32)
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    source_audio = audio.astype(np.float32, copy=False)
    try:
        frame = av.AudioFrame.from_ndarray(source_audio.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = int(source_rate)
        resampler = av.audio.resampler.AudioResampler(
            format="flt",
            layout="mono",
            rate=int(target_rate),
        )
        out_frames = resampler.resample(frame)
        out_frames.extend(resampler.resample(None))
        if out_frames:
            return np.concatenate([item.to_ndarray().reshape(-1) for item in out_frames]).astype(
                np.float32
            )
    except Exception:
        # Fallback path if PyAV resampling fails in the runtime environment.
        pass

    target_len = max(1, int(round((source_audio.shape[0] / source_rate) * target_rate)))
    source_positions = np.arange(source_audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(0, source_audio.shape[0] - 1, target_len, dtype=np.float64)
    return np.interp(target_positions, source_positions, source_audio).astype(np.float32)


def normalize_capture_audio(frame: np.ndarray) -> np.ndarray:
    samples = np.asarray(frame)
    if samples.size == 0:
        return np.empty((0,), dtype=np.float32)

    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(int(info.min)), int(info.max)))
        if scale <= 0.0:
            return np.zeros_like(samples, dtype=np.float32)
        normalized = samples.astype(np.float32) / scale
    else:
        normalized = samples.astype(np.float32, copy=False)

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 1.5:
        if peak <= 40_000.0:
            normalized = normalized / 32_768.0
        elif peak <= 2_400_000_000.0:
            normalized = normalized / 2_147_483_648.0
        else:
            normalized = normalized / peak

    return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)


def capture_to_mono(frame: np.ndarray) -> np.ndarray:
    normalized = normalize_capture_audio(frame)
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim != 2:
        return normalized.reshape(-1)

    channels = normalized
    if channels.shape[1] <= 1:
        return channels[:, 0]

    channel_energy = np.mean(np.square(channels, dtype=np.float32), axis=0)
    channel_index = int(np.argmax(channel_energy))
    return channels[:, channel_index]


def list_live_mic_devices() -> list[dict[str, Any]]:
    devices = sd.query_devices()
    host_apis = sd.query_hostapis()
    default_input_raw = sd.default.device[0] if sd.default.device else -1
    default_input = int(default_input_raw) if default_input_raw is not None else -1

    entries: list[dict[str, Any]] = []
    for idx, info in enumerate(devices):
        max_inputs = int(info["max_input_channels"])
        if max_inputs <= 0:
            continue
        hostapi_index = int(info["hostapi"])
        hostapi_name = host_apis[hostapi_index]["name"] if 0 <= hostapi_index < len(host_apis) else "Unknown"
        entries.append(
            {
                "id": str(idx),
                "label": f"{info['name']} ({hostapi_name})",
                "default": idx == default_input,
                "channels": max_inputs,
                "sample_rate": int(info["default_samplerate"] or 44_100),
            }
        )
    return entries


def list_live_system_devices() -> list[dict[str, Any]]:
    com_initialized = initialize_com_for_thread()
    try:
        default_speaker = sc.default_speaker()
        default_id = str(default_speaker.id) if default_speaker is not None else None
        speakers = sc.all_speakers()

        entries: list[dict[str, Any]] = []
        for speaker in speakers:
            entries.append(
                {
                    "id": str(speaker.id),
                    "label": speaker.name,
                    "default": str(speaker.id) == default_id,
                    "channels": int(speaker.channels),
                    "sample_rate": 48_000,
                }
            )
        return entries
    finally:
        if com_initialized:
            uninitialize_com_for_thread()


def resolve_selected_live_device(source: str, requested_device_id: str | None) -> tuple[str, str]:
    if source == "mic":
        devices = list_live_mic_devices()
        source_label = "microphone"
    else:
        devices = list_live_system_devices()
        source_label = "system output"

    if not devices:
        raise RuntimeError(f"No {source_label} devices were found.")

    if requested_device_id:
        for device in devices:
            if device["id"] == requested_device_id:
                return str(device["id"]), str(device["label"])
        raise RuntimeError(f"The selected {source_label} device is unavailable.")

    for device in devices:
        if device.get("default"):
            return str(device["id"]), str(device["label"])
    first = devices[0]
    return str(first["id"]), str(first["label"])


def resolve_live_mic_device(device_id: str) -> tuple[int, int, int, str]:
    try:
        device_index = int(device_id)
    except ValueError as exc:
        raise RuntimeError("Invalid microphone device id.") from exc

    try:
        info = sd.query_devices(device_index)
    except Exception as exc:  # pragma: no cover - backend/device dependent
        raise RuntimeError("Unable to query selected microphone device.") from exc

    max_inputs = int(info["max_input_channels"])
    if max_inputs <= 0:
        raise RuntimeError("Selected device is not a microphone input.")

    host_apis = sd.query_hostapis()
    hostapi_index = int(info["hostapi"])
    hostapi_name = host_apis[hostapi_index]["name"] if 0 <= hostapi_index < len(host_apis) else "Unknown"
    label = f"{info['name']} ({hostapi_name})"
    channels = max(1, min(2, max_inputs))
    sample_rate = int(info["default_samplerate"] or 44_100)
    return device_index, channels, sample_rate, label


def resolve_system_loopback_microphone(device_id: str) -> tuple[Any, int, int, str]:
    try:
        speaker = sc.get_speaker(device_id)
    except Exception as exc:  # pragma: no cover - backend/device dependent
        raise RuntimeError("Unable to query selected system output device.") from exc

    if speaker is None:
        raise RuntimeError("Selected system output device is unavailable.")

    loopback_mic = sc.get_microphone(id=str(speaker.id), include_loopback=True)
    if loopback_mic is None:
        raise RuntimeError("Unable to initialize system loopback capture device.")

    channels = max(1, min(2, int(loopback_mic.channels)))
    sample_rate = 48_000
    return loopback_mic, channels, sample_rate, speaker.name


def initialize_com_for_thread() -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    result = ctypes.windll.ole32.CoInitializeEx(None, 0x0)
    return result in (0, 1)


def uninitialize_com_for_thread() -> None:
    if hasattr(ctypes, "windll"):
        ctypes.windll.ole32.CoUninitialize()


async def broadcast_live_event(event: dict[str, Any]) -> None:
    stale: list[WebSocket] = []
    for client in list(live_clients):
        try:
            await client.send_json(event)
        except Exception:
            stale.append(client)
    for client in stale:
        live_clients.discard(client)


def push_live_event(event: dict[str, Any]) -> None:
    if live_main_loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(broadcast_live_event(event), live_main_loop)
    except RuntimeError:
        return


def transcribe_live_chunk(
    model: WhisperModel,
    raw_chunk: np.ndarray,
    source_rate: int,
    chunk_start: float,
    base_offset: float,
    beam_size: int = 1,
    condition_on_previous_text: bool = False,
    vad_filter: bool = True,
    language: str | None = None,
    on_resampled_chunk: Callable[[np.ndarray], None] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    chunk = resample_audio(raw_chunk, source_rate, LIVE_TARGET_SAMPLE_RATE)
    if chunk.size == 0:
        return [], chunk
    if on_resampled_chunk is not None:
        on_resampled_chunk(chunk)

    segments_iter, _ = model.transcribe(
        chunk,
        beam_size=beam_size,
        best_of=1,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
        language=language,
    )

    segments: list[dict[str, Any]] = []
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        local_start = float(segment.start)
        local_end = float(segment.end)
        start = base_offset + chunk_start + local_start
        end = base_offset + chunk_start + local_end
        segments.append(
            {
                "start": start,
                "end": max(end, start),
                "text": text,
                "local_start": local_start,
                "local_end": max(local_end, local_start),
            }
        )
    return segments, chunk


def run_live_transcription(
    session_id: str,
    source: str,
    mode: str,
    model_name: str,
    language: str | None,
    capture_mode: str,
    diarize: bool,
    diarization_speakers: int | None,
    selected_device_id: str,
    selected_device_label: str,
    base_offset: float,
    stop_event: threading.Event,
) -> None:
    global live_thread, live_stop_event

    session_started_at = time.time()
    source_rate_for_session: int | None = None
    session_outcome = "stopped"
    session_error: str | None = None

    stats_lock = threading.Lock()
    stats: dict[str, Any] = {
        "frame_chunks_received": 0,
        "frame_samples_received": 0,
        "frame_queue_max_depth": 0,
        "transcription_queue_max_depth": 0,
        "transcribe_chunks_total": 0,
        "transcribe_chunks_with_text": 0,
        "emitted_segments_total": 0,
        "dropped_frame_chunks": 0,
        "dropped_transcription_chunks": 0,
        "dropped_capture_artifact_chunks": 0,
        "dropped_asr_artifact_chunks": 0,
        "asr_audio_chunks_enqueued": 0,
        "level_event_count": 0,
        "level_rms_sum": 0.0,
        "level_rms_max": 0.0,
        "level_peak_max": 0.0,
        "level_dbfs_min": None,
        "level_dbfs_max": LIVE_LEVEL_FLOOR_DBFS,
        "level_dbfs_last": LIVE_LEVEL_FLOOR_DBFS,
    }
    live_speaker_label_map: dict[str, str] = {}
    live_diarization_pipeline: Any | None = None
    live_diarization_device = "cpu"
    live_diarization_active = bool(diarize and capture_mode == "buffered_hq")
    live_diarization_status = "running" if live_diarization_active else "disabled"
    live_diarization_error: str | None = None
    live_diarization_seconds = 0.0

    frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1024)
    last_level_event = 0.0
    audio_queue: queue.Queue[np.ndarray | None] | None = None
    audio_writer_thread: threading.Thread | None = None
    audio_writer_errors: list[Exception] = []
    audio_samples_written = 0
    audio_file_path: Path | None = None
    audio_source_rate = 0

    asr_audio_queue: queue.Queue[np.ndarray | None] | None = None
    asr_audio_writer_thread: threading.Thread | None = None
    asr_audio_writer_errors: list[Exception] = []
    asr_audio_samples_written = 0
    asr_audio_file_path: Path | None = None

    capture_writer_finished = False
    asr_writer_finished = False
    capture_writer_error_message: str | None = None
    asr_writer_error_message: str | None = None

    def emit_live_metrics(**updates: Any) -> None:
        metrics_payload = update_live_metrics(**updates)
        push_live_event({"type": "live_metrics", "metrics": metrics_payload})

    def start_audio_capture_writer(sample_rate: int) -> None:
        nonlocal audio_queue, audio_writer_thread, audio_file_path, audio_source_rate, audio_samples_written
        audio_source_rate = int(sample_rate)
        audio_samples_written = 0
        audio_file_path = LIVE_CAPTURE_DIR / f"live-{session_id}.wav"
        audio_queue = queue.Queue(maxsize=2048)

        def writer_worker() -> None:
            nonlocal audio_samples_written
            assert audio_queue is not None
            assert audio_file_path is not None
            try:
                with wave.open(str(audio_file_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(audio_source_rate)
                    while True:
                        try:
                            packet = audio_queue.get(timeout=0.25)
                        except queue.Empty:
                            if stop_event.is_set():
                                continue
                            continue
                        if packet is None:
                            break

                        pcm = np.clip(packet, -1.0, 1.0)
                        pcm_i16 = (pcm * 32767.0).astype(np.int16)
                        wav_file.writeframes(pcm_i16.tobytes())
                        audio_samples_written += int(packet.shape[0])
            except Exception as exc:  # pragma: no cover - filesystem/environment specific
                audio_writer_errors.append(exc)
                stop_event.set()

        audio_writer_thread = threading.Thread(
            target=writer_worker,
            daemon=True,
            name="live-audio-writer-thread",
        )
        audio_writer_thread.start()

    def finish_audio_capture_writer() -> None:
        nonlocal audio_queue, audio_writer_thread, audio_file_path
        if audio_queue is not None:
            try:
                audio_queue.put_nowait(None)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(None)
                except queue.Full:
                    pass

        if audio_writer_thread is not None:
            audio_writer_thread.join(timeout=30.0)

        if audio_writer_errors:
            raise RuntimeError(f"Failed to write live audio capture: {audio_writer_errors[0]}")
        if audio_file_path is None:
            return

        duration = (audio_samples_written / audio_source_rate) if audio_source_rate > 0 else 0.0
        if duration <= 0.0:
            try:
                if audio_file_path.exists():
                    audio_file_path.unlink()
            except Exception:
                pass
            return

        set_live_audio_artifact(
            path=audio_file_path,
            sample_rate=audio_source_rate,
            duration_seconds=duration,
            session_id=session_id,
        )

    def start_asr_audio_writer() -> None:
        nonlocal asr_audio_queue, asr_audio_writer_thread, asr_audio_file_path, asr_audio_samples_written
        asr_audio_samples_written = 0
        asr_audio_file_path = LIVE_CAPTURE_DIR / f"live-{session_id}.whisper-16k.wav"
        asr_audio_queue = queue.Queue(maxsize=1024)

        def writer_worker() -> None:
            nonlocal asr_audio_samples_written
            assert asr_audio_queue is not None
            assert asr_audio_file_path is not None
            try:
                with wave.open(str(asr_audio_file_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(LIVE_TARGET_SAMPLE_RATE)
                    while True:
                        try:
                            packet = asr_audio_queue.get(timeout=0.25)
                        except queue.Empty:
                            if stop_event.is_set():
                                continue
                            continue
                        if packet is None:
                            break

                        pcm = np.clip(packet, -1.0, 1.0)
                        pcm_i16 = (pcm * 32767.0).astype(np.int16)
                        wav_file.writeframes(pcm_i16.tobytes())
                        asr_audio_samples_written += int(packet.shape[0])
            except Exception as exc:  # pragma: no cover - filesystem/environment specific
                asr_audio_writer_errors.append(exc)

        asr_audio_writer_thread = threading.Thread(
            target=writer_worker,
            daemon=True,
            name="live-asr-audio-writer-thread",
        )
        asr_audio_writer_thread.start()

    def finish_asr_audio_writer() -> str | None:
        nonlocal asr_audio_queue, asr_audio_writer_thread, asr_audio_file_path
        try:
            if asr_audio_queue is not None:
                try:
                    asr_audio_queue.put_nowait(None)
                except queue.Full:
                    try:
                        asr_audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        asr_audio_queue.put_nowait(None)
                    except queue.Full:
                        pass

            if asr_audio_writer_thread is not None:
                asr_audio_writer_thread.join(timeout=30.0)

            if asr_audio_writer_errors:
                return f"Failed to write 16 kHz Whisper debug audio: {asr_audio_writer_errors[0]}"
            if asr_audio_file_path is None:
                return None

            duration = asr_audio_samples_written / LIVE_TARGET_SAMPLE_RATE
            if duration <= 0.0:
                try:
                    if asr_audio_file_path.exists():
                        asr_audio_file_path.unlink()
                except Exception:
                    pass
                return None

            set_live_asr_audio_artifact(
                path=asr_audio_file_path,
                sample_rate=LIVE_TARGET_SAMPLE_RATE,
                duration_seconds=duration,
                session_id=session_id,
            )
            return None
        except Exception as exc:  # pragma: no cover - filesystem/environment specific
            return f"Failed to finalize 16 kHz Whisper debug audio: {exc}"

    def cleanup_writers(raise_capture_error: bool) -> None:
        nonlocal capture_writer_finished, asr_writer_finished
        nonlocal capture_writer_error_message, asr_writer_error_message

        if not capture_writer_finished:
            try:
                finish_audio_capture_writer()
            except Exception as exc:
                if capture_writer_error_message is None:
                    capture_writer_error_message = str(exc)
                if raise_capture_error:
                    raise
            finally:
                capture_writer_finished = True

        if not asr_writer_finished:
            asr_error = finish_asr_audio_writer()
            if asr_error and asr_writer_error_message is None:
                asr_writer_error_message = asr_error
            asr_writer_finished = True

    def enqueue_asr_audio_chunk(chunk: np.ndarray) -> None:
        if chunk.size == 0 or asr_audio_queue is None:
            return
        packet = chunk.copy()
        with stats_lock:
            stats["asr_audio_chunks_enqueued"] += 1
        try:
            asr_audio_queue.put_nowait(packet)
        except queue.Full:
            try:
                asr_audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                asr_audio_queue.put_nowait(packet)
            except queue.Full:
                return
            with stats_lock:
                stats["dropped_asr_artifact_chunks"] += 1

    try:
        model, _ = load_model(model_name)
        if live_diarization_active:
            try:
                live_diarization_pipeline, live_diarization_device = load_diarization_pipeline()
            except Exception as exc:
                live_diarization_status = "failed"
                live_diarization_error = str(exc)
                live_diarization_active = False
        elif diarize and capture_mode != "buffered_hq":
            live_diarization_status = "unsupported"
            live_diarization_error = "Live diarization is available only in buffered_hq mode."

        mode_label = "buffered HQ" if capture_mode == "buffered_hq" else "low latency"
        initial_buffer_eta = LIVE_BUFFERED_CHUNK_SECONDS if capture_mode == "buffered_hq" else None
        initial_buffer_interval = LIVE_BUFFERED_CHUNK_SECONDS if capture_mode == "buffered_hq" else None
        status_message = f"Listening ({source}) on {selected_device_label} [{mode_label}]"
        if diarize:
            if live_diarization_status == "running":
                status_message = (
                    f"{status_message}; diarization on ({live_diarization_device}, "
                    f"{diarization_speakers or 'auto'} speakers)"
                )
            elif live_diarization_status == "failed":
                status_message = f"{status_message}; diarization unavailable"
            elif live_diarization_status == "unsupported":
                status_message = f"{status_message}; diarization requires buffered_hq"

        state_payload = update_live_state(
            status="running",
            message=status_message,
            source=source,
            mode=mode,
            model=model_name,
            language=language,
            capture_mode=capture_mode,
            diarize=bool(diarize),
            diarization_speakers=diarization_speakers,
            diarization_status=live_diarization_status,
            diarization_error=live_diarization_error,
            diarization_seconds=(0.0 if diarize else None),
            speaker_count=0,
            device_id=selected_device_id,
            device_label=selected_device_label,
            session_id=session_id,
            started_at=time.time(),
            error=None,
            input_level_rms=0.0,
            input_level_peak=0.0,
            input_level_dbfs=LIVE_LEVEL_FLOOR_DBFS,
            input_level_updated_at=time.time(),
            next_buffer_update_seconds=initial_buffer_eta,
            buffer_interval_seconds=initial_buffer_interval,
            buffer_queue_depth=0,
        )
        push_live_event({"type": "live_state", "state": state_payload})
        push_live_event(
            {
                "type": "live_metrics",
                "metrics": {
                    "input_level_rms": 0.0,
                    "input_level_peak": 0.0,
                    "input_level_dbfs": LIVE_LEVEL_FLOOR_DBFS,
                    "input_level_updated_at": time.time(),
                    "next_buffer_update_seconds": initial_buffer_eta,
                    "buffer_interval_seconds": initial_buffer_interval,
                    "buffer_queue_depth": 0,
                },
            }
        )

        start_asr_audio_writer()

        def maybe_emit_level(frame: np.ndarray, force: bool = False) -> None:
            nonlocal last_level_event

            now = time.monotonic()
            if not force and (now - last_level_event) < LIVE_LEVEL_EVENT_INTERVAL_SECONDS:
                return
            if frame.size == 0:
                return

            safe_frame = np.nan_to_num(frame.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
            peak = float(np.max(np.abs(safe_frame)))
            rms = float(np.sqrt(np.mean(np.square(safe_frame, dtype=np.float32))))
            if not np.isfinite(rms):
                rms = 0.0
            if not np.isfinite(peak):
                peak = 0.0
            if rms <= 1e-9:
                dbfs = LIVE_LEVEL_FLOOR_DBFS
            else:
                dbfs = float(max(LIVE_LEVEL_FLOOR_DBFS, min(0.0, 20.0 * np.log10(rms))))

            last_level_event = now
            with stats_lock:
                stats["level_event_count"] += 1
                stats["level_rms_sum"] += rms
                stats["level_rms_max"] = max(float(stats["level_rms_max"]), rms)
                stats["level_peak_max"] = max(float(stats["level_peak_max"]), peak)
                if stats["level_dbfs_min"] is None:
                    stats["level_dbfs_min"] = dbfs
                    stats["level_dbfs_max"] = dbfs
                else:
                    stats["level_dbfs_min"] = min(float(stats["level_dbfs_min"]), dbfs)
                    stats["level_dbfs_max"] = max(float(stats["level_dbfs_max"]), dbfs)
                stats["level_dbfs_last"] = dbfs
            emit_live_metrics(
                input_level_rms=rms,
                input_level_peak=peak,
                input_level_dbfs=dbfs,
            )

        def push_frame(frame: np.ndarray) -> None:
            if frame.size == 0:
                return

            maybe_emit_level(frame)
            with stats_lock:
                stats["frame_chunks_received"] += 1
                stats["frame_samples_received"] += int(frame.shape[0])

            if audio_queue is not None:
                try:
                    audio_queue.put_nowait(frame.copy())
                except queue.Full:
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        audio_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass
                    with stats_lock:
                        stats["dropped_capture_artifact_chunks"] += 1

            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    return

                with stats_lock:
                    stats["dropped_frame_chunks"] += 1
                    drop_count = int(stats["dropped_frame_chunks"])
                if drop_count == 1 or drop_count % 25 == 0:
                    status_payload = update_live_state(
                        message=f"Capture backlog detected ({drop_count} dropped frame chunks)."
                    )
                    push_live_event({"type": "live_state", "state": status_payload})
            frame_depth = frame_queue.qsize()
            with stats_lock:
                stats["frame_queue_max_depth"] = max(int(stats["frame_queue_max_depth"]), frame_depth)

        def emit_segments(
            raw_chunk: np.ndarray,
            source_rate: int,
            chunk_start: float,
            beam_size: int,
            condition_on_previous_text: bool,
            vad_filter: bool,
            emit_from: float,
        ) -> None:
            nonlocal live_diarization_active, live_diarization_status, live_diarization_error
            nonlocal live_diarization_seconds
            with stats_lock:
                stats["transcribe_chunks_total"] += 1
            segments, chunk_16k = transcribe_live_chunk(
                model=model,
                raw_chunk=raw_chunk,
                source_rate=source_rate,
                chunk_start=chunk_start,
                base_offset=base_offset,
                beam_size=beam_size,
                condition_on_previous_text=condition_on_previous_text,
                vad_filter=vad_filter,
                language=language,
                on_resampled_chunk=enqueue_asr_audio_chunk,
            )

            if segments and live_diarization_active and chunk_16k.size > 0:
                diarization_started = time.time()
                relative_segments: list[dict[str, Any]] = []
                for segment in segments:
                    relative_segments.append(
                        {
                            "start": float(segment.get("local_start", 0.0)),
                            "end": float(segment.get("local_end", 0.0)),
                            "text": str(segment.get("text", "")),
                        }
                    )
                try:
                    diarization_segments = run_speaker_diarization_on_waveform(
                        waveform=chunk_16k,
                        sample_rate=LIVE_TARGET_SAMPLE_RATE,
                        diarization_speakers=diarization_speakers,
                        pipeline=live_diarization_pipeline,
                    )
                    assign_speakers_to_segments(
                        relative_segments,
                        diarization_segments,
                        speaker_label_map=live_speaker_label_map,
                    )
                    live_diarization_seconds += max(0.0, time.time() - diarization_started)
                    for idx, rel_segment in enumerate(relative_segments):
                        speaker_label = str(rel_segment.get("speaker") or "").strip()
                        speaker_id = str(rel_segment.get("speaker_id") or "").strip()
                        if speaker_label:
                            segments[idx]["speaker"] = speaker_label
                        if speaker_id:
                            segments[idx]["speaker_id"] = speaker_id
                except Exception as diarization_exc:
                    live_diarization_seconds += max(0.0, time.time() - diarization_started)
                    live_diarization_active = False
                    live_diarization_status = "failed"
                    live_diarization_error = str(diarization_exc)
                    status_payload = update_live_state(
                        diarization_status=live_diarization_status,
                        diarization_error=live_diarization_error,
                        diarization_seconds=live_diarization_seconds,
                        speaker_count=len(live_speaker_label_map),
                        message=(
                            "Live diarization failed; continuing transcription "
                            "without speaker labels."
                        ),
                    )
                    push_live_event({"type": "live_state", "state": status_payload})

            if segments:
                with stats_lock:
                    stats["transcribe_chunks_with_text"] += 1
            for segment in segments:
                if float(segment["end"]) <= (emit_from + 0.02):
                    continue
                if float(segment["start"]) < emit_from:
                    segment["start"] = emit_from
                appended, live_snapshot = append_live_segment(
                    start=segment["start"],
                    end=segment["end"],
                    text=str(segment["text"]),
                    speaker=str(segment.get("speaker") or ""),
                    speaker_id=str(segment.get("speaker_id") or ""),
                )
                if live_diarization_active and live_diarization_status != "running":
                    live_diarization_status = "running"
                if live_diarization_active:
                    live_snapshot["diarization_status"] = live_diarization_status
                    live_snapshot["diarization_error"] = live_diarization_error
                    live_snapshot["diarization_seconds"] = live_diarization_seconds
                    live_snapshot["diarize"] = bool(diarize)
                    live_snapshot["diarization_speakers"] = diarization_speakers
                with stats_lock:
                    stats["emitted_segments_total"] += 1
                push_live_event(
                    {
                        "type": "segment",
                        "segment": appended,
                        "state": live_snapshot,
                    }
                )

        def process_low_latency(source_rate: int, capture_done: threading.Event | None = None) -> None:
            emit_live_metrics(
                next_buffer_update_seconds=None,
                buffer_interval_seconds=None,
                buffer_queue_depth=0,
            )
            chunk_samples = max(1, int(source_rate * LIVE_LOW_LATENCY_CHUNK_SECONDS))
            flush_samples = max(1, int(source_rate * LIVE_LOW_LATENCY_FLUSH_SECONDS))
            pending = np.empty((0,), dtype=np.float32)
            captured_seconds = 0.0

            while True:
                capture_finished = capture_done.is_set() if capture_done is not None else stop_event.is_set()
                if stop_event.is_set() and capture_finished and frame_queue.empty():
                    break

                try:
                    frame = frame_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                pending = np.concatenate((pending, frame))
                while pending.shape[0] >= chunk_samples:
                    chunk = pending[:chunk_samples]
                    pending = pending[chunk_samples:]
                    chunk_start = captured_seconds
                    captured_seconds += chunk.shape[0] / source_rate
                    emit_segments(
                        raw_chunk=chunk,
                        source_rate=source_rate,
                        chunk_start=chunk_start,
                        beam_size=1,
                        condition_on_previous_text=False,
                        vad_filter=LIVE_USE_VAD_FILTER,
                        emit_from=base_offset + chunk_start,
                    )

            if pending.shape[0] >= flush_samples:
                chunk_start = captured_seconds
                captured_seconds += pending.shape[0] / source_rate
                emit_segments(
                    raw_chunk=pending,
                    source_rate=source_rate,
                    chunk_start=chunk_start,
                    beam_size=1,
                    condition_on_previous_text=False,
                    vad_filter=LIVE_USE_VAD_FILTER,
                    emit_from=base_offset + chunk_start,
                )

        def process_buffered_hq(source_rate: int, capture_done: threading.Event | None = None) -> None:
            chunk_samples = max(1, int(source_rate * LIVE_BUFFERED_CHUNK_SECONDS))
            overlap_samples = max(0, int(source_rate * LIVE_BUFFERED_OVERLAP_SECONDS))
            overlap_samples = min(overlap_samples, chunk_samples - 1)
            step_samples = max(1, chunk_samples - overlap_samples)
            flush_samples = max(1, int(source_rate * LIVE_BUFFERED_FLUSH_SECONDS))
            overlap_seconds = overlap_samples / source_rate

            pending = np.empty((0,), dtype=np.float32)
            next_chunk_start = 0.0
            chunk_index = 0
            last_buffer_metrics_event = 0.0

            transcription_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=128)
            transcription_done = threading.Event()
            transcription_errors: list[Exception] = []

            def emit_buffer_metrics(force: bool = False) -> None:
                nonlocal last_buffer_metrics_event
                now = time.monotonic()
                if not force and (now - last_buffer_metrics_event) < LIVE_BUFFER_METRICS_INTERVAL_SECONDS:
                    return

                interval_seconds = (
                    (chunk_samples / source_rate) if chunk_index == 0 else (step_samples / source_rate)
                )
                remaining_seconds = max(0.0, (chunk_samples - pending.shape[0]) / source_rate)
                queue_depth = transcription_queue.qsize()
                last_buffer_metrics_event = now
                with stats_lock:
                    stats["transcription_queue_max_depth"] = max(
                        int(stats["transcription_queue_max_depth"]), queue_depth
                    )
                emit_live_metrics(
                    next_buffer_update_seconds=remaining_seconds,
                    buffer_interval_seconds=interval_seconds,
                    buffer_queue_depth=queue_depth,
                )

            def enqueue_transcription_chunk(chunk_data: np.ndarray, chunk_start: float, emit_from: float) -> None:
                payload = {
                    "chunk": chunk_data.copy(),
                    "chunk_start": float(chunk_start),
                    "emit_from": float(max(emit_from, chunk_start)),
                }
                try:
                    transcription_queue.put_nowait(payload)
                except queue.Full:
                    try:
                        transcription_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        transcription_queue.put_nowait(payload)
                    except queue.Full:
                        return

                    with stats_lock:
                        stats["dropped_transcription_chunks"] += 1
                        drop_count = int(stats["dropped_transcription_chunks"])
                    if drop_count == 1 or drop_count % 10 == 0:
                        status_payload = update_live_state(
                            message=(
                                f"Transcription backlog detected "
                                f"({drop_count} dropped buffered chunks)."
                            )
                        )
                        push_live_event({"type": "live_state", "state": status_payload})
                emit_buffer_metrics(force=True)

            def transcription_worker() -> None:
                try:
                    while True:
                        try:
                            item = transcription_queue.get(timeout=0.25)
                        except queue.Empty:
                            if stop_event.is_set():
                                continue
                            continue

                        if item is None:
                            break

                        emit_segments(
                            raw_chunk=item["chunk"],
                            source_rate=source_rate,
                            chunk_start=float(item["chunk_start"]),
                            beam_size=5,
                            condition_on_previous_text=True,
                            vad_filter=LIVE_USE_VAD_FILTER,
                            emit_from=float(item["emit_from"]),
                        )
                except Exception as exc:  # pragma: no cover - model/device dependent
                    transcription_errors.append(exc)
                    stop_event.set()
                finally:
                    transcription_done.set()

            worker = threading.Thread(
                target=transcription_worker,
                daemon=True,
                name="buffered-transcription-worker",
            )
            worker.start()
            emit_buffer_metrics(force=True)

            try:
                while True:
                    capture_finished = capture_done.is_set() if capture_done is not None else stop_event.is_set()
                    if stop_event.is_set() and capture_finished and frame_queue.empty():
                        break

                    try:
                        frame = frame_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    pending = np.concatenate((pending, frame))
                    while pending.shape[0] >= chunk_samples:
                        chunk = pending[:chunk_samples]
                        pending = pending[step_samples:]
                        chunk_start = next_chunk_start
                        emit_from = chunk_start if chunk_index == 0 else (chunk_start + overlap_seconds)
                        enqueue_transcription_chunk(chunk, chunk_start, emit_from)
                        next_chunk_start += step_samples / source_rate
                        chunk_index += 1
                    emit_buffer_metrics(force=False)

                if pending.shape[0] >= flush_samples:
                    chunk_start = next_chunk_start
                    pending_duration = pending.shape[0] / source_rate
                    keep_overlap = min(overlap_seconds, max(0.0, pending_duration - 0.25))
                    emit_from = chunk_start if chunk_index == 0 else (chunk_start + keep_overlap)
                    enqueue_transcription_chunk(pending, chunk_start, emit_from)
                emit_buffer_metrics(force=True)
            finally:
                transcription_queue.put(None)
                worker.join(timeout=600.0)
                emit_buffer_metrics(force=True)

            if not transcription_done.is_set():
                raise RuntimeError("Buffered transcription worker did not finish cleanly.")
            if transcription_errors:
                raise RuntimeError(str(transcription_errors[0]))

        if source == "system":
            capture_done = threading.Event()
            capture_error: list[Exception] = []
            source_rate = 48_000
            source_rate_for_session = source_rate
            start_audio_capture_writer(sample_rate=source_rate)

            def loopback_capture_worker() -> None:
                com_initialized = initialize_com_for_thread()
                try:
                    loopback_mic, channels, _, _ = resolve_system_loopback_microphone(selected_device_id)
                    frame_count = 4096
                    with loopback_mic.recorder(samplerate=source_rate, channels=channels) as recorder:
                        while not stop_event.is_set():
                            frame = recorder.record(numframes=frame_count)
                            mono = capture_to_mono(frame)
                            push_frame(mono)
                except Exception as exc:  # pragma: no cover - device/environment specific
                    capture_error.append(exc)
                    stop_event.set()
                finally:
                    if com_initialized:
                        uninitialize_com_for_thread()
                    capture_done.set()

            capture_thread = threading.Thread(
                target=loopback_capture_worker,
                daemon=True,
                name="system-loopback-capture-thread",
            )
            capture_thread.start()
            if capture_mode == "buffered_hq":
                process_buffered_hq(source_rate=source_rate, capture_done=capture_done)
            else:
                process_low_latency(source_rate=source_rate, capture_done=capture_done)
            capture_thread.join(timeout=2.0)
            if capture_error:
                raise RuntimeError(str(capture_error[0]))
        else:
            device_index, channels, source_rate, resolved_label = resolve_live_mic_device(
                selected_device_id
            )
            source_rate_for_session = int(source_rate)
            start_audio_capture_writer(sample_rate=source_rate)

            if resolved_label != selected_device_label:
                status_payload = update_live_state(
                    device_label=resolved_label,
                    message=f"Listening ({source}) on {resolved_label} [{mode_label}]",
                )
                push_live_event({"type": "live_state", "state": status_payload})

            def callback(indata: np.ndarray, _: int, __: Any, status: sd.CallbackFlags) -> None:
                if status:
                    status_payload = update_live_state(message=f"Audio status: {status}")
                    push_live_event({"type": "live_state", "state": status_payload})

                mono = capture_to_mono(indata)
                push_frame(mono)

            stream_kwargs: dict[str, Any] = {
                "device": device_index,
                "channels": channels,
                "samplerate": source_rate,
                "dtype": "float32",
                "callback": callback,
                "blocksize": 0,
            }

            with sd.InputStream(**stream_kwargs):
                if capture_mode == "buffered_hq":
                    process_buffered_hq(source_rate=source_rate)
                else:
                    process_low_latency(source_rate=source_rate)

        cleanup_writers(raise_capture_error=True)
        emit_live_metrics(
            input_level_rms=0.0,
            input_level_peak=0.0,
            input_level_dbfs=LIVE_LEVEL_FLOOR_DBFS,
            next_buffer_update_seconds=None,
            buffer_interval_seconds=None,
            buffer_queue_depth=0,
        )
        final_diarization_status = live_diarization_status
        if diarize and final_diarization_status == "running":
            final_diarization_status = "completed"
        live_diarization_status = final_diarization_status
        state_payload = update_live_state(
            status="idle",
            message="Live transcription stopped.",
            session_id=None,
            started_at=None,
            error=None,
            diarize=bool(diarize),
            diarization_speakers=diarization_speakers,
            diarization_status=final_diarization_status,
            diarization_error=live_diarization_error,
            diarization_seconds=(live_diarization_seconds if diarize else None),
            speaker_count=len(live_speaker_label_map),
            input_level_rms=0.0,
            input_level_peak=0.0,
            input_level_dbfs=LIVE_LEVEL_FLOOR_DBFS,
            input_level_updated_at=time.time(),
            next_buffer_update_seconds=None,
            buffer_interval_seconds=None,
            buffer_queue_depth=0,
        )
        push_live_event({"type": "live_state", "state": state_payload})
    except Exception as exc:
        session_outcome = "error"
        session_error = str(exc)
        try:
            cleanup_writers(raise_capture_error=False)
        except Exception:
            pass
        emit_live_metrics(
            input_level_rms=0.0,
            input_level_peak=0.0,
            input_level_dbfs=LIVE_LEVEL_FLOOR_DBFS,
            next_buffer_update_seconds=None,
            buffer_interval_seconds=None,
            buffer_queue_depth=0,
        )
        if diarize and live_diarization_status == "running":
            live_diarization_status = "failed"
            if not live_diarization_error:
                live_diarization_error = str(exc)
        state_payload = update_live_state(
            status="error",
            message="Live transcription failed.",
            session_id=None,
            started_at=None,
            error=str(exc),
            diarize=bool(diarize),
            diarization_speakers=diarization_speakers,
            diarization_status=live_diarization_status,
            diarization_error=live_diarization_error,
            diarization_seconds=(live_diarization_seconds if diarize else None),
            speaker_count=len(live_speaker_label_map),
            input_level_rms=0.0,
            input_level_peak=0.0,
            input_level_dbfs=LIVE_LEVEL_FLOOR_DBFS,
            input_level_updated_at=time.time(),
            next_buffer_update_seconds=None,
            buffer_interval_seconds=None,
            buffer_queue_depth=0,
        )
        push_live_event({"type": "live_state", "state": state_payload})
    finally:
        try:
            cleanup_writers(raise_capture_error=False)
        except Exception:
            pass

        with stats_lock:
            stats_snapshot = dict(stats)

        capture_duration_seconds = (audio_samples_written / audio_source_rate) if audio_source_rate > 0 else 0.0
        asr_duration_seconds = asr_audio_samples_written / LIVE_TARGET_SAMPLE_RATE
        level_events = int(stats_snapshot.get("level_event_count", 0) or 0)
        rms_sum = float(stats_snapshot.get("level_rms_sum", 0.0) or 0.0)
        rms_avg = (rms_sum / level_events) if level_events > 0 else 0.0
        total_segment_count = len(get_live_segments_copy())

        diagnostics_payload: dict[str, Any] = {
            "session_id": session_id,
            "status": session_outcome,
            "error": session_error,
            "source": source,
            "mode": mode,
            "capture_mode": capture_mode,
            "model": model_name,
            "language": language,
            "device_id": selected_device_id,
            "device_label": selected_device_label,
            "base_offset_seconds": float(base_offset),
            "started_at": session_started_at,
            "ended_at": time.time(),
            "duration_seconds": max(0.0, time.time() - session_started_at),
            "sample_rates": {
                "source_hz": source_rate_for_session,
                "target_hz": LIVE_TARGET_SAMPLE_RATE,
            },
            "audio": {
                "capture_samples": int(audio_samples_written),
                "capture_duration_seconds": float(capture_duration_seconds),
                "asr_samples_16k": int(asr_audio_samples_written),
                "asr_duration_seconds": float(asr_duration_seconds),
            },
            "chunks": {
                "capture_frames_received": int(stats_snapshot.get("frame_chunks_received", 0) or 0),
                "capture_samples_received": int(stats_snapshot.get("frame_samples_received", 0) or 0),
                "transcribe_requests": int(stats_snapshot.get("transcribe_chunks_total", 0) or 0),
                "transcribe_requests_with_text": int(
                    stats_snapshot.get("transcribe_chunks_with_text", 0) or 0
                ),
                "segments_emitted": int(stats_snapshot.get("emitted_segments_total", 0) or 0),
                "segments_total_after_session": int(total_segment_count),
                "asr_audio_chunks_enqueued": int(stats_snapshot.get("asr_audio_chunks_enqueued", 0) or 0),
            },
            "drops": {
                "frame_queue_drops": int(stats_snapshot.get("dropped_frame_chunks", 0) or 0),
                "buffered_queue_drops": int(stats_snapshot.get("dropped_transcription_chunks", 0) or 0),
                "capture_artifact_queue_drops": int(
                    stats_snapshot.get("dropped_capture_artifact_chunks", 0) or 0
                ),
                "asr_artifact_queue_drops": int(stats_snapshot.get("dropped_asr_artifact_chunks", 0) or 0),
            },
            "queues": {
                "frame_queue_max_depth": int(stats_snapshot.get("frame_queue_max_depth", 0) or 0),
                "transcription_queue_max_depth": int(
                    stats_snapshot.get("transcription_queue_max_depth", 0) or 0
                ),
            },
            "levels": {
                "events": level_events,
                "rms_avg": float(rms_avg),
                "rms_max": float(stats_snapshot.get("level_rms_max", 0.0) or 0.0),
                "peak_max": float(stats_snapshot.get("level_peak_max", 0.0) or 0.0),
                "dbfs_min": (
                    float(stats_snapshot["level_dbfs_min"])
                    if stats_snapshot.get("level_dbfs_min") is not None
                    else LIVE_LEVEL_FLOOR_DBFS
                ),
                "dbfs_max": float(stats_snapshot.get("level_dbfs_max", LIVE_LEVEL_FLOOR_DBFS)),
                "dbfs_last": float(stats_snapshot.get("level_dbfs_last", LIVE_LEVEL_FLOOR_DBFS)),
            },
            "writer_errors": {
                "capture": capture_writer_error_message,
                "asr_16k": asr_writer_error_message,
            },
            "diarization": {
                "enabled": bool(diarize),
                "requested_speakers": diarization_speakers,
                "status": live_diarization_status,
                "error": live_diarization_error,
                "seconds": (float(live_diarization_seconds) if diarize else None),
                "speaker_count": len(live_speaker_label_map),
                "device": live_diarization_device if diarize else None,
            },
        }

        try:
            diagnostics_path = LIVE_CAPTURE_DIR / f"live-{session_id}.diagnostics.json"
            diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")
            set_live_diagnostics_artifact(path=diagnostics_path, session_id=session_id)
        except Exception:
            clear_live_diagnostics_artifact(delete_file=False)

        push_live_event({"type": "live_state", "state": build_live_state_response()})
        with live_state_lock:
            live_thread = None
            live_stop_event = None


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

        diarization_status = "pending" if job.diarize else "disabled"
        diarization_error: str | None = None
        diarization_seconds: float | None = None
        speaker_count = 0

        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue
            segment_index += 1
            start = float(segment.start)
            end = float(segment.end)

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
                progress = min(0.12 + (fraction * 0.80), 0.92)
            else:
                progress = min(0.12 + (segment_index * 0.0025), 0.92)

            update_job(
                job_id,
                progress=progress,
                message=f"Transcribing segment {segment_index}",
                segment_count=segment_index,
            )

        if job.diarize:
            if not segments_list:
                diarization_status = "skipped"
                diarization_error = "No transcript segments were produced."
            else:
                diarization_started = time.time()
                update_job(job_id, message="Running speaker diarization", progress=0.94)
                try:
                    diarization_segments = run_speaker_diarization(audio_path, job.diarization_speakers)
                    speaker_count = assign_speakers_to_segments(segments_list, diarization_segments)
                    diarization_status = "completed"
                    update_job(job_id, message="Finalizing outputs", progress=0.98)
                except Exception as diarization_exc:
                    diarization_status = "failed"
                    diarization_error = str(diarization_exc)
                diarization_seconds = max(0.0, time.time() - diarization_started)

        with txt_path.open("w", encoding="utf-8") as txt_file, srt_path.open("w", encoding="utf-8") as srt_file:
            txt_file.write("# Transcription\n")
            txt_file.write(f"# File: {job.original_filename}\n")
            txt_file.write(f"# Model: {job.model}\n")
            txt_file.write(f"# Device: {selected[0]}\n")
            txt_file.write(f"# Compute type: {selected[1]}\n")
            txt_file.write(
                f"# Language: {info.language} (probability={info.language_probability:.3f})\n"
            )
            txt_file.write(f"# Diarization: {diarization_status}\n")
            if job.diarization_speakers is not None:
                txt_file.write(f"# Requested speakers: {job.diarization_speakers}\n")
            txt_file.write(f"# Detected speakers: {speaker_count}\n")
            if diarization_error:
                txt_file.write(f"# Diarization error: {diarization_error}\n")
            txt_file.write("\n")

            for segment in segments_list:
                segment_text = format_segment_text(
                    str(segment["text"]), str(segment.get("speaker") or "")
                )
                txt_file.write(f"[{format_hhmmss(float(segment['start']))}] {segment_text}\n")

            for index, segment in enumerate(segments_list, start=1):
                segment_text = format_segment_text(
                    str(segment["text"]), str(segment.get("speaker") or "")
                )
                srt_file.write(f"{index}\n")
                srt_file.write(
                    f"{format_srt_time(float(segment['start']))} --> {format_srt_time(float(segment['end']))}\n"
                )
                srt_file.write(segment_text + "\n\n")

        with segments_path.open("w", encoding="utf-8") as segment_file:
            json.dump(segments_list, segment_file, ensure_ascii=False, indent=2)

        completion_message = "Completed"
        if diarization_status == "failed":
            completion_message = "Completed (diarization unavailable)"
        elif diarization_status == "completed" and speaker_count > 0:
            completion_message = "Completed with speaker labels"

        update_job(
            job_id,
            status="completed",
            message=completion_message,
            progress=1.0,
            txt_path=str(txt_path),
            srt_path=str(srt_path),
            segments_path=str(segments_path),
            language=info.language,
            language_probability=float(info.language_probability),
            segment_count=segment_index,
            diarization_status=diarization_status,
            diarization_error=diarization_error,
            diarization_seconds=diarization_seconds,
            speaker_count=speaker_count,
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


@app.on_event("startup")
async def startup_event() -> None:
    global live_main_loop
    live_main_loop = asyncio.get_running_loop()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    with live_state_lock:
        stop_event = live_stop_event
    if stop_event is not None:
        stop_event.set()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form("small"),
    export_folder: str = Form("default"),
    diarize: str = Form("false"),
    diarization_speakers: str | None = Form(None),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if is_live_active():
        raise HTTPException(
            status_code=409,
            detail="Live transcription is active. Stop it before submitting a file job.",
        )
    model_name = (model or "small").strip().lower()
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{model_name}'.")
    try:
        diarize_enabled = parse_form_bool(diarize, default=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    diarization_speakers_value: int | None = None
    if diarization_speakers is not None and diarization_speakers.strip():
        try:
            diarization_speakers_value = int(diarization_speakers.strip())
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="diarization_speakers must be an integer between 1 and 20.",
            ) from exc
        if diarization_speakers_value < 1 or diarization_speakers_value > 20:
            raise HTTPException(
                status_code=400,
                detail="diarization_speakers must be between 1 and 20.",
            )
    if diarization_speakers_value is not None and not diarize_enabled:
        raise HTTPException(
            status_code=400,
            detail="diarization_speakers was provided but diarize is disabled.",
        )

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
        model=model_name,
        export_folder=export_folder or "default",
        audio_path=str(upload_path),
        status="queued",
        progress=0.0,
        message="Queued",
        created_at=now,
        updated_at=now,
        diarize=diarize_enabled,
        diarization_speakers=diarization_speakers_value,
        diarization_status="pending" if diarize_enabled else "disabled",
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


@app.get("/api/live/state")
def get_live_state() -> dict[str, Any]:
    return {"state": build_live_state_response()}


@app.get("/api/live/devices")
def get_live_devices() -> dict[str, list[dict[str, Any]]]:
    try:
        mic_devices = list_live_mic_devices()
    except Exception:
        mic_devices = []
    try:
        system_devices = list_live_system_devices()
    except Exception:
        system_devices = []
    return {"mic": mic_devices, "system": system_devices}


@app.get("/api/live/segments")
def get_live_segments() -> list[dict[str, Any]]:
    return get_live_segments_copy()


@app.post("/api/live/start")
def start_live(request: LiveStartRequest) -> dict[str, Any]:
    global live_thread, live_stop_event

    source = request.source.strip().lower()
    mode = request.mode.strip().lower()
    model_name = request.model.strip().lower() or LIVE_DEFAULT_MODEL
    requested_language = request.language.strip().lower() if request.language else LIVE_DEFAULT_LANGUAGE
    language = None if requested_language in {"", "auto", "detect"} else requested_language
    requested_device_id = request.device_id.strip() if request.device_id else None
    capture_mode = request.capture_mode.strip().lower() if request.capture_mode else "buffered_hq"
    diarize = bool(request.diarize)
    diarization_speakers = int(request.diarization_speakers) if request.diarization_speakers is not None else None
    if source not in {"mic", "system"}:
        raise HTTPException(status_code=400, detail="source must be 'mic' or 'system'")
    if mode not in {"new", "append"}:
        raise HTTPException(status_code=400, detail="mode must be 'new' or 'append'")
    if capture_mode not in {"low_latency", "buffered_hq"}:
        raise HTTPException(status_code=400, detail="capture_mode must be 'low_latency' or 'buffered_hq'")
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{model_name}'.")
    if language is not None and not re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2})?", language):
        raise HTTPException(status_code=400, detail="language must be auto/detect or a valid code like 'en'.")
    if diarization_speakers is not None and (diarization_speakers < 1 or diarization_speakers > 20):
        raise HTTPException(status_code=400, detail="diarization_speakers must be between 1 and 20.")
    if diarization_speakers is not None and not diarize:
        raise HTTPException(
            status_code=400, detail="diarization_speakers was provided but diarize is disabled."
        )
    if diarize and capture_mode != "buffered_hq":
        raise HTTPException(
            status_code=400,
            detail="Live diarization is only supported in buffered_hq capture mode.",
        )
    if has_active_batch_jobs():
        raise HTTPException(
            status_code=409,
            detail="A file transcription job is running. Wait for it to finish first.",
        )

    try:
        selected_device_id, selected_device_label = resolve_selected_live_device(
            source, requested_device_id
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with live_state_lock:
        if live_state["status"] in {"starting", "running", "stopping"}:
            raise HTTPException(status_code=409, detail="Live transcription is already active.")

        if mode == "new":
            live_segments.clear()
        clear_paths = [
            live_audio_meta.get("path"),
            live_asr_audio_meta.get("path"),
            live_diagnostics_meta.get("path"),
        ]
        live_audio_meta.update({"path": None, "sample_rate": None, "duration_seconds": 0.0, "session_id": None})
        live_asr_audio_meta.update({"path": None, "sample_rate": None, "duration_seconds": 0.0, "session_id": None})
        live_diagnostics_meta.update({"path": None, "session_id": None})

        base_offset = float(live_segments[-1]["end"]) if live_segments else 0.0
        session_id = uuid.uuid4().hex
        live_stop_event = threading.Event()
        live_state.update(
            {
                "status": "starting",
                "source": source,
                "mode": mode,
                "model": model_name,
                "language": language,
                "capture_mode": capture_mode,
                "diarize": diarize,
                "diarization_speakers": diarization_speakers,
                "diarization_status": "pending" if diarize else "disabled",
                "diarization_error": None,
                "diarization_seconds": (0.0 if diarize else None),
                "speaker_count": len(
                    {str(item.get("speaker")) for item in live_segments if str(item.get("speaker") or "").strip()}
                ),
                "device_id": selected_device_id,
                "device_label": selected_device_label,
                "message": "Starting live transcription...",
                "error": None,
                "session_id": session_id,
                "started_at": time.time(),
                "updated_at": time.time(),
                "segment_count": len(live_segments),
                "input_level_rms": 0.0,
                "input_level_peak": 0.0,
                "input_level_dbfs": LIVE_LEVEL_FLOOR_DBFS,
                "input_level_updated_at": time.time(),
                "next_buffer_update_seconds": (
                    LIVE_BUFFERED_CHUNK_SECONDS if capture_mode == "buffered_hq" else None
                ),
                "buffer_interval_seconds": (
                    LIVE_BUFFERED_CHUNK_SECONDS if capture_mode == "buffered_hq" else None
                ),
                "buffer_queue_depth": 0,
            }
        )
        state_payload = dict(live_state)

    for clear_path in clear_paths:
        if not clear_path:
            continue
        try:
            stale_path = Path(str(clear_path))
            if stale_path.exists():
                stale_path.unlink()
        except Exception:
            pass

    thread = threading.Thread(
        target=run_live_transcription,
        args=(
            session_id,
            source,
            mode,
            model_name,
            language,
            capture_mode,
            diarize,
            diarization_speakers,
            selected_device_id,
            selected_device_label,
            base_offset,
            live_stop_event,
        ),
        daemon=True,
        name="live-transcription-thread",
    )
    live_thread = thread
    thread.start()

    push_live_event({"type": "live_state", "state": state_payload})
    with live_state_lock:
        metrics_payload = build_live_metrics_response_locked()
    push_live_event({"type": "live_metrics", "metrics": metrics_payload})
    return {"state": build_live_state_response()}


@app.post("/api/live/stop")
def stop_live() -> dict[str, Any]:
    with live_state_lock:
        status = live_state["status"]
        stop_event = live_stop_event
        if status not in {"starting", "running"} or stop_event is None:
            return {"state": build_live_state_response_locked()}
        live_state["status"] = "stopping"
        live_state["message"] = "Stopping live transcription..."
        live_state["updated_at"] = time.time()
        state_payload = build_live_state_response_locked()

    stop_event.set()
    push_live_event({"type": "live_state", "state": state_payload})
    return {"state": build_live_state_response()}


@app.get("/api/live/download/{kind}")
def download_live_artifact(kind: str) -> Response:
    if kind == "audio":
        artifact = get_live_audio_artifact_copy()
        audio_path_raw = artifact.get("path")
        if not audio_path_raw:
            raise HTTPException(status_code=409, detail="No live audio capture is available to download.")
        audio_path = Path(str(audio_path_raw))
        if not audio_path.exists():
            clear_live_audio_artifact(delete_file=False)
            raise HTTPException(status_code=404, detail="Live audio capture is missing.")
        return FileResponse(audio_path, media_type="audio/wav", filename=audio_path.name)
    if kind == "audio_16k":
        artifact = get_live_asr_audio_artifact_copy()
        audio_path_raw = artifact.get("path")
        if not audio_path_raw:
            raise HTTPException(
                status_code=409,
                detail="No 16 kHz Whisper-input debug audio is available to download.",
            )
        audio_path = Path(str(audio_path_raw))
        if not audio_path.exists():
            clear_live_asr_audio_artifact(delete_file=False)
            raise HTTPException(status_code=404, detail="16 kHz Whisper-input debug audio is missing.")
        return FileResponse(audio_path, media_type="audio/wav", filename=audio_path.name)
    if kind == "diagnostics":
        artifact = get_live_diagnostics_artifact_copy()
        diagnostics_path_raw = artifact.get("path")
        if not diagnostics_path_raw:
            raise HTTPException(status_code=409, detail="No live diagnostics dump is available to download.")
        diagnostics_path = Path(str(diagnostics_path_raw))
        if not diagnostics_path.exists():
            clear_live_diagnostics_artifact(delete_file=False)
            raise HTTPException(status_code=404, detail="Live diagnostics dump is missing.")
        return FileResponse(diagnostics_path, media_type="application/json", filename=diagnostics_path.name)

    segments = get_live_segments_copy()
    if not segments:
        raise HTTPException(status_code=409, detail="No live transcript is available to download.")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if kind == "txt":
        body = build_txt_content(segments, title=stamp)
        filename = f"live-{stamp}.transcript.txt"
        media_type = "text/plain; charset=utf-8"
    elif kind == "srt":
        body = build_srt_content(segments)
        filename = f"live-{stamp}.transcript.srt"
        media_type = "application/x-subrip"
    else:
        raise HTTPException(status_code=404, detail="Unknown live artifact type.")

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=body.encode("utf-8"), media_type=media_type, headers=headers)


@app.websocket("/ws/live")
async def live_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    live_clients.add(websocket)
    await websocket.send_json({"type": "live_state", "state": build_live_state_response()})
    with live_state_lock:
        metrics_payload = build_live_metrics_response_locked()
    await websocket.send_json({"type": "live_metrics", "metrics": metrics_payload})
    await websocket.send_json({"type": "live_snapshot", "segments": get_live_segments_copy()})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        live_clients.discard(websocket)


@app.get("/api/models")
def list_models() -> dict[str, list[str]]:
    return {"models": AVAILABLE_MODELS}


@app.post("/api/models/preload")
def preload_model(request: ModelPreloadRequest) -> dict[str, Any]:
    model_name = (request.model or "small").strip().lower()
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{model_name}'.")

    cached_before = is_model_cached(model_name)
    started = time.time()
    try:
        _, selected = load_model(model_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to preload model '{model_name}': {exc}") from exc

    return {
        "model": model_name,
        "device": selected[0],
        "compute_type": selected[1],
        "cached_before": cached_before,
        "load_seconds": max(0.0, time.time() - started),
    }
