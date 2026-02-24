from __future__ import annotations

import asyncio
import ctypes
import json
import queue
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
            return np.frombuffer(buffer_view, dtype=dtype, count=count)
        except TypeError:
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

for directory in (UPLOAD_DIR, OUTPUT_DIR):
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


class LiveStartRequest(BaseModel):
    source: str = "mic"
    mode: str = "new"
    model: str = LIVE_DEFAULT_MODEL
    language: str | None = LIVE_DEFAULT_LANGUAGE
    device_id: str | None = None
    capture_mode: str = "buffered_hq"


class ModelPreloadRequest(BaseModel):
    model: str = "small"


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)
model_cache: dict[tuple[str, str, str], WhisperModel] = {}
model_cache_lock = threading.Lock()

live_state_lock = threading.Lock()
live_segments: list[dict[str, Any]] = []
live_state: dict[str, Any] = {
    "status": "idle",
    "source": "mic",
    "mode": "new",
    "model": LIVE_DEFAULT_MODEL,
    "language": LIVE_DEFAULT_LANGUAGE,
    "capture_mode": "buffered_hq",
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
    state_copy["segment_count"] = segment_count
    if segment_count > 0:
        state_copy["downloads"] = {
            "txt": "/api/live/download/txt",
            "srt": "/api/live/download/srt",
        }
    return state_copy


def build_live_state_response() -> dict[str, Any]:
    with live_state_lock:
        return build_live_state_response_locked()


def update_live_state(**updates: Any) -> dict[str, Any]:
    with live_state_lock:
        live_state.update(updates)
        live_state["updated_at"] = time.time()
        live_state["segment_count"] = len(live_segments)
        return dict(live_state)


def update_live_metrics(**updates: Any) -> dict[str, Any]:
    with live_state_lock:
        live_state.update(updates)
        if {"input_level_rms", "input_level_peak", "input_level_dbfs"} & set(updates.keys()):
            live_state["input_level_updated_at"] = time.time()
        return build_live_metrics_response_locked()


def append_live_segment(start: float, end: float, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with live_state_lock:
        segment = {
            "index": len(live_segments) + 1,
            "start": float(start),
            "end": float(max(end, start)),
            "text": text,
        }
        live_segments.append(segment)
        live_state["segment_count"] = len(live_segments)
        live_state["updated_at"] = time.time()
        return dict(segment), dict(live_state)


def build_txt_content(segments: list[dict[str, Any]], title: str) -> str:
    lines = ["# Live Transcription", f"# Session: {title}", ""]
    for segment in segments:
        lines.append(f"[{format_hhmmss(float(segment['start']))}] {segment['text']}")
    return "\n".join(lines).rstrip() + "\n"


def build_srt_content(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(str(index))
        blocks.append(
            f"{format_srt_time(float(segment['start']))} --> {format_srt_time(float(segment['end']))}"
        )
        blocks.append(str(segment["text"]))
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


def capture_to_mono(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 1:
        return frame.astype(np.float32, copy=False)
    if frame.ndim != 2:
        return np.asarray(frame, dtype=np.float32).reshape(-1)

    channels = frame.astype(np.float32, copy=False)
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
) -> list[dict[str, Any]]:
    chunk = resample_audio(raw_chunk, source_rate, LIVE_TARGET_SAMPLE_RATE)
    if chunk.size == 0:
        return []

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
        start = base_offset + chunk_start + float(segment.start)
        end = base_offset + chunk_start + float(segment.end)
        segments.append({"start": start, "end": max(end, start), "text": text})
    return segments


def run_live_transcription(
    session_id: str,
    source: str,
    mode: str,
    model_name: str,
    language: str | None,
    capture_mode: str,
    selected_device_id: str,
    selected_device_label: str,
    base_offset: float,
    stop_event: threading.Event,
) -> None:
    global live_thread, live_stop_event

    def emit_live_metrics(**updates: Any) -> None:
        metrics_payload = update_live_metrics(**updates)
        push_live_event({"type": "live_metrics", "metrics": metrics_payload})

    try:
        model, _ = load_model(model_name)
        mode_label = "buffered HQ" if capture_mode == "buffered_hq" else "low latency"
        initial_buffer_eta = LIVE_BUFFERED_CHUNK_SECONDS if capture_mode == "buffered_hq" else None
        initial_buffer_interval = LIVE_BUFFERED_CHUNK_SECONDS if capture_mode == "buffered_hq" else None
        state_payload = update_live_state(
            status="running",
            message=f"Listening ({source}) on {selected_device_label} [{mode_label}]",
            source=source,
            mode=mode,
            model=model_name,
            language=language,
            capture_mode=capture_mode,
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

        frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1024)
        dropped_frame_chunks = 0
        dropped_transcription_chunks = 0
        dropped_lock = threading.Lock()
        last_level_event = 0.0

        def maybe_emit_level(frame: np.ndarray, force: bool = False) -> None:
            nonlocal last_level_event

            now = time.monotonic()
            if not force and (now - last_level_event) < LIVE_LEVEL_EVENT_INTERVAL_SECONDS:
                return
            if frame.size == 0:
                return

            peak = float(np.max(np.abs(frame)))
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))
            if rms <= 1e-9:
                dbfs = LIVE_LEVEL_FLOOR_DBFS
            else:
                dbfs = float(max(LIVE_LEVEL_FLOOR_DBFS, min(0.0, 20.0 * np.log10(rms))))

            last_level_event = now
            emit_live_metrics(
                input_level_rms=rms,
                input_level_peak=peak,
                input_level_dbfs=dbfs,
            )

        def push_frame(frame: np.ndarray) -> None:
            nonlocal dropped_frame_chunks
            if frame.size == 0:
                return

            maybe_emit_level(frame)

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

                with dropped_lock:
                    dropped_frame_chunks += 1
                    drop_count = dropped_frame_chunks
                if drop_count == 1 or drop_count % 25 == 0:
                    status_payload = update_live_state(
                        message=f"Capture backlog detected ({drop_count} dropped frame chunks)."
                    )
                    push_live_event({"type": "live_state", "state": status_payload})

        def emit_segments(
            raw_chunk: np.ndarray,
            source_rate: int,
            chunk_start: float,
            beam_size: int,
            condition_on_previous_text: bool,
            vad_filter: bool,
            emit_from: float,
        ) -> None:
            segments = transcribe_live_chunk(
                model=model,
                raw_chunk=raw_chunk,
                source_rate=source_rate,
                chunk_start=chunk_start,
                base_offset=base_offset,
                beam_size=beam_size,
                condition_on_previous_text=condition_on_previous_text,
                vad_filter=vad_filter,
                language=language,
            )
            for segment in segments:
                if float(segment["end"]) <= (emit_from + 0.02):
                    continue
                if float(segment["start"]) < emit_from:
                    segment["start"] = emit_from
                appended, live_snapshot = append_live_segment(
                    start=segment["start"], end=segment["end"], text=segment["text"]
                )
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
            nonlocal dropped_transcription_chunks

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
                last_buffer_metrics_event = now
                emit_live_metrics(
                    next_buffer_update_seconds=remaining_seconds,
                    buffer_interval_seconds=interval_seconds,
                    buffer_queue_depth=transcription_queue.qsize(),
                )

            def enqueue_transcription_chunk(chunk_data: np.ndarray, chunk_start: float, emit_from: float) -> None:
                nonlocal dropped_transcription_chunks
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

                    with dropped_lock:
                        dropped_transcription_chunks += 1
                        drop_count = dropped_transcription_chunks
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

        emit_live_metrics(
            input_level_rms=0.0,
            input_level_peak=0.0,
            input_level_dbfs=LIVE_LEVEL_FLOOR_DBFS,
            next_buffer_update_seconds=None,
            buffer_interval_seconds=None,
            buffer_queue_depth=0,
        )
        state_payload = update_live_state(
            status="idle",
            message="Live transcription stopped.",
            session_id=None,
            started_at=None,
            error=None,
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
        emit_live_metrics(
            input_level_rms=0.0,
            input_level_peak=0.0,
            input_level_dbfs=LIVE_LEVEL_FLOOR_DBFS,
            next_buffer_update_seconds=None,
            buffer_interval_seconds=None,
            buffer_queue_depth=0,
        )
        state_payload = update_live_state(
            status="error",
            message="Live transcription failed.",
            session_id=None,
            started_at=None,
            error=str(exc),
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

    thread = threading.Thread(
        target=run_live_transcription,
        args=(
            session_id,
            source,
            mode,
            model_name,
            language,
            capture_mode,
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
