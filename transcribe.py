from __future__ import annotations

import argparse
import time
from pathlib import Path

from faster_whisper import WhisperModel


def fmt_hhmmss(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def fmt_srt_time(seconds: float) -> str:
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
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            return model, (device, compute_type)
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc

    raise RuntimeError(f"Could not initialize model '{model_name}'. Last error: {last_error}")


def transcribe_file(audio_path: Path, model_name: str) -> None:
    model, selected = load_model(model_name)

    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=1,
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=True,
    )

    segments = [segment for segment in segments_iter if segment.text.strip()]

    txt_output = audio_path.with_suffix(".transcript.txt")
    srt_output = audio_path.with_suffix(".transcript.srt")

    with txt_output.open("w", encoding="utf-8") as txt_file:
        txt_file.write("# Transcription\n")
        txt_file.write(f"# File: {audio_path.name}\n")
        txt_file.write(f"# Model: {model_name}\n")
        txt_file.write(f"# Device: {selected[0]}\n")
        txt_file.write(f"# Compute type: {selected[1]}\n")
        txt_file.write(
            f"# Language: {info.language} (probability={info.language_probability:.3f})\n\n"
        )
        for segment in segments:
            txt_file.write(f"[{fmt_hhmmss(segment.start)}] {segment.text.strip()}\n")

    with srt_output.open("w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(segments, start=1):
            srt_file.write(f"{index}\n")
            srt_file.write(f"{fmt_srt_time(segment.start)} --> {fmt_srt_time(segment.end)}\n")
            srt_file.write(segment.text.strip() + "\n\n")

    print("TRANSCRIPTION_COMPLETE")
    print(f"device={selected[0]}")
    print(f"compute_type={selected[1]}")
    print(f"language={info.language}")
    print(f"language_probability={info.language_probability:.3f}")
    print(f"segments={len(segments)}")
    print(f"txt_output={txt_output}")
    print(f"srt_output={srt_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with faster-whisper.")
    parser.add_argument("audio_file", type=Path, help="Path to the input audio file (e.g. .mp3)")
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size/name (tiny, base, small, medium, large-v3, ...). Default: small",
    )
    args = parser.parse_args()

    audio_path = args.audio_file
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start = time.time()
    transcribe_file(audio_path, args.model)
    elapsed = time.time() - start
    print(f"elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
