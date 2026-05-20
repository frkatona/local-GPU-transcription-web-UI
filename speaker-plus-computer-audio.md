# Speaker + Computer Audio Plan

## Goal

Support live transcription from both:

- the system audio loopback stream
- the microphone stream

at the same time, with output that is still compatible with the current live transcript UI, downloads, and buffered/low-latency processing paths.

## Current State

Today the app chooses exactly one live source:

- `system`: captured through `resolve_system_loopback_microphone(...)`
- `mic`: captured through `resolve_live_mic_device(...)`

That choice flows into `run_live_transcription(...)`, where both capture modes eventually feed a single mono stream through:

- `push_frame(...)`
- `frame_queue`
- `process_low_latency(...)` or `process_buffered_hq(...)`
- `emit_segments(...)`

This is good news, because the transcription pipeline already expects one normalized mono stream. The easiest path is to make a new "both" mode that still produces one mixed stream for ASR while preserving the original stems for debugging and future improvements.

## Recommended Approach

Use a phased rollout:

1. Capture both sources concurrently.
2. Resample/synchronize them to one internal mix rate.
3. Mix them into one mono stream and feed the existing ASR path.
4. Save the mixed stream plus optional per-source stems.
5. Add source-aware metadata now, and leave per-source transcript separation as a later improvement.

This keeps the current live architecture mostly intact.

## API and UI Changes

### Frontend

Add a third live source option:

- `system`
- `mic`
- `both`

When `both` is selected, show two device pickers instead of one:

- `System Device`
- `Microphone Device`

Optional nice-to-haves:

- show `Input Level (System + Mic)` in the diagnostics card
- add small per-source level readouts later
- show a warning that physical speaker playback may leak into the mic and cause duplicate words

### Backend request model

Extend `LiveStartRequest` so `both` mode can carry both device ids:

- `source: "system" | "mic" | "both"`
- `device_id` can remain for single-source compatibility
- add `system_device_id`
- add `mic_device_id`

For single-source starts, keep the current behavior unchanged.

## Capture Design

### New mode

Add a new source branch in `run_live_transcription(...)`:

- current branches: `system` and `mic`
- new branch: `both`

### Capture workers

Spawn two capture inputs in parallel:

- a loopback worker using the existing `resolve_system_loopback_microphone(...)` path
- a mic worker using the existing `sounddevice.InputStream(...)` path

Each worker should emit frames into its own queue as small records like:

```text
{
  "source": "system" | "mic",
  "samples": np.ndarray,
  "sample_rate": int,
  "captured_at": monotonic_timestamp
}
```

Do not push either source directly into the current shared `frame_queue`. Instead, add a small mixer stage between capture and transcription.

## Synchronization and Mixing

### Internal target rate

Use one internal mixing rate for both sources before ASR:

- recommended: `48_000 Hz`

Why:

- system loopback already uses `48_000`
- it avoids repeated rate churn before the final Whisper resample to `16 kHz`
- it is a good rate for alignment and diagnostics artifacts

### Alignment strategy

Both streams will drift slightly and will not arrive on identical callback boundaries. To combine them cleanly:

1. Resample each incoming frame to the internal mix rate.
2. Convert each frame to mono float32.
3. Timestamp frames with `time.monotonic()` at capture time.
4. Write them into short per-source ring buffers.
5. A mixer thread reads from both buffers in fixed windows, for example 20-40 ms.

If one side is temporarily missing for a window, mix silence for that side rather than blocking the whole transcription path.

### Mix formula

Start simple and conservative:

```text
mixed = 0.5 * system + 0.5 * mic
mixed = clip(mixed, -1.0, 1.0)
```

Then add light protection:

- peak normalization or limiter only if clipping is detected
- optional independent gain trims per source later

Avoid aggressive noise suppression or echo cancellation in phase 1. Those features can easily make the transcript worse.

## Transcription Strategy

### Phase 1 recommendation: transcribe the mixed stream

Feed the mixed mono stream into the existing path:

- `push_frame(...)`
- `frame_queue`
- `process_low_latency(...)` / `process_buffered_hq(...)`

This is the lowest-risk implementation because it reuses the current buffering, metrics, downloads, and segment append flow.

### Important limitation

A mixed transcript will not automatically tell us whether text came from:

- the human at the mic
- the computer audio

That is acceptable for phase 1 if the main goal is "hear both and transcribe both."

## Source Attribution Options

If we want the transcript to show whether text came from the speaker or the computer, there are two realistic follow-up paths.

### Option A: source tagging on top of the mixed transcript

Keep phase 1 mixed ASR, but save both stems and later estimate which source dominated each transcript segment by comparing segment timing against:

- mic stem energy
- system stem energy

Possible labels:

- `MIC`
- `SYSTEM`
- `MIXED`

Pros:

- small change on top of the recommended design
- preserves one coherent transcript

Cons:

- attribution is heuristic, not true source separation

### Option B: dual transcription plus merge

Run Whisper separately on:

- the mic stem
- the system stem

Then merge the two transcripts into one time-ordered stream.

Pros:

- better source labeling
- easier to force labels like `MIC:` and `SYSTEM:`

Cons:

- roughly doubles live ASR cost
- makes chunk merging and deduplication much harder
- overlapping speech becomes messy fast

Recommendation: do not start with dual transcription.

## Diarization Considerations

Speaker diarization becomes trickier in `both` mode.

### Recommended rule for phase 1

Treat system audio as non-diarized content and only consider diarization for the mic contribution later.

Practical choices:

- simplest: disable diarization when `source == "both"`
- better follow-up: allow diarization only on the mic stem, not on the mixed stream

Running diarization on the mixed stream is likely to mislabel computer audio as another "speaker."

## Artifacts and Downloads

In `both` mode, save:

- `live-<session>.mixed.wav`
- `live-<session>.mic.wav`
- `live-<session>.system.wav`
- existing Whisper `16 kHz` debug wav for the mixed ASR input
- diagnostics JSON describing both sources

The transcript downloads can remain:

- TXT
- SRT
- Markdown

with optional metadata headers such as:

- `# Live source: both`
- `# System device: ...`
- `# Microphone device: ...`

## Live Metrics

Extend live metrics to track:

- mixed input level
- mic input level
- system input level
- queue depth for each capture buffer
- mixer underruns or dropped windows

The current UI can keep showing only the mixed level at first.

## Failure Handling

If one source fails during a `both` session:

- keep transcribing the surviving source if possible
- push a live-state warning to the UI
- mark diagnostics with which source failed

Do not fail the whole session unless both sources are gone or the mixer cannot continue.

## Testing Plan

### Manual tests

1. System audio only still works unchanged.
2. Mic only still works unchanged.
3. Both mode captures a podcast/video plus spoken mic input in the same session.
4. Buffered HQ still produces coherent segments with long captures.
5. Low latency still emits text continuously with both sources active.
6. Device unplug/reselection failures degrade gracefully.
7. Downloads contain the expected mixed/stem files.

### Edge cases

- mic hears the laptop speakers and duplicates the system track
- one source is much louder than the other
- one source starts late
- different hardware sample rates
- queue backlog under heavy GPU load

## Suggested Implementation Order

1. Add `both` to the frontend source selector.
2. Extend the live start API to accept both device ids.
3. Add dual capture workers and a mixer thread in `run_live_transcription(...)`.
4. Feed the mixed stream into the existing transcription path.
5. Save mixed plus per-source artifacts.
6. Add diagnostics and UI labels for `both`.
7. Decide whether to disable or redesign diarization for combined mode.
8. Only after that, consider source attribution or dual-ASR merging.

## Bottom Line

The cleanest first implementation is:

- capture loopback and mic simultaneously
- align and mix them into one mono stream
- reuse the existing live Whisper pipeline
- preserve both original stems for debugging and future source-aware features

That gives the feature with the least disruption to the current architecture, and it leaves room for smarter per-source labeling later.
