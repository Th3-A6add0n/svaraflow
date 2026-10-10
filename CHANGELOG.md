# Changelog

All notable changes to SvaraFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Workspace and tooling**

- Cargo workspace with six crates: `svaraflow-core`, `svaraflow-asr`,
  `svaraflow-text`, `svaraflow-render`, `svaraflow-rpc`, `svaraflow-cli`.
- CI workflow running fmt, clippy, test on ubuntu + windows, cargo-deny.
- `deny.toml` rejecting GPL and AGPL.
- `.gitattributes` forcing LF on Rust sources, test fixtures, TOML,
  Markdown and YAML, so byte-for-byte golden comparisons survive a Windows
  checkout.
- `tools/check-env.ps1` environment probe.
- `tools/convert-model.ps1`: Hugging Face Whisper checkpoint → GGML, with
  a bfloat16→float32 normalisation pass, whisper.cpp quantization, and a
  SHA-256 + size report ready to paste into `models.json`.

**`svaraflow-core`**

- `timeline`: `Timeline`, `Track`, `Clip`, `Fps`, `TrackKind` with serde
  and full validation. Per-kind track index uniqueness.
- `audio`: analysis-signal pipeline (symphonia → ffmpeg → resample →
  speed → placement → average mix).
- `audio::envelope`: dBFS envelope, 10 ms hop, 20 ms window, 3-frame
  median.
- `silence`: threshold detector, `detect_silence_with_vad`,
  `auto_noise_level_from_vad`, outer-edge coverage, millisecond
  parameters, five built-in presets (`relaxed`, `natural`, `brisk`,
  `tight`, `max`), `min_between_cuts_ms`, `SnapMode::{Inward, Nearest}`,
  and `AnalysisReport` explainability data.
- `cut`: `CutMode::{Remove, Keep, Mute, Gaps}`, non-destructive.
- `vad`: engine-agnostic VAD surface.
- `xmeml`: FCP7 XML v4 with bare `<!DOCTYPE xmeml>`, per-source `<file>`
  definitions with references, `<uuid>`, `<timecode>`, empty video
  `<track/>` for audio-only timelines, `<sourcetrack>` on every clipitem,
  stereo L/R linked clipitems, and verbatim-prefix stripping in
  `encode_path_url`.
- `edl`: CMX3600 EDL.

**`svaraflow-asr`**

- `models`: pinned manifest (`models.json`) plus a resumable,
  SHA-256-verifying downloader. `models.json` ships with `url` and
  `sha256` marked `"TODO"` so the tool is testable before the hashes are
  filled in.
- `transcribe`: whisper.cpp engine. VAD-driven chunking with 0.3 s
  overlap, greedy or beam decoding, temperature fallback, no-speech
  threshold, repetition guard for repeated n-grams, optional initial
  prompt built from a vocabulary list. Token-level DTW timestamps with
  proportional fallback flagged `estimated`.
- `cache`: transcript cache keyed on audio SHA-256, model name, and
  output-affecting settings. File-backed under the platform cache
  directory.
- `exports`: SRT, WebVTT, ASS with `\kf` karaoke tags, transcript JSON.
- `bench`: automatic tier selection. Runs the Standard tier over a
  20-second benchmark clip when present and picks Standard at 2x real
  time or faster; falls back to a CPU/GPU heuristic otherwise.
- `vad`: Silero VAD v5 through ONNX Runtime with `load-dynamic`.

**`svaraflow-cli`**

- `timeline from-media` with `--fps`, `--width`, `--height`. Handles
  video + audio, video only, and audio only.
- `silence analyze` with `--json`, `--report`.
- `silence apply` with `--xml`, `--xml-profile minimal|cuts|full`,
  `--edl`, `--sequence-name`, `--report`.
- `silence presets` (table of built-in and user presets).
- `silence preset save|rename|delete|list|import|export`.
- `transcribe <file>` with `--tier auto|light|standard`, `--json`,
  `--out-json`, `--srt`, `--vtt`, `--ass`, `--vad-model`, `--no-dtw`,
  `--beam`, `--vocabulary`, `--language`, `--no-cache`. Prints the
  real-time factor.
- `models list|download|hash|remove`.
- `--preset relaxed|natural|brisk|tight|max`, ms flags,
  `--min-between-cuts-ms`, `--snap`, `--audio-channels`.
- Deprecated `--min-silence`, `--min-talk`, `--pad-before`,
  `--pad-after` still parse, marked hidden, warning printed.

**Documentation**

- `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `docs/SCHEMAS.md`, `docs/DECISIONS.md`, `docs/BUILD.md`,
  `docs/PROJECT_STATE.md`.

### Changed

- `svaraflow-core`: `CoreError` moved to `error.rs` and extended.
- `svaraflow-core`: module layout split into `error.rs`, `timeline.rs`,
  `silence.rs`, `cut.rs`, `xmeml.rs`, `edl.rs`, `vad.rs` and
  `audio/{mod,decode,mixdown,envelope}.rs`.
- `Vad::speech_probabilities` takes `&mut self` (ADR-0019).
- Silence parameter units moved from seconds to milliseconds at the
  boundary, with exact conversion internally (ADR-0020).
- `svaraflow-cli/src/silence_cmd.rs`: `run_analysis` builds the analysis
  signal once and reuses it for both the envelope and the VAD.
- `svaraflow-cli/src/timeline_cmd.rs`: `Probe` now carries a
  `disposition` block, per-stream `codec_name`, `index` and `duration`;
  `first_video` filters attached cover art; `duration_seconds` falls back
  from `format.duration` to the longest stream `duration`.
- `docs/SCHEMAS.md` documents the transcript JSON, the model manifest,
  and the cache file.

### Fixed

- `tests/audio/fixtures/README.md` was inadvertently ignored by the
  `/tests/audio/*` rule in `.gitignore`.
- **XML cross-track linking was a stub.** `<link>` blocks are now written
  for every clip in a group that shares `(source, start)` and contains
  both video and audio (ADR-0017).
- **Silero VAD (spec D) was not implemented.** `Method::Vad` and
  `Method::Hybrid` now run a real VAD through the CLI.
- **VAD-mask auto noise (spec C7 first clause) was not implemented.**
  Added `auto_noise_level_from_vad`.
- **Track index uniqueness was checked globally** instead of per-kind.
- **`encode_path_url` produced uppercase hex** while the spec writes
  lowercase.
- **`resample_to` produced fewer samples than expected at 48k → 16k.**
- **Silence ranges did not cover the full timeline** when the envelope's
  last frame started before the audio ended.
- **`--noise -42.5` was rejected by clap** as an unknown flag.
- **`ort-sys` build failure with `download-binaries`.** Switched to
  `default-features = false` with `load-dynamic` (ADR-0018).
- **`svaraflow-cli` did not declare a `vad` feature.**
- **`ort` call sites were written against the wrong rc API.**
- **`svaraflow-cli timeline from-media` failed on audio-only files.**
- **`timeline from-media` mis-classified attached cover art as video.**
- **XML was rejected by Premiere and Resolve.** The DOCTYPE referenced a
  DTD that does not exist, the `\\?\` verbatim prefix leaked into every
  `<pathurl>`, every clipitem redefined its source instead of referencing
  it, and the sequence lacked the `<uuid>`, `<timecode>` and empty video
  `<track/>` the hosts expect (ADR-0023).
- **Imported cuts played 3 dB down.** The XML declared mono for stereo
  sources, so both hosts applied a constant-power pan law. The channel
  count, sample rate and bit depth now come from `ffprobe` on each source
  (ADR-0022).

### Security

- Nothing yet.

## [0.1.0] - Unreleased

First planned release. Scope: silence removal and captions with SRT/VTT
import, style presets and translation, for Adobe Premiere Pro and DaVinci
Resolve (Free and Studio).

[Unreleased]: https://github.com/Th3-A6add0n/svaraflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Th3-A6add0n/svaraflow/releases/tag/v0.1.0