# Changelog

All notable changes to SvaraFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cargo workspace with six crates.
- `svaraflow-cli --version`.
- CI workflow.
- `deny.toml`.
- Project meta documents.
- `tools/check-env.ps1`.
- `svaraflow-core`: timeline model.
- `docs/SCHEMAS.md`.
- `svaraflow-core::audio`: analysis-signal pipeline.
- `svaraflow-core::audio::envelope`: dBFS envelope.
- `svaraflow-core::silence`: threshold detector; `detect_silence_with_vad`;
  `auto_noise_level_from_vad`.
- `svaraflow-core::cut`: `apply_cuts` with `CutMode::{Remove, Keep, Mute,
  Gaps}`.
- `svaraflow-core::vad`: engine-agnostic VAD surface.
- `svaraflow-core::xmeml`: FCP7 XML v4 with cross-track `<link>` blocks.
- `svaraflow-core::edl`: CMX3600 EDL.
- `svaraflow-asr::vad`: **Silero VAD v5 through ONNX Runtime.**
  - `SileroVad::from_file(path)` loads the standard v5 export and
    discovers input/output names from the session, so both the v5
    (`state`/`stateN`) and v4 (`h`/`hn`) graphs load.
  - 512-sample windows, 16 kHz, LSTM state fed back between windows, final
    partial window zero-padded.
  - Implements `svaraflow_core::vad::Vad`.
  - Feature-gated behind `vad` (default on). `--no-default-features`
    compiles the crate without ONNX Runtime; VAD calls then return
    `AsrError::VadFeatureDisabled`.
  - `ort = "2.0.0-rc.9"` with `download-binaries` (prebuilt runtime fetched
    at build time). Set `ORT_STRATEGY=system` to link a system runtime.
  - 6 unit tests for name discovery, state shape, and the version helper.
- `svaraflow-cli`: `--vad-model <path>` flag on `silence analyze` and
  `silence apply`. Defaults to `models/silero_vad_v5.onnx` relative to the
  CWD. Used only when `--method vad` or `--method hybrid`.

### Changed

- `svaraflow-cli/src/silence_cmd.rs`: `run_analysis` now builds the
  analysis signal once and reuses it for both the envelope and the VAD, so
  decode runs once per invocation. `load_silero_vad` is a new helper.

### Fixed

- **Silero VAD (spec section D) is now implemented.** `Method::Vad` and
  `Method::Hybrid` on the CLI actually run a VAD instead of returning
  `NotImplemented` from `svaraflow-core`.

### Security

- Nothing yet.

## [0.1.0] - Unreleased

First planned release. Scope: silence removal and captions with SRT/VTT import,
style presets and translation, for Adobe Premiere Pro and DaVinci Resolve (Free
and Studio).

[Unreleased]: https://github.com/Th3-A6add0n/svaraflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Th3-A6add0n/svaraflow/releases/tag/v0.1.0