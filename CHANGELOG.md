
# Changelog

All notable changes to SvaraFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cargo workspace with six crates: `svaraflow-core`, `svaraflow-asr`,
  `svaraflow-text`, `svaraflow-render`, `svaraflow-rpc`, `svaraflow-cli`.
- `svaraflow-cli --version`.
- CI workflow running fmt, clippy, test on ubuntu + windows, cargo-deny.
- `deny.toml` rejecting GPL and AGPL.
- Project meta documents.
- `tools/check-env.ps1`.
- `svaraflow-core::timeline`.
- `docs/SCHEMAS.md` — normative `timeline.json` v0.0.1.
- `svaraflow-core::audio` — analysis-signal pipeline.
- `svaraflow-core::audio::envelope` — dBFS envelope.
- `svaraflow-core::silence` — threshold detector, `detect_silence_with_vad`,
  `auto_noise_level_from_vad`.
- `svaraflow-core::cut` — four modes, non-destructive.
- `svaraflow-core::vad` — engine-agnostic VAD surface.
- `svaraflow-core::xmeml` — FCP7 XML v4 with cross-track `<link>` blocks.
- `svaraflow-core::edl` — CMX3600 EDL.
- `svaraflow-asr::vad` — Silero VAD v5 through ONNX Runtime.
- `svaraflow-cli` — `timeline from-media`, `silence analyze`,
  `silence apply` with the full spec flag set.
- **`svaraflow-cli/tests/common/mod.rs`** — WAV writer, JSON path escaper,
  temp-path helper. Unit tests cover the RIFF header length, silence and
  sine generators, and the JSON escaping rule.
- **`svaraflow-cli/tests/pipeline.rs`** — end-to-end tests that generate a
  synthetic WAV, write a `timeline.json`, and run the compiled binary as a
  subprocess:
  - `silence_analyze_json_reports_expected_shape` — asserts on the JSON
    shape and the `kept + removed == total` invariant.
  - `silence_analyze_human_output_reports_a_keep_range` — asserts on the
    human-readable table.
  - `silence_apply_remove_writes_xml_and_edl` — asserts on the XML DOCTYPE,
    sequence name, NTSC flag, pathurl and clipitem count, and on the EDL
    title, FCM header and event count.
  - `silence_apply_keep_preserves_timeline_length` — asserts the
    three-clipitem split and the unchanged 90-frame sequence duration.
  - `silence_apply_mute_marks_silence_clips_disabled` — asserts that some
    but not all clipitems emit `<enabled>FALSE</enabled>`.
  - `silence_apply_without_xml_or_edl_fails` — asserts the required-flag
    error.
  - `silence_analyze_missing_timeline_reports_the_path` — asserts the
    I/O error names the file.
- **`svaraflow-core/tests/perf.rs`** — performance gates.
  - `analysis_of_short_signal_is_fast` runs by default, 10 s signal, 2 s
    budget.
  - `one_hour_synthetic_timeline_analyzed_in_under_30s` is `#[ignore]`d
    and matches the spec's "1-hour synthetic timeline in 30 s or less"
    clause. Run with
    `cargo test --release -p svaraflow-core --test perf -- --ignored`.

### Changed

- `svaraflow-core`: `CoreError` moved to `error.rs` and extended.
- `svaraflow-core`: module layout split into `error.rs`, `timeline.rs`,
  `silence.rs`, `cut.rs`, `xmeml.rs`, `edl.rs`, `vad.rs` and
  `audio/{mod,decode,mixdown,envelope}.rs`.
- `svaraflow-core::xmeml`: `LinkIndex` built once per `write_xmeml`.
- `svaraflow-cli/src/silence_cmd.rs`: `run_analysis` builds the analysis
  signal once and reuses it for both the envelope and the VAD.

### Fixed

- `tests/audio/fixtures/README.md` was inadvertently ignored by the
  `/tests/audio/*` rule in `.gitignore`. Fixed in batch 4.
- **XML cross-track linking was a stub** in batch 7. Fixed in reply 1 of
  the gap-fix workstream. Golden fixture regenerated.
- **Silero VAD (spec D) was not implemented.** Fixed in reply 3.
  `Method::Vad` and `Method::Hybrid` now run a real VAD through the CLI.
- **VAD-mask auto noise (spec C7 first clause) was not implemented.**
  Fixed in reply 2 via `auto_noise_level_from_vad`.

### Security

- Nothing yet.

## [0.1.0] - Unreleased

First planned release. Scope: silence removal and captions with SRT/VTT import,
style presets and translation, for Adobe Premiere Pro and DaVinci Resolve (Free
and Studio).

[Unreleased]: https://github.com/Th3-A6add0n/svaraflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Th3-A6add0n/svaraflow/releases/tag/v0.1.0