# Changelog

All notable changes to SvaraFlow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cargo workspace with six crates: `svaraflow-core`, `svaraflow-asr`,
  `svaraflow-text`, `svaraflow-render`, `svaraflow-rpc`, `svaraflow-cli`.
- `svaraflow-cli --version` flag.
- CI workflow (`.github/workflows/ci.yml`) running `cargo fmt --check`,
  `cargo clippy -D warnings`, `cargo test` on `ubuntu-latest` and
  `windows-latest`, and `cargo deny check`.
- `deny.toml` rejecting GPL and AGPL dependencies.
- Project meta documents: `README.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `docs/PROJECT_STATE.md`, `docs/DECISIONS.md`.

### Changed

- Nothing yet.

### Deprecated

- Nothing yet.

### Removed

- Nothing yet.

### Fixed

- Nothing yet.

### Security

- Nothing yet.

## [0.1.0] - Unreleased

First planned release. Scope: silence removal and captions with SRT/VTT import,
style presets and translation, for Adobe Premiere Pro and DaVinci Resolve (Free
and Studio).

[Unreleased]: https://github.com/Th3-A6add0n/svaraflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Th3-A6add0n/svaraflow/releases/tag/v0.1.0