# SvaraFlow

SvaraFlow is an open-source (MIT) tool by **Th3 A6add0n** that adds two features to Adobe Premiere Pro and DaVinci Resolve (Free and Studio): **silence removal** and **word-by-word animated captions for Hindi–English (Hinglish) speech written in Roman script**. All AI runs locally and offline, using small models — speech-to-text is provided by `whisper.cpp` through Rust bindings (Light: *Whisper-Hindi2Hinglish-Swift*; Standard: *Whisper-Hindi2Hinglish-Apex Q5_0*), and speech activity is detected with Silero VAD. Edits are non-destructive: the original timeline is never modified — instead, a new FCP7 XML (`xmeml`) timeline is generated from a `timeline.json` and imported by the host. A single Tauri 2 + React + TypeScript desktop app is shared across hosts, with thin adapters for DaVinci Resolve (Python bridge scripts) and Premiere Pro (UXP TypeScript plugin).

**Status:** early development. APIs, file formats, and preset names are unstable and may change without notice.

## Repository layout

| Path | Purpose |
| --- | --- |
| `svaraflow-core/` | Audio I/O, silence detection, timeline model, cut engine, FCP7 XML export. |
| `svaraflow-asr/` | Speech-to-text (whisper.cpp bindings), model selection, Silero VAD. |
| `svaraflow-text/` | SRT/VTT, dictionary, chunking, styles, translation. |
| `svaraflow-render/` | Caption renderer. |
| `svaraflow-rpc/` | JSON-RPC 2.0 server on `127.0.0.1` with a one-time token. |
| `svaraflow-cli/` | Command-line front end reusing the same crates. |
| `app/` | Tauri 2 + React + TypeScript desktop application. |
| `hosts/resolve/` | DaVinci Resolve Python bridge scripts (in-app for Free, external for Studio). |
| `hosts/premiere/` | Adobe Premiere Pro UXP plugin (TypeScript). |
| `tools/` | Developer scripts (environment checks, model fetchers, XML probes). |
| `tests/audio/` | Local test media (not committed). |
| `docs/` | `PROJECT_STATE.md`, `DECISIONS.md`, and other living docs. |

## Version 1 scope

- Silence removal.
- Captions with SRT/VTT import, style presets, and translation.

**Not in v1:** J-cut/L-cut transitions, emoji, per-speaker styles, Devanagari output.

## Requirements

- Windows 11 (primary dev target) or Linux.
- Rust (stable), `cmake`, `clang`.
- Node.js (for `app/` and `hosts/premiere/`).
- Python 3.10+ (for `hosts/resolve/`).
- FFmpeg on `PATH`.
- For GPU acceleration: an NVIDIA GPU with a matching CUDA toolkit (`nvidia-smi`, `nvcc`).

Run `pwsh tools/check-env.ps1` to verify the toolchain.

## Building and testing

```powershell
cargo build
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Contributing
See `CONTRIBUTING.md`. Security issues: see `SECURITY.md`. Community standards: `CODE_OF_CONDUCT.md`.

## License
MIT — see LICENSE.