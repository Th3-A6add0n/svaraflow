# SvaraFlow

SvaraFlow is an open-source (MIT) tool by **Th3 A6add0n** that adds two
features to Adobe Premiere Pro and DaVinci Resolve (Free and Studio):
**silence removal** and **word-by-word animated captions for Hindi–English
(Hinglish) speech written in Roman script**. All AI runs locally and
offline. Speech-to-text is `whisper.cpp` with two model tiers that write
Roman Hinglish directly; speech activity is detected with Silero VAD. Edits
are non-destructive: the original timeline is never modified. A new FCP7 XML
(`xmeml`) timeline is generated from a `timeline.json` and imported by the
host. A single Tauri 2 + React + TypeScript desktop app is shared across
hosts, with thin adapters for DaVinci Resolve (Python bridge scripts) and
Premiere Pro (UXP TypeScript plugin).

**Status:** early development. APIs, file formats, and preset names are
unstable and may change without notice.

## Repository layout

| Path | Purpose |
| --- | --- |
| `svaraflow-core/` | Audio I/O, silence detection, timeline model, cut engine, FCP7 XML export. |
| `svaraflow-asr/` | Speech-to-text (whisper.cpp), Silero VAD, model manifest, cache, subtitle exports. |
| `svaraflow-text/` | SRT/VTT parsing, dictionary, chunking, styles, translation. (Scaffold.) |
| `svaraflow-render/` | Caption renderer. (Scaffold.) |
| `svaraflow-rpc/` | JSON-RPC 2.0 server on `127.0.0.1` with a one-time token. (Scaffold.) |
| `svaraflow-cli/` | Command-line front end. |
| `app/` | Tauri 2 + React + TypeScript desktop application. |
| `hosts/resolve/` | DaVinci Resolve Python bridge scripts. |
| `hosts/premiere/` | Adobe Premiere Pro UXP plugin (TypeScript). |
| `tools/` | `check-env.ps1`, `convert-model.ps1`. |
| `docs/` | `SCHEMAS.md`, `DECISIONS.md`, `BUILD.md`, `PROJECT_STATE.md`. |

## Version 1 scope

- Silence removal.
- Word-by-word captions in Roman Hinglish, with SRT/VTT import, style
  presets, and translation.

**Not in v1:** J-cut/L-cut transitions, emoji, per-speaker styles,
Devanagari output.

## The CLI

Everything is driven by `svaraflow-cli`. The three subcommand trees are
`timeline`, `silence`, and `transcribe`; a fourth, `models`, manages model
files.

### Build a timeline from a media file

```powershell
svaraflow-cli timeline from-media interview.mov --out tl.json
```

The command probes the file with `ffprobe`. Video sources get one video
track and, when they carry audio, one linked audio track. Audio-only
sources get one audio track and no video track; the frame rate comes from
`--fps` (default `30000/1001`) and the raster size from `--width` and
`--height` (default `1920x1080`), because the timeline model requires
non-zero dimensions even without picture.

### Remove silence

```powershell
# See what would be cut.
svaraflow-cli silence analyze tl.json --preset natural --json

# Apply the cut and write FCP7 XML and a CMX3600 EDL.
svaraflow-cli silence apply tl.json --preset natural `
    --xml out.xml --edl out.edl --report report.json
```

Five built-in presets, from gentlest to most aggressive:

| Preset | remove (ms) | keep (ms) | margin (ms) | min between cuts |
| --- | --- | --- | --- | --- |
| `relaxed` | 500 | 250 | 150 | off |
| `natural` | 300 | 150 | 100 | off |
| `brisk` | 180 | 100 | 80 | off |
| `tight` | 100 | 70 | 60 | off |
| `max` | 50 | 50 | 50 | off |

Explicit flags (`--remove-silences-ms`, `--keep-talks-ms`,
`--margin-before-ms`, `--margin-after-ms`, `--min-between-cuts-ms`)
override the preset. `--snap inward|nearest` controls frame-grid snapping;
`max` uses `nearest` by default, the others use `inward`.

User presets are read from
`%APPDATA%\SvaraFlow\presets\silence.json` on Windows and
`$XDG_CONFIG_HOME/svaraflow/presets/silence.json` on Linux. Manage them
with `silence preset save|rename|delete|list|import|export`. The table is
`silence presets`.

### Transcribe speech

```powershell
# Auto-pick the tier, write SRT and the transcript JSON.
svaraflow-cli transcribe interview.mov --tier auto `
    --srt out.srt --json --out-json out.json
```

Model tiers:

- **Light** — `Whisper-Hindi2Hinglish-Swift`, whisper-base size. Fast,
  lower accuracy.
- **Standard** — `Whisper-Hindi2Hinglish-Apex`, large-v3-turbo based,
  GGML Q5_0, ~547 MB. Slower, higher accuracy.

With `--tier auto`, the CLI runs the Standard model over a 20-second
benchmark clip (`tests/audio/benchmark20s.wav`, not committed) and picks
Standard when it achieves 2x real time or faster. If the clip is missing,
it falls back to a CPU-core and NVIDIA-GPU probe.

Exports: `--srt`, `--vtt`, `--ass` (with `\kf` karaoke word highlighting),
`--json` (to stdout) or `--out-json <path>`.

`transcribe` prints the real-time factor on every run: audio seconds
divided by engine wall-clock seconds. Above `1.0` is faster than real
time.

### Model files

Models live outside the repository:

- Windows: `%LOCALAPPDATA%\SvaraFlow\models`
- Linux: `$XDG_DATA_HOME/svaraflow/models`, or
  `~/.local/share/svaraflow/models`

Manage them with:

```powershell
svaraflow-cli models list
svaraflow-cli models download Whisper-Hindi2Hinglish-Apex
svaraflow-cli models hash .\ggml-hindi2hinglish-swift.bin
svaraflow-cli models remove light
```

`models.json` in the repository root pins each model's name, URL, revision,
size, SHA-256 and license. The `url` and `sha256` fields ship as the
literal `"TODO"` until the maintainer fills them in; the downloader treats
`"TODO"` as "skip verification", which lets the pipeline run before the
hashes are known. Downloads resume from `.part` files and are verified
against the manifest when a real hash is present.

To convert your own Hugging Face checkpoint:

```powershell
pwsh tools/convert-model.ps1 `
    -RepoId "Oriserve/Whisper-Hindi2Hinglish-Swift" `
    -OutputName "ggml-hindi2hinglish-swift" `
    -Quantize f16
```

The script normalises bfloat16 tensors to float32 before running
whisper.cpp's own converter, quantizes with whisper.cpp's `quantize`
binary, and prints the resulting SHA-256 and size for pasting into
`models.json`.

### Silero VAD and ONNX Runtime

The `--method vad` and `--method hybrid` flags on `silence`, and the
chunking step inside `transcribe`, use Silero VAD through ONNX Runtime.
The runtime is loaded dynamically at process start from `ORT_DYLIB_PATH`
if set, then from the platform loader path. `docs/BUILD.md` has the
details and the download link.

## Requirements

- Windows 11 (primary dev target) or Linux.
- Rust (stable). The workspace declares `rust-version = "1.75"`.
- `cmake` and `clang` for transitive native builds.
- Node.js for `app/` and `hosts/premiere/`.
- Python 3.10+ for `hosts/resolve/` and `tools/convert-model.ps1`.
- FFmpeg on `PATH`. Used as a decode fallback, and required by
  `timeline from-media` for `ffprobe`.
- For the VAD path: ONNX Runtime (see `docs/BUILD.md`).
- For the `cuda` feature on `svaraflow-asr`: a CUDA toolkit matching your
  NVIDIA driver.

Run `pwsh tools/check-env.ps1` to verify the toolchain.

## Building and testing

```powershell
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

CPU-only build with no ONNX Runtime:

```powershell
cargo build --workspace --no-default-features
```

The `transcribe` and `silence --method vad|hybrid` paths then return a
clear error. The threshold path and every other subcommand are unaffected.

CUDA build:

```powershell
cargo build --workspace --features svaraflow-asr/cuda --release
```

The performance gate for the silence pipeline is `#[ignore]`d by default:

```powershell
cargo test --release -p svaraflow-core --test perf -- --ignored
```

## Contributing

See `CONTRIBUTING.md`. Security issues: `SECURITY.md`. Community
standards: `CODE_OF_CONDUCT.md`.

## License

MIT — see `LICENSE`.