<!-- tools/README.md -->
# SvaraFlow developer tools

## `convert-model.ps1`

Converts an Oriserve Whisper-Hindi2Hinglish checkpoint from Hugging Face into
a quantized GGML file usable by `whisper.cpp` and by SvaraFlow's
`svaraflow-cli transcribe`.

The script manages its own Python venv under `tools/.venv`, so it does not
touch the system Python or any other project's dependencies.

### Prerequisites

| Tool | Where to get it | Notes |
| --- | --- | --- |
| Python 3.10+ | <https://python.org> | Any version that `python -m venv` supports. |
| git | <https://git-scm.com> | Used once to clone `openai/whisper` if missing. |
| CMake 3.18+ | <https://cmake.org> | Used to build `whisper-quantize` if missing. |
| A built `whisper.cpp` | <https://github.com/ggml-org/whisper.cpp> | Build with `cmake -B build; cmake --build build --config Release`. |
| About 8 GB free | — | On the drive that holds `-OutDir`. |

### Usage

The script has one required parameter, `-Model`, and sensible defaults for
everything else.

**Swift tier, Q5_0, default paths** — the smallest and fastest model:

```powershell
pwsh tools/convert-model.ps1 -Model Swift -Quant q5_0
```

**Apex tier, Q5_0** — larger, slower, more accurate, matches the Standard tier:

```powershell
pwsh tools/convert-model.ps1 -Model Apex -Quant q5_0
```

**Apex tier, Q8_0** — the largest output, best accuracy, ~600 MB:

```powershell
pwsh tools/convert-model.ps1 -Model Apex -Quant q8_0
```

**Fully explicit, with paths that contain spaces:**

```powershell
pwsh tools/convert-model.ps1 `
    -Model Apex `
    -Quant q5_0 `
    -WhisperCppDir 'C:\Users\Alice\Desktop\My Projects\whisper.cpp' `
    -OpenAIWhisperDir 'C:\Users\Alice\Desktop\My Projects\whisper' `
    -OutDir 'C:\Users\Alice\AppData\Local\SvaraFlow\models'
```

### Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `-Model` | required | `Swift` or `Apex`. |
| `-Quant` | `q5_0` | `none`, `q5_0` or `q8_0`. `none` leaves the f16 output as-is. |
| `-WhisperCppDir` | `$env:SVARAFLOW_WHISPERCPP` | A built whisper.cpp checkout. If neither the parameter nor the env var is set, the script prompts. |
| `-OpenAIWhisperDir` | sibling of `-WhisperCppDir` named `whisper` | The converter reads `whisper/assets/mel_filters.npz` and tokenizer tables from here. Cloned on demand. |
| `-OutDir` | `%LOCALAPPDATA%\SvaraFlow\models` | Where the final GGML file lands. |
| `-Python` | `python` | Interpreter used to create the venv. |
| `-SkipSanityTest` | off | Do not run the `whisper-cli` transcription even if the binary is present. |

### Setting the default whisper.cpp path once

```powershell
# Persist for the current user; new shells pick it up.
[Environment]::SetEnvironmentVariable(
    'SVARAFLOW_WHISPERCPP',
    'C:\Users\Alice\Desktop\My Projects\whisper.cpp',
    'User'
)
```

After that, `-WhisperCppDir` can be omitted.

### Output

Two files land in `-OutDir`:

- `ggml-hinglish-swift-<quant>.bin`
- `ggml-hinglish-apex-<quant>.bin`

The script prints the file size, its SHA-256, the resolved Hugging Face
commit hash, and a ready-to-paste `models.json` entry. Paste the entry into
`models.json` at the repository root and rebuild the CLI so the embedded
manifest picks it up:

```powershell
cargo build --release -p svaraflow-cli
```

### Sanity test

If `whisper-cli` is built under the whisper.cpp checkout, the script
transcribes `tests/audio/benchmark20s.wav` with the freshly converted model
and prints the first 200 characters. That output should be Roman Hinglish.
If it is Devanagari, the wrong checkpoint was converted or the model is
misconfigured.

### Troubleshooting

**`No CUDA toolset found` during the CMake build.** Visual Studio BuildTools
does not get the CUDA MSBuild integration from the CUDA installer. Copy
`CUDA 13.4.props`, `CUDA 13.4.targets`, `CUDA 13.4.xml` and
`Nvda.Build.CudaTasks.v13.4.dll` from
`%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions` into
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations`.

**`FileNotFoundError: whisper/assets/mel_filters.npz`.** The converter needs
an `openai/whisper` clone as its second positional argument. The script
clones one automatically to a sibling of `-WhisperCppDir`. If you passed an
existing directory via `-OpenAIWhisperDir`, make sure it is a full clone,
not an empty directory.

**`ModuleNotFoundError: torch`.** The venv is under `tools/.venv`. The
script installs torch CPU from the PyTorch CPU wheel index. If the install
failed, delete `tools/.venv` and re-run.

**Conversion succeeds but sanity test prints Devanagari.** The checkpoint
was converted from the wrong revision. The script pins the revision in the
`$models` table; confirm the hash matches the model you expect.

## `check-env.ps1`

Probes the local machine for the tools the engine and this script need.
Prints each tool in green (present) or red (missing). Run it before opening
an issue about a build failure.