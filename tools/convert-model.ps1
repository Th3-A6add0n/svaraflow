<#
.SYNOPSIS
    Converts a Hugging Face Whisper checkpoint to GGML and optionally
    quantizes it, using whisper.cpp's own conversion scripts.

.DESCRIPTION
    Wraps `whisper.cpp/models/convert-h5-to-ggml.py` and the `quantize`
    binary that whisper.cpp builds. Handles the common failure mode where a
    HF checkpoint stores weights as bfloat16 and whisper.cpp's converter
    calls `.numpy()` on them: the script first re-saves every tensor as
    float32 in a scratch copy, then runs the converter against that copy.

    Also prints the SHA-256 and byte size of the output, ready to paste
    into `models.json` and `models.json`'s `size_bytes`.

.PARAMETER RepoId
    Hugging Face repo, e.g. `Oriserve/Whisper-Hindi2Hinglish-Swift`. The
    script downloads it with `huggingface-cli download`. Mutually exclusive
    with `-ModelDir`.

.PARAMETER ModelDir
    A local directory that already contains a downloaded HF checkpoint
    (`pytorch_model.bin` or `model.safetensors` plus `config.json` and
    tokenizer files). Mutually exclusive with `-RepoId`.

.PARAMETER OutputName
    Base name for the output GGML file, without the `.bin` extension. The
    script appends `.bin` and, when quantizing, `-<quant>.bin`.

.PARAMETER Quantize
    Quantization to apply. `f16` is no quantization. Default `q5_0` matches
    the Standard tier.

.PARAMETER WhisperCppPath
    Path to a whisper.cpp checkout that has been built. The script uses
    `<path>/models/convert-h5-to-ggml.py` and `<path>/build/bin/quantize`
    (`quantize.exe` on Windows).

.PARAMETER WorkDir
    Scratch directory for the float32-converted checkpoint. Created if
    missing.

.EXAMPLE
    # Standard tier: whisper.cpp already ships a converter for large-v3-turbo
    # derivatives.
    pwsh tools/convert-model.ps1 `
        -RepoId "Oriserve/Whisper-Hindi2Hinglish-Swift" `
        -OutputName "ggml-hindi2hinglish-swift" `
        -Quantize f16

.EXAMPLE
    # If you already downloaded the checkpoint:
    pwsh tools/convert-model.ps1 `
        -ModelDir "C:\hf\Whisper-Hindi2Hinglish-Swift" `
        -OutputName "ggml-hindi2hinglish-swift" `
        -Quantize q5_0
#>
[CmdletBinding(DefaultParameterSetName = 'Repo')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Repo')]
    [string] $RepoId,

    [Parameter(Mandatory, ParameterSetName = 'Local')]
    [string] $ModelDir,

    [Parameter(Mandatory)]
    [string] $OutputName,

    [ValidateSet('f16', 'f32', 'q8_0', 'q5_0', 'q5_1', 'q4_0', 'q4_1')]
    [string] $Quantize = 'q5_0',

    [string] $WhisperCppPath = '..\whisper.cpp',

    [string] $WorkDir = '.\tools\.cache\convert'
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

function Write-Step {
    param([string] $Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string] $Message)
    Write-Host "    $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string] $Message)
    Write-Host "    $Message" -ForegroundColor Yellow
}

function Invoke-Native {
    param(
        [string] $Command,
        [string[]] $Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "command failed with exit $LASTEXITCODE`: $Command $($Arguments -join ' ')"
    }
}

function Require-Command {
    param([string] $Name)
    if (-not (Get-Command -Name $Name -ErrorAction SilentlyContinue)) {
        throw "`$Name` was not found on PATH. Install it and re-run."
    }
}

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

Write-Step 'checking prerequisites'

$whisperCppRoot = (Resolve-Path -LiteralPath $WhisperCppPath -ErrorAction SilentlyContinue).Path
if (-not $whisperCppRoot) {
    throw "whisper.cpp path does not exist: $WhisperCppPath"
}
$converter = Join-Path $whisperCppRoot 'models\convert-h5-to-ggml.py'
if (-not (Test-Path -LiteralPath $converter)) {
    throw "converter not found at $converter; is this a whisper.cpp checkout?"
}

$quantizeExe = Join-Path $whisperCppRoot 'build\bin\quantize.exe'
if (-not (Test-Path -LiteralPath $quantizeExe)) {
    $quantizeExe = Join-Path $whisperCppRoot 'build\bin\quantize'
}
if (-not (Test-Path -LiteralPath $quantizeExe)) {
    throw "quantize binary not found under $whisperCppRoot\build\bin; build whisper.cpp first (cmake --build build --config Release)"
}

Require-Command -Name 'python'
Write-Ok "whisper.cpp at $whisperCppRoot"
Write-Ok "converter at $converter"
Write-Ok "quantize at $quantizeExe"

# ---------------------------------------------------------------------------
# 1. obtain the HF checkpoint
# ---------------------------------------------------------------------------

$WorkDir = (New-Item -ItemType Directory -Path $WorkDir -Force).FullName

if ($PSCmdlet.ParameterSetName -eq 'Repo') {
    Write-Step "downloading $RepoId with huggingface-cli"

    $haveCli = $null -ne (Get-Command 'huggingface-cli' -ErrorAction SilentlyContinue)
    if (-not $haveCli) {
        throw @"
`huggingface-cli` is not on PATH. Install it with:

    python -m pip install --user huggingface_hub[cli]

or download the repository by hand and pass -ModelDir instead.
"@
    }

    $localRepo = Join-Path $WorkDir ($RepoId -replace '[\\/:]', '_')
    if (Test-Path -LiteralPath $localRepo) {
        Write-Warn "re-using existing download at $localRepo"
    } else {
        Invoke-Native -Command 'huggingface-cli' -Arguments @(
            'download', $RepoId,
            '--local-dir', $localRepo,
            '--local-dir-use-symlinks', 'False'
        )
    }
    $sourceDir = $localRepo
} else {
    $sourceDir = (Resolve-Path -LiteralPath $ModelDir).Path
    Write-Step "using local checkpoint at $sourceDir"
}

if (-not (Test-Path -LiteralPath (Join-Path $sourceDir 'config.json'))) {
    throw "no config.json under $sourceDir; this does not look like a Hugging Face Whisper checkpoint"
}

# ---------------------------------------------------------------------------
# 2. re-save every tensor as float32
# ---------------------------------------------------------------------------
#
# whisper.cpp's converter calls `.numpy()` on torch tensors. A bfloat16
# tensor has no NumPy equivalent, so the call raises. Rather than patch the
# converter, load the checkpoint here, cast every tensor, and write a
# scratch copy the converter can read without modification.

Write-Step 'normalising tensor dtype to float32'

$floatDir = Join-Path $WorkDir "$OutputName-float32"
if (Test-Path -LiteralPath $floatDir) {
    Remove-Item -LiteralPath $floatDir -Recurse -Force
}
New-Item -ItemType Directory -Path $floatDir | Out-Null

# Copy the non-weight files (config.json, tokenizer, vocab) as-is.
Get-ChildItem -LiteralPath $sourceDir -File |
    Where-Object { $_.Name -notin @('pytorch_model.bin', 'model.safetensors', 'model.safetensors.index.json') } |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $floatDir }

$patchPy = Join-Path $WorkDir '_svaraflow_float32.py'
@'
import json
import os
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]

state = None

safetensors_path = os.path.join(src, "model.safetensors")
bin_path = os.path.join(src, "pytorch_model.bin")

if os.path.isfile(safetensors_path):
    try:
        from safetensors.torch import load_file as st_load
    except ImportError:
        print("safetensors is required to read model.safetensors", file=sys.stderr)
        sys.exit(2)
    state = st_load(safetensors_path)
    print(f"loaded {len(state)} tensors from {safetensors_path}")
elif os.path.isfile(bin_path):
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    print(f"loaded {len(state)} tensors from {bin_path}")
else:
    print(f"no model.safetensors or pytorch_model.bin under {src}", file=sys.stderr)
    sys.exit(3)

converted = 0
for name, tensor in state.items():
    if tensor.dtype == torch.bfloat16:
        state[name] = tensor.to(torch.float32)
        converted += 1

out = os.path.join(dst, "pytorch_model.bin")
torch.save(state, out)
print(f"wrote {out} ({converted} tensors cast from bfloat16 to float32)")
'@ | Set-Content -LiteralPath $patchPy -Encoding UTF8

Invoke-Native -Command 'python' -Arguments @($patchPy, $sourceDir, $floatDir)

# ---------------------------------------------------------------------------
# 3. run whisper.cpp's converter
# ---------------------------------------------------------------------------

Write-Step 'converting to GGML'

$convertedName = "$OutputName-f16.bin"
$convertedPath = Join-Path $WorkDir $convertedName

Push-Location $whisperCppRoot
try {
    Invoke-Native -Command 'python' -Arguments @(
        $converter, $floatDir, $OutputName
    )
} finally {
    Pop-Location
}

# convert-h5-to-ggml.py writes `ggml-<name>.bin` into the current directory
# (whisper.cpp root) unless it was already named that way. Find whichever it
# produced and move it into our work dir under a predictable name.
$generated = Get-ChildItem -LiteralPath $whisperCppRoot -Filter "ggml-*.bin" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $generated) {
    throw "converter did not produce a ggml-*.bin under $whisperCppRoot"
}
Move-Item -LiteralPath $generated.FullName -Destination $convertedPath -Force
Write-Ok "wrote $convertedPath"

# ---------------------------------------------------------------------------
# 4. quantize if requested
# ---------------------------------------------------------------------------

if ($Quantize -eq 'f16' -or $Quantize -eq 'f32') {
    $finalPath = $convertedPath
    Write-Warn "no quantization requested; the f16 file is the output"
} else {
    Write-Step "quantizing to $Quantize"
    $finalName = "$OutputName-$Quantize.bin"
    $finalPath = Join-Path $WorkDir $finalName
    Invoke-Native -Command $quantizeExe -Arguments @(
        $convertedPath, $finalPath, $Quantize
    )
    Write-Ok "wrote $finalPath"
}

# ---------------------------------------------------------------------------
# 5. hash and report
# ---------------------------------------------------------------------------

Write-Step 'computing SHA-256'

$hash = (Get-FileHash -LiteralPath $finalPath -Algorithm SHA256).Hash.ToLowerInvariant()
$size = (Get-Item -LiteralPath $finalPath).Length

Write-Host ''
Write-Host 'Paste these into models.json:' -ForegroundColor Cyan
Write-Host ''
Write-Host ("  filename:    {0}" -f (Split-Path -Leaf $finalPath))
Write-Host ("  size_bytes:  {0}" -f $size)
Write-Host ("  sha256:      {0}" -f $hash)
Write-Host ''
Write-Host 'Then place the file at:' -ForegroundColor Cyan
Write-Host ("  %LOCALAPPDATA%\SvaraFlow\models\{0}" -f (Split-Path -Leaf $finalPath))
Write-Host ''
Write-Host 'Done.' -ForegroundColor Green