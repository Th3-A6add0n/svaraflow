# tools/convert-model.ps1
<#
.SYNOPSIS
    Converts an Oriserve Whisper-Hindi2Hinglish checkpoint to GGML and
    optionally quantizes it.

.DESCRIPTION
    End-to-end conversion of either Oriserve model to a whisper.cpp-compatible
    GGML file. See tools/README.md for full usage.

    This file is delivered in full on each change; do not diff it against a
    previous version, replace it entirely.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('Swift', 'Apex')]
    [string] $Model,

    [ValidateSet('none', 'q5_0', 'q8_0')]
    [string] $Quant = 'q5_0',

    [string] $WhisperCppDir,

    [string] $OpenAIWhisperDir,

    [string] $OutDir,

    [string] $Python = 'python',

    [switch] $SkipSanityTest
)

$ErrorActionPreference = 'Stop'

$models = @{
    Swift = @{
        RepoId      = 'Oriserve/Whisper-Hindi2Hinglish-Swift'
        RepoCommit  = '28c045e53a283e9ae3332de21050bf8155e8ff2b'
        FilePrefix  = 'ggml-hinglish-swift'
        License     = 'Apache-2.0'
        Description = 'Whisper-base-sized Hindi-to-Hinglish model. Fast, lower accuracy.'
        QuantForF16 = 'f16'
    }
    Apex = @{
        RepoId      = 'Oriserve/Whisper-Hindi2Hinglish-Apex'
        RepoCommit  = 'f3214eed20b4e4d4144e739982d911f87b9cb223'
        FilePrefix  = 'ggml-hinglish-apex'
        License     = 'Apache-2.0'
        Description = 'Large-v3-turbo-based Hindi-to-Hinglish model. Slower, higher accuracy.'
        QuantForF16 = 'q5_0'
    }
}

$modelInfo = $models[$Model]

function Write-Step { param([string] $M) Write-Host "==> $M" -ForegroundColor Cyan }
function Write-Ok   { param([string] $M) Write-Host "    $M" -ForegroundColor Green }
function Write-Warn { param([string] $M) Write-Host "    $M" -ForegroundColor Yellow }
function Fail       { param([string] $M) throw $M }

function Require-Command {
    param([string] $Name, [string] $Hint)
    if (-not (Get-Command -Name $Name -ErrorAction SilentlyContinue)) {
        Fail "``$Name`` was not found on PATH.$Hint"
    }
}

function ConvertTo-PythonString {
    param([string] $Path)
    $normalized = $Path -replace '\\', '/'
    $escaped = $normalized -replace "'", "\\'"
    return "'$escaped'"
}

function Get-FreeBytes {
    param([string] $Path)
    $full = [System.IO.Path]::GetFullPath($Path)
    $root = [System.IO.Path]::GetPathRoot($full)
    $drive = (Get-Item -LiteralPath $root).PSDrive.Name
    $psDrive = Get-PSDrive -Name $drive
    return [int64]$psDrive.Free
}

function Find-WhisperBinary {
    param([string] $Root, [string[]] $Names)
    $dirs = @(
        'build\bin\Release',
        'build\bin\RelWithDebInfo',
        'build\bin',
        'build\Release',
        'build',
        'build\examples\Release',
        'build\examples'
    )
    foreach ($dir in $dirs) {
        foreach ($name in $Names) {
            foreach ($ext in @('.exe', '')) {
                $p = Join-Path (Join-Path $Root $dir) ($name + $ext)
                if (Test-Path -LiteralPath $p) { return $p }
            }
        }
    }
    $buildDir = Join-Path $Root 'build'
    if (Test-Path -LiteralPath $buildDir) {
        $found = @(Get-ChildItem -Path $buildDir -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.BaseName -in $Names -and
                ($_.Extension -eq '' -or $_.Extension -eq '.exe')
            } |
            Sort-Object LastWriteTime -Descending)
        if ($found.Count -gt 0) { return $found[0].FullName }
    }
    return $null
}

function Invoke-Native {
    param([string] $Command, [string[]] $Arguments)
    $display = "$Command $($Arguments -join ' ')"
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "command failed with exit $LASTEXITCODE`: $display"
    }
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

Write-Step "converting $Model to $Quant"
Write-Step 'checking prerequisites'

Require-Command -Name 'python' -Hint " Install Python 3.10+ from https://python.org and re-run."

if (-not $WhisperCppDir) {
    if ($env:SVARAFLOW_WHISPERCPP -and (Test-Path -LiteralPath $env:SVARAFLOW_WHISPERCPP)) {
        $WhisperCppDir = $env:SVARAFLOW_WHISPERCPP
        Write-Ok "whisper.cpp from SVARAFLOW_WHISPERCPP: $WhisperCppDir"
    } else {
        Write-Host ''
        Write-Host 'Enter the path to a built whisper.cpp checkout:' -ForegroundColor Yellow
        $WhisperCppDir = Read-Host -Prompt 'whisper.cpp path'
    }
}

$WhisperCppDir = $WhisperCppDir.TrimEnd('\', '/')

if (-not (Test-Path -LiteralPath $WhisperCppDir)) {
    Fail "whisper.cpp directory does not exist: $WhisperCppDir"
}
$whisperCppRoot = (Resolve-Path -LiteralPath $WhisperCppDir).Path

$converter = Join-Path $whisperCppRoot 'models\convert-h5-to-ggml.py'
if (-not (Test-Path -LiteralPath $converter)) {
    Fail "converter not found at $converter; is this a whisper.cpp checkout?"
}

if (-not $OutDir) {
    if (-not $env:LOCALAPPDATA) {
        Fail 'LOCALAPPDATA is not set and -OutDir was not provided.'
    }
    $OutDir = Join-Path (Join-Path $env:LOCALAPPDATA 'SvaraFlow') 'models'
}
if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

if (-not $OpenAIWhisperDir) {
    $parent = Split-Path -Parent $whisperCppRoot
    $OpenAIWhisperDir = Join-Path $parent 'whisper'
}

$freeBytes = Get-FreeBytes -Path $OutDir
$requiredBytes = 8GB
if ($freeBytes -lt $requiredBytes) {
    $freeGB = [math]::Round($freeBytes / 1GB, 2)
    Fail "not enough free space on the drive holding ${OutDir}: $freeGB GB free, need about 8 GB."
}
$freeGB = [math]::Round($freeBytes / 1GB, 2)
Write-Ok "free space: $freeGB GB on the output drive"

Write-Ok "model:        $Model ($($modelInfo.RepoId))"
Write-Ok "checkpoint:   $($modelInfo.RepoCommit)"
Write-Ok "whisper.cpp:  $whisperCppRoot"
Write-Ok "converter:    $converter"
Write-Ok "out dir:      $OutDir"
Write-Ok "openai/whisper: $OpenAIWhisperDir"

# ---------------------------------------------------------------------------
# 1. Python venv
# ---------------------------------------------------------------------------

Write-Step 'preparing Python venv'

$toolsDir = Split-Path -Parent $PSCommandPath
$venvDir = Join-Path $toolsDir '.venv'
$venvPython = Join-Path (Join-Path $venvDir 'Scripts') 'python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Warn "creating venv at $venvDir"
    Invoke-Native -Command $Python -Arguments @('-m', 'venv', $venvDir)
} else {
    Write-Ok "re-using venv at $venvDir"
}

& $venvPython --version
if ($LASTEXITCODE -ne 0) { Fail "venv Python failed: $venvPython" }

$packages = @(
    @{ Import = 'torch';           Pip = 'torch';           Extra = @('--index-url', 'https://download.pytorch.org/whl/cpu') },
    @{ Import = 'transformers';    Pip = 'transformers';    Extra = @() },
    @{ Import = 'numpy';           Pip = 'numpy';           Extra = @() },
    @{ Import = 'huggingface_hub'; Pip = 'huggingface_hub'; Extra = @() },
    @{ Import = 'safetensors';     Pip = 'safetensors';     Extra = @() }
)

foreach ($pkg in $packages) {
    & $venvPython -c "import $($pkg.Import)" 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Ok "$($pkg.Import) present"; continue }
    Write-Warn "installing $($pkg.Pip)"
    $installArgs = @('-m', 'pip', 'install', '--quiet', '--upgrade')
    $installArgs += $pkg.Extra
    $installArgs += $pkg.Pip
    Invoke-Native -Command $venvPython -Arguments $installArgs
    Write-Ok "$($pkg.Import) installed"
}

# ---------------------------------------------------------------------------
# 2. Download checkpoint
# ---------------------------------------------------------------------------

Write-Step 'downloading checkpoint from Hugging Face'

$workDir = Join-Path $toolsDir '.cache\convert'
if (-not (Test-Path -LiteralPath $workDir)) {
    New-Item -ItemType Directory -Path $workDir -Force | Out-Null
}
$workDir = (Resolve-Path -LiteralPath $workDir).Path

$checkpointDir = Join-Path $workDir ($modelInfo.RepoId -replace '[\\/:]', '_')
$downloadResultFile = Join-Path $workDir '_download_result.json'
if (Test-Path -LiteralPath $downloadResultFile) { Remove-Item -LiteralPath $downloadResultFile -Force }

$repoIdPy = ConvertTo-PythonString $modelInfo.RepoId
$checkpointDirPy = ConvertTo-PythonString $checkpointDir
$revisionPy = ConvertTo-PythonString $modelInfo.RepoCommit
$resultFilePy = ConvertTo-PythonString $downloadResultFile

$downloadSource = @"
import glob, json, os, sys
from huggingface_hub import snapshot_download

repo_id = $repoIdPy
revision = $revisionPy
local_dir = $checkpointDirPy
result_path = $resultFilePy

try:
    path = snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)
except Exception as exc:
    print(f"snapshot_download failed: {exc}", file=sys.stderr)
    sys.exit(2)

commit = ''
for pattern in (
    os.path.join(path, '.cache', 'huggingface', 'download', '**', '*.metadata'),
    os.path.join(path, '.cache', 'huggingface', '**', '*.metadata'),
):
    for hit in glob.glob(pattern, recursive=True):
        try:
            with open(hit, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
        except Exception:
            continue
        value = meta.get('commit_hash')
        if value:
            commit = value
            break
    if commit:
        break

with open(result_path, 'w', encoding='utf-8') as fh:
    json.dump({'path': path, 'commit': commit}, fh)

print(f"checkpoint at {path}")
print(f"commit hash: {commit or '(not resolved)'}")
"@
Set-Content -LiteralPath (Join-Path $workDir '_download.py') -Value $downloadSource -Encoding UTF8
$downloadPy = Join-Path $workDir '_download.py'

Invoke-Native -Command $venvPython -Arguments @($downloadPy)

if (-not (Test-Path -LiteralPath $downloadResultFile)) { Fail "download did not produce $downloadResultFile" }
$downloadResult = Get-Content -LiteralPath $downloadResultFile -Raw | ConvertFrom-Json
$checkpointPath = $downloadResult.path
$resolvedCommit = $downloadResult.commit
if (-not $resolvedCommit) {
    $resolvedCommit = $modelInfo.RepoCommit
    Write-Warn "could not resolve commit hash from cache; using pinned hash"
}
Write-Ok "checkpoint at $checkpointPath"
Write-Ok "commit hash: $resolvedCommit"

# ---------------------------------------------------------------------------
# 3. Normalise bfloat16 weights to float32 if present
# ---------------------------------------------------------------------------
#
# whisper.cpp's converter does `list_vars[src].squeeze().numpy()`. PyTorch's
# `.numpy()` refuses bfloat16 because NumPy has no equivalent dtype.
#
# The rewrite must handle four things that affect what `from_pretrained`
# actually loads:
#
#   1. The scratch directory must contain no `.safetensors` file. If it does,
#      `from_pretrained` prefers safetensors over `pytorch_model.bin` and
#      loads the original bfloat16 weights.
#   2. `config.json` must declare float32 in every field that `transformers`
#      might read. Older versions read `torch_dtype`; newer versions read
#      `dtype`. Both are set.
#   3. `config.json` must not carry `_name_or_path` pointing at the original
#      Hugging Face repo. Left unchanged, `from_pretrained` may resolve the
#      weights back to the hub and re-download the original bfloat16
#      safetensors. It is set to the scratch directory.
#   4. The scratch contents are printed so any stray file is visible.

Write-Step 'preparing checkpoint for the converter'

$normalisedDir = Join-Path $workDir "$($modelInfo.FilePrefix)-float32"
$normaliseResultFile = Join-Path $workDir '_normalise_result.json'
if (Test-Path -LiteralPath $normaliseResultFile) { Remove-Item -LiteralPath $normaliseResultFile -Force }

$checkpointPathPy = ConvertTo-PythonString $checkpointPath
$normalisedDirPy = ConvertTo-PythonString $normalisedDir
$normaliseResultPy = ConvertTo-PythonString $normaliseResultFile

$normaliseSource = @"
import json, os, shutil, sys
import torch

src = $checkpointPathPy
dst = $normalisedDirPy
result_path = $normaliseResultPy

state = None
weight_source = None

safetensors_path = os.path.join(src, 'model.safetensors')
bin_path = os.path.join(src, 'pytorch_model.bin')

if os.path.isfile(safetensors_path):
    try:
        from safetensors.torch import load_file as st_load
    except ImportError:
        print('safetensors is required to read model.safetensors', file=sys.stderr)
        sys.exit(3)
    state = st_load(safetensors_path)
    weight_source = safetensors_path
elif os.path.isfile(bin_path):
    state = torch.load(bin_path, map_location='cpu', weights_only=True)
    weight_source = bin_path
else:
    print(f'no model.safetensors or pytorch_model.bin under {src}', file=sys.stderr)
    sys.exit(4)

dtypes = sorted({str(t.dtype) for t in state.values()})
bf16_count = sum(1 for t in state.values() if t.dtype == torch.bfloat16)
print(f'weight source: {weight_source}')
print(f'dtypes: {", ".join(dtypes)}')
print(f'bfloat16 tensors: {bf16_count}')

if bf16_count == 0:
    with open(result_path, 'w', encoding='utf-8') as fh:
        json.dump({'dir': src, 'normalised': False, 'bf16_count': 0}, fh)
    print('checkpoint is already float32; using it as-is')
    sys.exit(0)

if os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst, exist_ok=True)

# Copy non-weight files. Anything named like a weights file is skipped, and
# any stray .safetensors in the result is removed after the copy.
weight_exact = {
    'model.safetensors',
    'pytorch_model.bin',
    'model.safetensors.index.json',
    'pytorch_model.bin.index.json',
}
for name in os.listdir(src):
    if name in weight_exact:
        continue
    if name.endswith('.safetensors') or name.endswith('.bin'):
        continue
    full = os.path.join(src, name)
    if os.path.isfile(full):
        shutil.copy2(full, os.path.join(dst, name))

# Belt and braces: drop any safetensors that slipped through.
for name in os.listdir(dst):
    if name.endswith('.safetensors'):
        os.remove(os.path.join(dst, name))

# Rewrite config.json.
config_path = os.path.join(dst, 'config.json')
if os.path.isfile(config_path):
    with open(config_path, 'r', encoding='utf-8') as fh:
        cfg = json.load(fh)
    print(f'config.json keys before: {sorted(cfg.keys())}')
    print(f"  dtype:        {cfg.get('dtype')!r}")
    print(f"  torch_dtype:  {cfg.get('torch_dtype')!r}")
    print(f"  _name_or_path: {cfg.get('_name_or_path')!r}")
    cfg['dtype'] = 'float32'
    cfg['torch_dtype'] = 'float32'
    cfg['_name_or_path'] = dst
    with open(config_path, 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, indent=2)
    print('config.json: dtype = float32, torch_dtype = float32, _name_or_path = scratch dir')

# Cast bfloat16 to float32 and save as pytorch_model.bin.
converted = 0
for name, tensor in list(state.items()):
    if tensor.dtype == torch.bfloat16:
        state[name] = tensor.to(torch.float32)
        converted += 1

out_path = os.path.join(dst, 'pytorch_model.bin')
torch.save(state, out_path)
print(f'wrote {out_path} ({converted} tensors cast)')

# Print the scratch contents so a stray safetensors is visible.
files = sorted(os.listdir(dst))
print(f'scratch contents ({len(files)}): {files}')

with open(result_path, 'w', encoding='utf-8') as fh:
    json.dump({'dir': dst, 'normalised': True, 'bf16_count': converted}, fh)
"@
Set-Content -LiteralPath (Join-Path $workDir '_normalise.py') -Value $normaliseSource -Encoding UTF8
$normalisePy = Join-Path $workDir '_normalise.py'

Invoke-Native -Command $venvPython -Arguments @($normalisePy)

if (-not (Test-Path -LiteralPath $normaliseResultFile)) { Fail "normalisation did not produce $normaliseResultFile" }
$normaliseResult = Get-Content -LiteralPath $normaliseResultFile -Raw | ConvertFrom-Json
$convertInputDir = $normaliseResult.dir
if ($normaliseResult.normalised) {
    Write-Ok "bfloat16 detected; wrote float32 copy to $convertInputDir"
} else {
    Write-Ok "checkpoint is already float32; using it as-is"
}

# ---------------------------------------------------------------------------
# 4. openai/whisper assets
# ---------------------------------------------------------------------------

Write-Step 'preparing openai/whisper assets'

$openaiWhisperRoot = $null
if (Test-Path -LiteralPath $OpenAIWhisperDir) {
    $openaiWhisperRoot = (Resolve-Path -LiteralPath $OpenAIWhisperDir).Path
    $assets = Join-Path $openaiWhisperRoot 'whisper\assets\mel_filters.npz'
    if (-not (Test-Path -LiteralPath $assets)) {
        Fail "found $OpenAIWhisperDir but $assets is missing; delete the directory and re-run."
    }
    Write-Ok "using existing openai/whisper at $openaiWhisperRoot"
} else {
    Require-Command -Name 'git' -Hint " Install git from https://git-scm.com/downloads and re-run."
    Write-Warn "cloning https://github.com/openai/whisper into $OpenAIWhisperDir"
    Invoke-Native -Command 'git' -Arguments @('clone', '--depth', '1', 'https://github.com/openai/whisper.git', $OpenAIWhisperDir)
    $openaiWhisperRoot = (Resolve-Path -LiteralPath $OpenAIWhisperDir).Path
    Write-Ok "cloned to $openaiWhisperRoot"
}

# ---------------------------------------------------------------------------
# 5. Run converter
# ---------------------------------------------------------------------------

Write-Step 'converting checkpoint to GGML'

$converterOutDir = Join-Path $workDir "$($modelInfo.FilePrefix)-out"
if (Test-Path -LiteralPath $converterOutDir) { Remove-Item -LiteralPath $converterOutDir -Recurse -Force }
New-Item -ItemType Directory -Path $converterOutDir -Force | Out-Null

Push-Location $whisperCppRoot
try {
    Invoke-Native -Command $venvPython -Arguments @($converter, $convertInputDir, $openaiWhisperRoot, $converterOutDir)
} finally {
    Pop-Location
}

$converterOutput = Join-Path $converterOutDir 'ggml-model.bin'
if (-not (Test-Path -LiteralPath $converterOutput)) {
    $fallback = @(Get-ChildItem -LiteralPath $converterOutDir -File -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($fallback.Count -eq 0) { Fail "converter did not produce a file under $converterOutDir" }
    $converterOutput = $fallback[0].FullName
    Write-Warn "converter wrote $($fallback[0].Name); expected ggml-model.bin"
}

$f16Name = "$($modelInfo.FilePrefix)-$($modelInfo.QuantForF16).bin"
$f16Path = Join-Path $workDir $f16Name
Move-Item -LiteralPath $converterOutput -Destination $f16Path -Force
Write-Ok "wrote $f16Path"

# ---------------------------------------------------------------------------
# 6. Quantize
# ---------------------------------------------------------------------------

$finalPath = $f16Path
if ($Quant -ne 'none') {
    Write-Step 'locating or building the quantize binary'

    $quantizeNames = @('whisper-quantize', 'quantize')
    $quantizeExe = Find-WhisperBinary -Root $whisperCppRoot -Names $quantizeNames
    if (-not $quantizeExe) {
        Require-Command -Name 'cmake' -Hint " Install CMake from https://cmake.org/download/ and re-run."
        Write-Warn 'quantize binary not found; building with CMake'
        $buildDir = Join-Path $whisperCppRoot 'build'
        Invoke-Native -Command 'cmake' -Arguments @('-B', $buildDir, '-DCMAKE_BUILD_TYPE=Release')
        Invoke-Native -Command 'cmake' -Arguments @('--build', $buildDir, '--config', 'Release', '-j')
        $quantizeExe = Find-WhisperBinary -Root $whisperCppRoot -Names $quantizeNames
        if (-not $quantizeExe) { Fail "built whisper.cpp but no quantize binary was produced under $buildDir" }
    }
    Write-Ok "quantize binary: $quantizeExe"

    Write-Step "quantizing to $Quant"
    $finalName = "$($modelInfo.FilePrefix)-$Quant.bin"
    $finalPath = Join-Path $OutDir $finalName
    Invoke-Native -Command $quantizeExe -Arguments @($f16Path, $finalPath, $Quant)
    Write-Ok "wrote $finalPath"
} else {
    $finalName = "$($modelInfo.FilePrefix)-f16.bin"
    $finalPath = Join-Path $OutDir $finalName
    Move-Item -LiteralPath $f16Path -Destination $finalPath -Force
    Write-Ok "wrote $finalPath"
}

# ---------------------------------------------------------------------------
# 7. Sanity test
# ---------------------------------------------------------------------------

if (-not $SkipSanityTest) {
    Write-Step 'running sanity transcription'
    $whisperCli = Find-WhisperBinary -Root $whisperCppRoot -Names @('whisper-cli', 'main')
    if (-not $whisperCli) {
        Write-Warn 'whisper-cli not built; skipping sanity test'
    } else {
        $benchmark = Join-Path (Split-Path -Parent $toolsDir) 'tests\audio\benchmark20s.wav'
        if (-not (Test-Path -LiteralPath $benchmark)) {
            Write-Warn "benchmark file not found at $benchmark; skipping sanity test"
        } else {
            Write-Ok "whisper-cli: $whisperCli"
            $transcription = & $whisperCli -m $finalPath -f $benchmark -l hi 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "sanity transcription exited $LASTEXITCODE"
            } else {
                $firstLines = ($transcription | Where-Object { $_ -match '\S' } | Select-Object -First 8) -join ' '
                $snippet = if ($firstLines.Length -gt 200) { $firstLines.Substring(0, 200) } else { $firstLines }
                Write-Host ''
                Write-Host 'Sanity transcription (first 200 chars):' -ForegroundColor Cyan
                Write-Host "  $snippet"
                Write-Host ''
                $nonAscii = ($snippet.ToCharArray() | Where-Object { [int]$_ -gt 127 }).Count
                if ($nonAscii -eq 0) {
                    Write-Ok 'output is Roman script (no non-ASCII characters detected)'
                } else {
                    Write-Warn "$nonAscii non-ASCII characters; verify the model produced Roman Hinglish"
                }
            }
        }
    }
}

# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------

Write-Step 'computing SHA-256'

$hash = (Get-FileHash -LiteralPath $finalPath -Algorithm SHA256).Hash.ToLowerInvariant()
$size = (Get-Item -LiteralPath $finalPath).Length
$fileName = Split-Path -Leaf $finalPath

Write-Host ''
Write-Host 'Paste into models.json:' -ForegroundColor Cyan
Write-Host ''
Write-Host '  {'
Write-Host "    `"name`": `"Whisper-Hindi2Hinglish-$Model`","
Write-Host "    `"tier`": `"$(if ($Model -eq 'Swift') { 'light' } else { 'standard' })`","
Write-Host "    `"description`": `"$($modelInfo.Description)`","
Write-Host "    `"filename`": `"$fileName`","
Write-Host "    `"url`": `"TODO`","
Write-Host "    `"revision`": `"$resolvedCommit`","
Write-Host "    `"size_bytes`": $size,"
Write-Host "    `"sha256`": `"$hash`","
Write-Host "    `"license`": `"$($modelInfo.License)`","
Write-Host "    `"quantization`": `"$(if ($Quant -eq 'none') { 'f16' } else { $Quant })`","
Write-Host "    `"source_repo`": `"$($modelInfo.RepoId)`""
Write-Host '  }'
Write-Host ''
Write-Host 'File details:' -ForegroundColor Cyan
Write-Host "  path:         $finalPath"
Write-Host "  size:         $size bytes ($([math]::Round($size / 1MB, 1)) MB)"
Write-Host "  sha256:       $hash"
Write-Host "  hf revision:  $resolvedCommit"
Write-Host ''
Write-Host 'Done.' -ForegroundColor Green