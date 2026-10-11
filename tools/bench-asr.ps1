<#
.SYNOPSIS
    Benchmark the Hinglish ASR models on the clips in tests/audio.

.DESCRIPTION
    Builds bench-asr twice (CPU-only and CUDA), then runs it on every model
    that is both declared below and present in the models folder, on every
    .wav / .mp4 clip in tests/audio with a matching .txt reference transcript.

    Peak RAM comes from the Windows-reported peak working set of the bench
    process. Peak VRAM comes from polling nvidia-smi while the process runs.

    Models that are declared but not on disk are skipped and named in the
    table caption. Missing reference .txt files cause the WER column to be
    blank for that clip; missing tests/timing/<stem>.json files cause the
    word-offset column to be blank.

.PARAMETER ModelsDir
    Where the GGML files live. Default: %LOCALAPPDATA%\SvaraFlow\models.

.PARAMETER AudioDir
    Test clips. Default: tests\audio.

.PARAMETER TimingDir
    Labeled timing JSON. Default: tests\timing.

.PARAMETER VadModel
    Optional Silero VAD ONNX path. If the file does not exist, chunking
    falls back to fixed windows. Default: models\silero_vad_v5.onnx.

.PARAMETER Output
    Markdown file to write. Default: bench-asr.md.

.PARAMETER Threads
    Decoder threads handed to whisper.cpp. 0 (default) means "all logical
    cores". Override to test scaling.

.PARAMETER SkipBuild
    Do not run cargo; use existing target\release\bench-asr-{cpu,gpu}.exe.

.PARAMETER CpuOnly / .PARAMETER GpuOnly
    Run only one mode.

.EXAMPLE
    .\tools\bench-asr.ps1

.EXAMPLE
    # Just the CPU half, with 8 threads, skipping the rebuild.
    .\tools\bench-asr.ps1 -CpuOnly -Threads 8 -SkipBuild
#>
[CmdletBinding()]
param(
    [string]$ModelsDir = "$env:LOCALAPPDATA\SvaraFlow\models",
    [string]$AudioDir = "tests\audio",
    [string]$TimingDir = "tests\timing",
    [string]$VadModel = "models\silero_vad_v5.onnx",
    [string]$Output = "bench-asr.md",
    [int]$Threads = 0,
    [switch]$SkipBuild,
    [switch]$CpuOnly,
    [switch]$GpuOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($CpuOnly -and $GpuOnly) {
    throw "CpuOnly and GpuOnly are mutually exclusive."
}

# ---------- desired model set ----------
# Present-on-disk wins; anything absent is silently skipped and named in
# the output.
$desired = @(
    [pscustomobject]@{ Name = "Swift F16"; File = "ggml-hinglish-swift-f16.bin";  Tier = "light"    }
    [pscustomobject]@{ Name = "Swift Q5";  File = "ggml-hinglish-swift-q5_0.bin"; Tier = "light"    }
    [pscustomobject]@{ Name = "Swift Q8";  File = "ggml-hinglish-swift-q8_0.bin"; Tier = "light"    }
    [pscustomobject]@{ Name = "Apex Q5_0"; File = "ggml-hinglish-apex-q5_0.bin";  Tier = "standard" }
    [pscustomobject]@{ Name = "Apex Q8_0"; File = "ggml-hinglish-apex-q8_0.bin";  Tier = "standard" }
)

$present = New-Object System.Collections.Generic.List[object]
$missing = New-Object System.Collections.Generic.List[string]
foreach ($m in $desired) {
    $path = Join-Path $ModelsDir $m.File
    if (Test-Path $path) {
        $m | Add-Member -NotePropertyName Path -NotePropertyValue $path -Force
        $present.Add($m)
    } else {
        $missing.Add($m.Name)
    }
}

if ($present.Count -eq 0) {
    Write-Error "No benchmark models found in $ModelsDir. Looked for: $($desired.File -join ', ')"
    exit 1
}
if ($missing.Count -gt 0) {
    Write-Host "Not on disk (will be skipped): $($missing -join ', ')"
}

# ---------- audio clips ----------
$audioFiles = Get-ChildItem -Path $AudioDir -File |
    Where-Object { $_.Extension -in ".wav", ".mp4" } |
    Sort-Object Name
if ($audioFiles.Count -eq 0) {
    Write-Error "No .wav or .mp4 files in $AudioDir"
    exit 1
}

$vadArg = $null
if ($VadModel -and (Test-Path $VadModel)) {
    $vadArg = (Resolve-Path $VadModel).Path
    Write-Host "VAD model: $vadArg"
} else {
    Write-Host "No VAD model found; chunking will use fixed windows."
}

# ---------- build ----------
$cpuExe = Join-Path $repoRoot "target\release\bench-asr-cpu.exe"
$gpuExe = Join-Path $repoRoot "target\release\bench-asr-gpu.exe"

if (-not $SkipBuild) {
    if (-not $GpuOnly) {
        Write-Host "Building CPU bench-asr ..."
        cargo build --release -p svaraflow-cli --bin bench-asr
        if ($LASTEXITCODE -ne 0) { throw "CPU build failed" }
        Copy-Item "target\release\bench-asr.exe" $cpuExe -Force
    }
    if (-not $CpuOnly) {
        Write-Host "Building GPU (CUDA) bench-asr ..."
        cargo build --release -p svaraflow-cli --bin bench-asr --features cuda
        if ($LASTEXITCODE -ne 0) { throw "GPU build failed" }
        Copy-Item "target\release\bench-asr.exe" $gpuExe -Force
    }
}

# ---------- runner ----------
# Uses System.Diagnostics.ProcessStartInfo.ArgumentList rather than
# Start-Process -ArgumentList. The latter takes a single raw command-line
# string and, in PowerShell 7, wraps the whole thing in an extra layer of
# quotes, so the child receives one argv element and clap reports every
# required flag as missing. ArgumentList escapes each element against the
# Windows CreateProcess rules, which is the guarantee we need for paths
# that contain spaces (this repo lives under "Personal CODING Projects").
function Invoke-Bench {
    param(
        [string]$Exe,
        [pscustomobject]$Model,
        [string]$AudioPath,
        [string]$Reference,
        [string]$Timing,
        [string]$Mode,
        [string]$Vad,
        [int]$Threads
    )

    $argList = @(
        "--model",      $Model.Path,
        "--model-name", $Model.Name,
        "--tier",       $Model.Tier,
        "--audio",      $AudioPath,
        "--threads",    "$Threads"
    )
    if ($Reference) { $argList += @("--reference", $Reference) }
    if ($Timing)    { $argList += @("--timing",    $Timing) }
    if ($Vad)       { $argList += @("--vad-model", $Vad) }

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName               = $Exe
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow         = $true
    foreach ($a in $argList) {
        [void]$psi.ArgumentList.Add([string]$a)
    }

    $vramSamples = New-Object System.Collections.Generic.List[int]
    $pollJob = $null
    if ($Mode -eq "GPU") {
        $pollJob = Start-Job -ScriptBlock {
            while ($true) {
                $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
                if ($v) { [int]$v }
                Start-Sleep -Milliseconds 200
            }
        }
    }

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    $stdout = ""
    $stderr = ""
    $code = -1
    $peakRamBytes = $null
    try {
        [void]$proc.Start()
        $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
        $stderrTask = $proc.StandardError.ReadToEndAsync()
        $proc.WaitForExit()
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $code = $proc.ExitCode
        # Windows-reported peak RSS for the process, in bytes. Unlike the
        # Rust allocator this includes whisper.cpp's C-side allocations.
        $peakRamBytes = $proc.PeakWorkingSet64
    } finally {
        if ($pollJob) {
            Stop-Job $pollJob -ErrorAction SilentlyContinue
            $vramSamples = @(Receive-Job $pollJob)
            Remove-Job $pollJob -Force -ErrorAction SilentlyContinue
        }
        $proc.Dispose()
    }

    if ($code -ne 0 -or -not $stdout) {
        Write-Warning "bench-asr failed for $($Model.Name) / $($AudioPath) (exit $code)"
        if ($stderr) {
            Write-Warning "stderr tail:"
            ($stderr -split "`n" | Select-Object -Last 8) | ForEach-Object { Write-Warning "  $_" }
        }
        return $null
    }

    try {
        $obj = $stdout | ConvertFrom-Json
    } catch {
        Write-Warning "bench-asr produced non-JSON for $($Model.Name) / $($AudioPath): $_"
        return $null
    }

    $peakVram = if ($vramSamples.Count -gt 0) {
        ($vramSamples | Measure-Object -Maximum).Maximum
    } else { $null }

    $obj | Add-Member -NotePropertyName mode           -NotePropertyValue $Mode -Force
    $obj | Add-Member -NotePropertyName peak_vram_mb   -NotePropertyValue $peakVram -Force
    $obj | Add-Member -NotePropertyName peak_ram_mb_os -NotePropertyValue ([double]($peakRamBytes / 1MB)) -Force
    return $obj
}

# ---------- run everything ----------
$results = New-Object System.Collections.Generic.List[object]
$modes = if ($CpuOnly) { @("CPU") } elseif ($GpuOnly) { @("GPU") } else { @("GPU", "CPU") }

foreach ($mode in $modes) {
    $exe = if ($mode -eq "GPU") { $gpuExe } else { $cpuExe }
    if (-not (Test-Path $exe)) {
        Write-Warning "Missing $exe; skipping $mode"
        continue
    }
    foreach ($model in $present) {
        foreach ($audio in $audioFiles) {
            $stem      = [System.IO.Path]::GetFileNameWithoutExtension($audio.Name)
            $reference = Join-Path $audio.DirectoryName "$stem.txt"
            if (-not (Test-Path $reference)) { $reference = $null }

            $timing = Join-Path $TimingDir "$stem.json"
            if (-not (Test-Path $timing)) { $timing = $null }

            Write-Host ("[{0}] {1} on {2}" -f $mode, $model.Name, $audio.Name)
            $r = Invoke-Bench -Exe $exe -Model $model -AudioPath $audio.FullName `
                -Reference $reference -Timing $timing -Mode $mode -Vad $vadArg -Threads $Threads
            if ($r) { $results.Add($r) }
        }
    }
}

# ---------- aggregate ----------
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# SvaraFlow ASR benchmark")
$lines.Add("")
if ($missing.Count -gt 0) {
    $lines.Add("Skipped (not on disk): $($missing -join ', ').")
    $lines.Add("")
}
if ($Threads -gt 0) {
    $lines.Add("CPU runs used $Threads decoder threads; GPU runs are unaffected.")
    $lines.Add("")
}
$lines.Add("| Model | Mode | Speed (xRT) | Load (s) | Peak RAM (MB) | Peak VRAM (MB) | WER | Median word offset (ms) | Estimated words |")
$lines.Add("|---|---|---|---|---|---|---|---|---|")

foreach ($model in $present) {
    foreach ($mode in $modes) {
        $subset = @($results | Where-Object { $_.model -eq $model.Name -and $_.mode -eq $mode })
        if ($subset.Count -eq 0) { continue }

        $avgRtf  = ($subset | Measure-Object -Property rtf_total      -Average).Average
        $avgLoad = ($subset | Measure-Object -Property load_s         -Average).Average
        $avgRam  = ($subset | Measure-Object -Property peak_ram_mb_os -Average).Average

        $vram = @($subset | Where-Object { $_.peak_vram_mb -ne $null } | ForEach-Object { $_.peak_vram_mb })
        $vramStr = if ($vram.Count -gt 0) {
            "{0:F0}" -f (($vram | Measure-Object -Average).Average)
        } else { "n/a" }

        $wers = @($subset | Where-Object { $_.wer -ne $null } | ForEach-Object { $_.wer })
        $werStr = if ($wers.Count -gt 0) {
            "{0:P1}" -f (($wers | Measure-Object -Average).Average)
        } else { "n/a" }

        $offs = @($subset | Where-Object { $_.median_word_offset_ms -ne $null } | ForEach-Object { $_.median_word_offset_ms })
        $offStr = if ($offs.Count -gt 0) {
            "{0:F1}" -f (($offs | Measure-Object -Average).Average)
        } else { "n/a" }

        $estTotal  = ($subset | Measure-Object -Property estimated_words -Sum).Sum
        $wordTotal = ($subset | Measure-Object -Property total_words     -Sum).Sum
        $estStr = if ($wordTotal -gt 0) {
            "{0:P0} ({1}/{2})" -f ($estTotal / $wordTotal), $estTotal, $wordTotal
        } else { "n/a" }

        $lines.Add(("| {0} | {1} | {2:F2} | {3:F2} | {4:F0} | {5} | {6} | {7} | {8} |" -f `
            $model.Name, $mode, $avgRtf, $avgLoad, $avgRam, $vramStr, $werStr, $offStr, $estStr))
    }
}

$text = $lines -join "`n"
$text | Out-File -FilePath $Output -Encoding utf8
Write-Host ""
Write-Host $text
Write-Host ""
Write-Host "Wrote $Output"