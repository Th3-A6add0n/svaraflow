# tools/bench-asr.ps1
<#
.SYNOPSIS
    Benchmark the Hinglish ASR models on the clips in tests/audio.

.DESCRIPTION
    Two modes.

    Default mode builds bench-asr twice (CPU and CUDA), then runs it on
    every model that is both declared below and present in the models
    folder, on every .wav / .mp4 clip in tests/audio with a matching .txt
    reference transcript. Produces one Markdown table per run.

    -Matrix mode runs one switch at a time against a baseline on a single
    clip, per device, at a fixed thread count, with a per-leg timeout.
    Produces a Markdown table the user can paste back for analysis.
    It does not change any defaults.

    Peak RAM comes from the OS-reported working set of the bench process.
    Peak VRAM comes from polling nvidia-smi while the process runs. Both
    are sampled in the main loop; no background job is used.

.PARAMETER ModelsDir
    Where the GGML files live. Default: %LOCALAPPDATA%\SvaraFlow\models.

.PARAMETER AudioDir
    Test clips. Default: tests\audio.

.PARAMETER TimingDir
    Labeled timing JSON. Default: tests\timing.

.PARAMETER VadModel
    Optional Silero VAD ONNX path.

.PARAMETER Output
    Markdown file to write. Default: bench-asr.md.

.PARAMETER Threads
    Decoder threads. 0 (default) means "all logical cores". -Matrix
    forces 12 unless overridden.

.PARAMETER Matrix
    Run one switch at a time against a baseline.

.PARAMETER MatrixAudio
    Clip for -Matrix. Default: tests\audio\benchmark20s.wav.

.PARAMETER TimeoutS
    Per-leg wall-clock limit in -Matrix. Default: 300.

.PARAMETER SkipBuild
    Do not run cargo; use existing bench-asr-{cpu,gpu}.exe.

.PARAMETER CpuOnly / .PARAMETER GpuOnly
    Run only one mode.

.EXAMPLE
    .\tools\bench-asr.ps1

.EXAMPLE
    .\tools\bench-asr.ps1 -Matrix -SkipBuild
#>
[CmdletBinding()]
param(
    [string]$ModelsDir = "$env:LOCALAPPDATA\SvaraFlow\models",
    [string]$AudioDir = "tests\audio",
    [string]$TimingDir = "tests\timing",
    [string]$VadModel = "models\silero_vad_v5.onnx",
    [string]$Output = "bench-asr.md",
    [int]$Threads = 0,
    [switch]$Matrix,
    [string]$MatrixAudio = "tests\audio\benchmark20s.wav",
    [int]$TimeoutS = 300,
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

# ---------- VAD ----------
$vadArg = $null
if ($VadModel -and (Test-Path $VadModel)) {
    $vadArg = (Resolve-Path $VadModel).Path
    Write-Host "VAD model: $vadArg"
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

# ---------- one process run ----------
#
# Uses System.Diagnostics.ProcessStartInfo.ArgumentList so paths with
# spaces (this repo lives under "Personal CODING Projects") are passed as
# individual argv entries. Samples WorkingSet64 and nvidia-smi memory.used
# while the process runs; no background job.
function Invoke-BenchOnce {
    param(
        [string]$Exe,
        [string[]]$ExtraArgs,
        [string]$Mode,
        [int]$TimeoutSeconds
    )

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName               = $Exe
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow         = $true
    foreach ($a in $ExtraArgs) {
        [void]$psi.ArgumentList.Add([string]$a)
    }

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    $stdout = ""
    $stderr = ""
    $code = -1
    $peakRamBytes = 0L
    $peakVramMb = $null
    $timedOut = $false

    try {
        [void]$proc.Start()
        $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
        $stderrTask = $proc.StandardError.ReadToEndAsync()

        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        while (-not $proc.WaitForExit(200)) {
            try {
                $proc.Refresh()
                $ws = $proc.WorkingSet64
                if ($ws -gt $peakRamBytes) { $peakRamBytes = $ws }
            } catch { }
            if ($Mode -eq "GPU") {
                $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
                if ($v) {
                    $vv = [int]($v | Select-Object -First 1)
                    if ($null -eq $peakVramMb -or $vv -gt $peakVramMb) { $peakVramMb = $vv }
                }
            }
            if ((Get-Date) -gt $deadline) {
                try { $proc.Kill($true) } catch { }
                $timedOut = $true
                break
            }
        }

        if (-not $timedOut) {
            try {
                $proc.Refresh()
                if ($proc.WorkingSet64 -gt $peakRamBytes) { $peakRamBytes = $proc.WorkingSet64 }
            } catch { }
        }

        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if (-not $timedOut) { $code = $proc.ExitCode }
    } finally {
        $proc.Dispose()
    }

    if ($timedOut) {
        Write-Warning "timed out after ${TimeoutSeconds}s"
        return [pscustomobject]@{
            Ok = $false
            TimedOut = $true
            PeakRamMb = $peakRamBytes / 1MB
            PeakVramMb = $peakVramMb
            Json = $null
            StderrTail = ($stderr -split "`n" | Select-Object -Last 6) -join "`n"
        }
    }

    $obj = $null
    if ($code -eq 0 -and $stdout) {
        try { $obj = $stdout | ConvertFrom-Json } catch { $obj = $null }
    }

    [pscustomobject]@{
        Ok = ($code -eq 0 -and $null -ne $obj)
        TimedOut = $false
        ExitCode = $code
        PeakRamMb = $peakRamBytes / 1MB
        PeakVramMb = $peakVramMb
        Json = $obj
        StderrTail = if ($code -ne 0) { ($stderr -split "`n" | Select-Object -Last 6) -join "`n" } else { "" }
    }
}

# ---------- helpers for pretty output ----------
function Format-Num([object]$v, [string]$fmt = "{0:F2}") {
    if ($null -eq $v) { return "n/a" }
    return ($fmt -f [double]$v)
}
function Format-Int([object]$v) {
    if ($null -eq $v) { return "n/a" }
    return ("{0:F0}" -f [double]$v)
}
function Format-Pct([object]$v) {
    if ($null -eq $v) { return "n/a" }
    return ("{0:P1}" -f [double]$v)
}

# ============================================================
# -Matrix mode
# ============================================================
if ($Matrix) {
    if (-not (Test-Path $MatrixAudio)) {
        Write-Error "Matrix audio not found: $MatrixAudio"
        exit 1
    }
    $matrixThreads = if ($Threads -gt 0) { $Threads } else { 12 }

    # One switch at a time against a baseline. --threads is always set to
    # matrixThreads and is not one of the varied switches.
    $rows = @(
        [pscustomobject]@{ Label = "baseline";          Args = @()                                  }
        [pscustomobject]@{ Label = "flash-attn";        Args = @("--flash-attn")                    }
        [pscustomobject]@{ Label = "no-temp-fallback";  Args = @("--no-temp-fallback")              }
        [pscustomobject]@{ Label = "beam 5";            Args = @("--beam", "5")                     }
        [pscustomobject]@{ Label = "audio-ctx 768";     Args = @("--audio-ctx", "768")              }
    )

    $modes = if ($CpuOnly) { @("CPU") } elseif ($GpuOnly) { @("GPU") } else { @("GPU", "CPU") }

    # Collect rows into a flat list of results.
    $matrixResults = New-Object System.Collections.Generic.List[object]
    foreach ($mode in $modes) {
        $exe = if ($mode -eq "GPU") { $gpuExe } else { $cpuExe }
        if (-not (Test-Path $exe)) { Write-Warning "Missing $exe"; continue }
        foreach ($model in $present) {
            foreach ($row in $rows) {
                Write-Host ("[matrix] {0} {1} / {2} on {3}" -f $mode, $model.Name, $row.Label, (Split-Path $MatrixAudio -Leaf))
                $argv = @(
                    "--model",      $model.Path,
                    "--model-name", $model.Name,
                    "--tier",       $model.Tier,
                    "--audio",      (Resolve-Path $MatrixAudio).Path,
                    "--threads",    "$matrixThreads"
                ) + $row.Args
                if ($vadArg) { $argv += @("--vad-model", $vadArg) }

                $r = Invoke-BenchOnce -Exe $exe -ExtraArgs $argv -Mode $mode -TimeoutSeconds $TimeoutS
                $matrixResults.Add([pscustomobject]@{
                    Mode    = $mode
                    Model   = $model.Name
                    Row     = $row.Label
                    Timeout = $r.TimedOut
                    Ok      = $r.Ok
                    PeakRam = $r.PeakRamMb
                    PeakVram = $r.PeakVramMb
                    Json    = $r.Json
                    Stderr  = $r.StderrTail
                })
                if ($r.TimedOut) { Write-Warning "  timed out" }
                elseif (-not $r.Ok) { Write-Warning "  exit $($r.ExitCode)" }
                else {
                    $j = $r.Json
                    Write-Host ("  xRT(wall)={0}  whisper_full={1}s  VRAM={2} MB" -f `
                        (Format-Num $j.rtf_total "{0:F2}"), `
                        (Format-Num $j.timings.whisper_full_s "{0:F2}"), `
                        (Format-Int $r.PeakVram))
                }
            }
        }
    }

    # ---------- matrix table ----------
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# SvaraFlow ASR -Matrix")
    $lines.Add("")
    $lines.Add("Clip: `$MatrixAudio` = $MatrixAudio  ")
    $lines.Add("Threads: $matrixThreads  ")
    $lines.Add("Timeout per leg: ${TimeoutS}s  ")
    if ($missing.Count -gt 0) { $lines.Add("Skipped (not on disk): $($missing -join ', ').") }
    $lines.Add("")
    $lines.Add("| Device | Model | Variant | xRT (wall) | decode | resample | VAD | plan | load | transcribe | whisper_full | post | export | decode_retries | DTW | Peak RAM (MB) | Peak VRAM (MB) | Status |")
    $lines.Add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    foreach ($mode in $modes) {
        foreach ($model in $present) {
            foreach ($row in $rows) {
                $cell = $matrixResults | Where-Object {
                    $_.Mode -eq $mode -and $_.Model -eq $model.Name -and $_.Row -eq $row.Label
                }
                if (-not $cell) { continue }
                if ($cell.Timeout) {
                    $lines.Add(("| {0} | {1} | {2} | - | - | - | - | - | - | - | - | - | - | - | - | {3} | {4} | timeout |" -f `
                        $mode, $model.Name, $row.Label, (Format-Int $cell.PeakRam), (Format-Int $cell.PeakVram)))
                    continue
                }
                if (-not $cell.Ok) {
                    $lines.Add(("| {0} | {1} | {2} | - | - | - | - | - | - | - | - | - | - | - | - | {3} | {4} | failed |" -f `
                        $mode, $model.Name, $row.Label, (Format-Int $cell.PeakRam), (Format-Int $cell.PeakVram)))
                    continue
                }
                $j = $cell.Json
                $t = $j.timings
                $dtw = if ($t.dtw_usable) { "yes" } else { "no" }
                $retries = if ($null -eq $t.decode_retries) { "n/a" } else { "$($t.decode_retries)" }
                $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} | {13} | {14} | {15} | {16} | ok |" -f `
                    $mode, $model.Name, $row.Label, `
                    (Format-Num $j.rtf_total "{0:F2}"), `
                    (Format-Num $t.audio_decode_s "{0:F3}"), `
                    (Format-Num $t.resample_s "{0:F3}"), `
                    (Format-Num $t.vad_s "{0:F3}"), `
                    (Format-Num $t.chunk_plan_s "{0:F3}"), `
                    (Format-Num $t.model_load_s "{0:F3}"), `
                    (Format-Num $t.transcribe_s "{0:F3}"), `
                    (Format-Num $t.whisper_full_s "{0:F3}"), `
                    (Format-Num $t.postprocess_s "{0:F3}"), `
                    (Format-Num $t.export_s "{0:F3}"), `
                    $retries, $dtw, `
                    (Format-Int $cell.PeakRam), `
                    (Format-Int $cell.PeakVram)))
            }
        }
    }

    $text = $lines -join "`n"
    $text | Out-File -FilePath $Output -Encoding utf8
    Write-Host ""
    Write-Host $text
    Write-Host ""
    Write-Host "Wrote $Output"
    return
}

# ============================================================
# Default mode: full sweep
# ============================================================
$audioFiles = Get-ChildItem -Path $AudioDir -File |
    Where-Object { $_.Extension -in ".wav", ".mp4" } |
    Sort-Object Name
if ($audioFiles.Count -eq 0) {
    Write-Error "No .wav or .mp4 files in $AudioDir"
    exit 1
}

$results = New-Object System.Collections.Generic.List[object]
$modes = if ($CpuOnly) { @("CPU") } elseif ($GpuOnly) { @("GPU") } else { @("GPU", "CPU") }

foreach ($mode in $modes) {
    $exe = if ($mode -eq "GPU") { $gpuExe } else { $cpuExe }
    if (-not (Test-Path $exe)) { Write-Warning "Missing $exe; skipping $mode"; continue }
    foreach ($model in $present) {
        foreach ($audio in $audioFiles) {
            $stem      = [System.IO.Path]::GetFileNameWithoutExtension($audio.Name)
            $reference = Join-Path $audio.DirectoryName "$stem.txt"
            if (-not (Test-Path $reference)) { $reference = $null }
            $timing = Join-Path $TimingDir "$stem.json"
            if (-not (Test-Path $timing)) { $timing = $null }

            Write-Host ("[{0}] {1} on {2}" -f $mode, $model.Name, $audio.Name)
            $argv = @(
                "--model",      $model.Path,
                "--model-name", $model.Name,
                "--tier",       $model.Tier,
                "--audio",      $audio.FullName,
                "--threads",    "$Threads"
            )
            if ($reference) { $argv += @("--reference", $reference) }
            if ($timing)    { $argv += @("--timing", $timing) }
            if ($vadArg)    { $argv += @("--vad-model", $vadArg) }

            $r = Invoke-BenchOnce -Exe $exe -ExtraArgs $argv -Mode $mode -TimeoutSeconds $TimeoutS
            if ($r.Ok) {
                $r.Json | Add-Member -NotePropertyName mode -NotePropertyValue $mode -Force
                $r.Json | Add-Member -NotePropertyName peak_ram_mb_os -NotePropertyValue $r.PeakRamMb -Force
                $r.Json | Add-Member -NotePropertyName peak_vram_mb -NotePropertyValue $r.PeakVramMb -Force
                $results.Add($r.Json)
            } else {
                Write-Warning "bench-asr failed for $($model.Name) / $($audio.Name)"
                if ($r.StderrTail) { Write-Warning $r.StderrTail }
            }
        }
    }
}

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# SvaraFlow ASR benchmark")
$lines.Add("")
if ($missing.Count -gt 0) { $lines.Add("Skipped (not on disk): $($missing -join ', ')."); $lines.Add("") }
if ($Threads -gt 0) { $lines.Add("CPU runs used $Threads decoder threads; GPU runs are unaffected."); $lines.Add("") }
$lines.Add("| Model | Mode | xRT (wall) | Load (s) | Peak RAM (MB) | Peak VRAM (MB) | WER | Median word offset (ms) | Estimated words | DTW |")
$lines.Add("|---|---|---|---|---|---|---|---|---|---|")

foreach ($model in $present) {
    foreach ($mode in $modes) {
        $subset = @($results | Where-Object { $_.model -eq $model.Name -and $_.mode -eq $mode })
        if ($subset.Count -eq 0) { continue }
        $avgRtf  = ($subset | Measure-Object -Property rtf_total -Average).Average
        $avgLoad = ($subset | Measure-Object -Property load_s -Average).Average
        $avgRam  = ($subset | Measure-Object -Property peak_ram_mb_os -Average).Average
        $vram = @($subset | Where-Object { $_.peak_vram_mb -ne $null } | ForEach-Object { $_.peak_vram_mb })
        $vramStr = if ($vram.Count -gt 0) { "{0:F0}" -f (($vram | Measure-Object -Average).Average) } else { "n/a" }
        $wers = @($subset | Where-Object { $_.wer -ne $null } | ForEach-Object { $_.wer })
        $werStr = if ($wers.Count -gt 0) { "{0:P1}" -f (($wers | Measure-Object -Average).Average) } else { "n/a" }
        $offs = @($subset | Where-Object { $_.median_word_offset_ms -ne $null } | ForEach-Object { $_.median_word_offset_ms })
        $offStr = if ($offs.Count -gt 0) { "{0:F1}" -f (($offs | Measure-Object -Average).Average) } else { "n/a" }
        $estTotal  = ($subset | Measure-Object -Property estimated_words -Sum).Sum
        $wordTotal = ($subset | Measure-Object -Property total_words     -Sum).Sum
        $estStr = if ($wordTotal -gt 0) { "{0:P0} ({1}/{2})" -f ($estTotal / $wordTotal), $estTotal, $wordTotal } else { "n/a" }
        $dtwYes = ($subset | Where-Object { $_.timings.dtw_usable } | Measure-Object).Count
        $dtwStr = if ($dtwYes -gt 0) { "yes" } else { "no" }

        $lines.Add(("| {0} | {1} | {2:F2} | {3:F2} | {4:F0} | {5} | {6} | {7} | {8} | {9} |" -f `
            $model.Name, $mode, $avgRtf, $avgLoad, $avgRam, $vramStr, $werStr, $offStr, $estStr, $dtwStr))
    }
}

$text = $lines -join "`n"
$text | Out-File -FilePath $Output -Encoding utf8
Write-Host ""
Write-Host $text
Write-Host ""
Write-Host "Wrote $Output"