# tools/thread-sweep.ps1
<#
.SYNOPSIS
    Sweep threads x flash attention x model on one audio clip, CPU only.

.DESCRIPTION
    Runs bench-asr-cpu.exe across every combination and writes a single
    Markdown table you can paste back. Every leg is cache-free: bench-asr
    never reads or writes the cache, and `--no-cache` is passed for
    symmetry with the CLI.

    Use it to answer one question: does our CPU path reach upstream
    whisper-cli's throughput on the same model and clip, and if not, which
    of (threads, flash attention, model) explains the gap.

.PARAMETER Audio
    Clip path. Default: tests\audio\benchmark20s.wav.

.PARAMETER Models
    Comma-separated model keys. Recognised: apex-q5_0, apex-q8_0,
    apex-f16, swift-f16. Defaults to apex-q5_0,apex-q8_0.

.PARAMETER Threads
    Comma-separated thread counts. Defaults to 4,8,12,24.

.PARAMETER Flash
    Comma-separated flash-attn settings. `off` or `on`.
    Defaults to off,on.

.PARAMETER Output
    Markdown output file. Default: thread-sweep.md.

.PARAMETER TimeoutS
    Per-leg wall-clock limit. Default: 600.

.EXAMPLE
    pwsh tools\thread-sweep.ps1

.EXAMPLE
    pwsh tools\thread-sweep.ps1 -Models apex-q5_0 -Threads 4,12 -Flash off
#>
[CmdletBinding()]
param(
    [string]$Audio = "tests\audio\benchmark20s.wav",
    [string]$Models = "apex-q5_0,apex-q8_0",
    [string]$Threads = "4,8,12,24",
    [string]$Flash = "off,on",
    [string]$Output = "thread-sweep.md",
    [int]$TimeoutS = 600
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$exe = Join-Path $repoRoot "target\release\bench-asr-cpu.exe"
if (-not (Test-Path $exe)) {
    throw "Missing $exe. Build it with: cargo build --release -p svaraflow-cli --bin bench-asr"
}
if (-not (Test-Path $Audio)) {
    throw "Audio not found: $Audio"
}

$modelMap = @{
    "apex-q5_0"  = @{ File = "ggml-hinglish-apex-q5_0.bin"; Name = "Apex Q5_0"; Tier = "standard" }
    "apex-q8_0"  = @{ File = "ggml-hinglish-apex-q8_0.bin"; Name = "Apex Q8_0"; Tier = "standard" }
    "apex-f16"   = @{ File = "ggml-hinglish-apex-f16.bin";  Name = "Apex F16";  Tier = "standard" }
    "swift-f16"  = @{ File = "ggml-hinglish-swift-f16.bin"; Name = "Swift F16"; Tier = "light"    }
}

$modelKeys = $Models -split ',' | ForEach-Object { $_.Trim() }
$threadList = $Threads -split ',' | ForEach-Object { [int]$_.Trim() }
$flashList = $Flash -split ',' | ForEach-Object { $_.Trim().ToLower() }

$modelDir = "$env:LOCALAPPDATA\SvaraFlow\models"
$audioFull = (Resolve-Path $Audio).Path

$rows = New-Object System.Collections.Generic.List[object]
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# SvaraFlow CPU thread sweep")
$lines.Add("")
$lines.Add("Clip: ``$Audio``  ")
$lines.Add("Audio duration: $(('{0:F2}' -f (New-Object -TypeName System.Media.SoundPlayer 2>$null).Length))  ")
$lines.Add("Machine: $env:COMPUTERNAME  ")
$lines.Add("")

foreach ($key in $modelKeys) {
    if (-not $modelMap.ContainsKey($key)) {
        Write-Warning "Unknown model key: $key (skipping)"
        continue
    }
    $m = $modelMap[$key]
    $modelPath = Join-Path $modelDir $m.File
    if (-not (Test-Path $modelPath)) {
        Write-Warning "Model not on disk: $modelPath (skipping)"
        continue
    }

    foreach ($threads in $threadList) {
        foreach ($flash in $flashList) {
            if ($flash -notin @("on","off")) {
                Write-Warning "Unknown flash setting: $flash (skipping)"
                continue
            }
            $argv = @(
                "--model",      $modelPath,
                "--model-name", $m.Name,
                "--tier",       $m.Tier,
                "--audio",      $audioFull,
                "--threads",    "$threads",
                "--no-cache",
                "--no-temp-fallback"
            )
            if ($flash -eq "on") { $argv += "--flash-attn" }

            Write-Host ("[{0}] threads={1} flash={2}" -f $m.Name, $threads, $flash)

            $psi = [System.Diagnostics.ProcessStartInfo]::new()
            $psi.FileName               = $exe
            $psi.UseShellExecute        = $false
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError  = $true
            $psi.CreateNoWindow         = $true
            foreach ($a in $argv) { [void]$psi.ArgumentList.Add([string]$a) }

            $proc = [System.Diagnostics.Process]::new()
            $proc.StartInfo = $psi
            $out = ""
            $err = ""
            $code = -1
            $timedOut = $false

            try {
                [void]$proc.Start()
                $outTask = $proc.StandardOutput.ReadToEndAsync()
                $errTask = $proc.StandardError.ReadToEndAsync()
                $deadline = (Get-Date).AddSeconds($TimeoutS)
                while (-not $proc.WaitForExit(200)) {
                    if ((Get-Date) -gt $deadline) {
                        try { $proc.Kill($true) } catch { }
                        $timedOut = $true
                        break
                    }
                }
                $out = $outTask.GetAwaiter().GetResult()
                $err = $errTask.GetAwaiter().GetResult()
                if (-not $timedOut) { $code = $proc.ExitCode }
            } finally {
                $proc.Dispose()
            }

            $row = [pscustomobject]@{
                Model    = $m.Name
                Threads  = $threads
                Flash    = $flash
                TimedOut = $timedOut
                Rtf      = $null
                TotalS   = $null
                WhisperS = $null
                Tokens   = $null
                Exit     = $code
            }

            if (-not $timedOut -and $code -eq 0 -and $out) {
                try {
                    $j = $out | ConvertFrom-Json
                    $row.Rtf      = $j.rtf_total
                    $row.TotalS   = $j.total_s
                    $row.WhisperS = $j.timings.whisper_full_s
                    $tokens = 0
                    foreach ($w in $j.timings.windows) { $tokens += $w.tokens }
                    $row.Tokens = $tokens
                } catch {
                    Write-Warning "Could not parse JSON for $($m.Name) t=$threads f=$flash"
                }
            } elseif ($timedOut) {
                Write-Warning "  timed out after ${TimeoutS}s"
            } else {
                Write-Warning "  exit $code"
                if ($err) { ($err -split "`n" | Select-Object -Last 6) | ForEach-Object { Write-Warning "    $_" } }
            }

            $rows.Add($row)
        }
    }
}

$lines.Add("| Model | Threads | Flash | RTF (wall) | Total (s) | whisper_full (s) | Tokens | Status |")
$lines.Add("|---|---|---|---|---|---|---|---|")
foreach ($r in $rows) {
    if ($r.TimedOut) {
        $lines.Add("| $($r.Model) | $($r.Threads) | $($r.Flash) | - | - | - | - | timeout |")
    } elseif ($null -eq $r.Rtf) {
        $lines.Add("| $($r.Model) | $($r.Threads) | $($r.Flash) | - | - | - | - | exit $($r.Exit) |")
    } else {
        $lines.Add(("| {0} | {1} | {2} | {3:F2} | {4:F2} | {5:F2} | {6} | ok |" -f `
            $r.Model, $r.Threads, $r.Flash, `
            $r.Rtf, $r.TotalS, $r.WhisperS, $r.Tokens))
    }
}

$text = $lines -join "`n"
$text | Out-File -FilePath $Output -Encoding utf8
Write-Host ""
Write-Host $text
Write-Host ""
Write-Host "Wrote $Output"