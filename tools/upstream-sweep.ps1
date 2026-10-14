# tools/upstream-sweep.ps1
<#
.SYNOPSIS
    Sweep upstream whisper-cli across threads x flash-attn.

.DESCRIPTION
    Runs whisper-cli on one clip at the same model file bench-asr uses,
    parses the whisper_print_timings block, and writes a Markdown table.

    whisper-cli's flags, confirmed against its own --help:
      -t N          threads
      -bs N         beam size
      -bo N         best of
      -nt           no timestamps
      -l LANG       language
      -fa           flash attention ON (this is the default)
      -nfa          flash attention OFF
      -m MODEL
      -f AUDIO

.PARAMETER Cli
    Path to whisper-cli.exe.

.PARAMETER Model
    Path to the GGML model file.

.PARAMETER Audio
    Clip path. Default: tests\audio\benchmark20s.wav.

.PARAMETER Threads
    Comma-separated thread counts. Defaults to 4,8,12,24.

.PARAMETER Flash
    Comma-separated flash-attn settings: `on` or `off`. Defaults to on,off.

.PARAMETER Language
    Two-letter code or `auto`. Default: hi.

.PARAMETER Output
    Markdown output file. Default: upstream-sweep.md.
#>
[CmdletBinding()]
param(
    [string]$Cli = "C:\Users\The Abaddon\Desktop\DESKTOP\Personal CODING Projects\whisper.cpp\build\bin\Release\whisper-cli.exe",
    [string]$Model = "$env:LOCALAPPDATA\SvaraFlow\models\ggml-hinglish-apex-q5_0.bin",
    [string]$Audio = "tests\audio\benchmark20s.wav",
    [string]$Threads = "4,8,12,24",
    [string]$Flash = "on,off",
    [string]$Language = "hi",
    [string]$Output = "upstream-sweep.md"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path $Cli))   { throw "whisper-cli not found: $Cli" }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
if (-not (Test-Path $Audio)) { throw "Audio not found: $Audio" }

$threadList = $Threads -split ',' | ForEach-Object { [int]$_.Trim() }
$flashList  = $Flash  -split ',' | ForEach-Object { $_.Trim().ToLower() }
$audioFull  = (Resolve-Path $Audio).Path

$rows  = New-Object System.Collections.Generic.List[object]
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Upstream whisper-cli sweep")
$lines.Add("")
$lines.Add("CLI: ``$Cli``  ")
$lines.Add("Model: ``$Model``  ")
$lines.Add("Clip: ``$Audio``  ")
$lines.Add("Language: $Language  ")
$lines.Add("")

foreach ($threads in $threadList) {
    foreach ($fa in $flashList) {
        Write-Host ("[upstream] threads={0} flash-attn={1}" -f $threads, $fa)

        $argv = @(
            "-m",  $Model,
            "-f",  $audioFull,
            "-t",  "$threads",
            "-l",  $Language,
            "-bs", "1",
            "-bo", "1",
            "-nt"
        )
        if ($fa -eq "on")  { $argv += "-fa"  }
        if ($fa -eq "off") { $argv += "-nfa" }

        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName               = $Cli
        $psi.UseShellExecute        = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError  = $true
        $psi.CreateNoWindow         = $true
        foreach ($a in $argv) { [void]$psi.ArgumentList.Add([string]$a) }

        $proc = [System.Diagnostics.Process]::new()
        $proc.StartInfo = $psi
        try {
            [void]$proc.Start()
            $outTask = $proc.StandardOutput.ReadToEndAsync()
            $errTask = $proc.StandardError.ReadToEndAsync()
            $proc.WaitForExit()
            $out = $outTask.GetAwaiter().GetResult()
            $err = $errTask.GetAwaiter().GetResult()
            $code = $proc.ExitCode
        } finally {
            $proc.Dispose()
        }

        $combined = $out + "`n" + $err
        $loadMs = $null; $encodeMs = $null; $decodeMs = $null
        $batchdMs = $null; $totalMs = $null; $fallbacks = $null
        foreach ($line in ($combined -split "`n")) {
            if ($line -match 'load time\s*=\s*([\d\.]+)\s*ms')        { $loadMs    = [double]$Matches[1] }
            if ($line -match 'encode time\s*=\s*([\d\.]+)\s*ms')      { $encodeMs  = [double]$Matches[1] }
            if ($line -match 'decode time\s*=\s*([\d\.]+)\s*ms')      { $decodeMs  = [double]$Matches[1] }
            if ($line -match 'batchd time\s*=\s*([\d\.]+)\s*ms')      { $batchdMs  = [double]$Matches[1] }
            if ($line -match 'fallbacks\s*=\s*(\d+)')                 { $fallbacks = [int]$Matches[1] }
            if ($line -match 'total time\s*=\s*([\d\.]+)\s*(ms|s)') {
                $v = [double]$Matches[1]
                $totalMs = if ($Matches[2] -eq 's') { $v * 1000.0 } else { $v }
            }
        }

        $rows.Add([pscustomobject]@{
            Threads   = $threads
            Flash     = $fa
            LoadMs    = $loadMs
            EncodeMs  = $encodeMs
            DecodeMs  = $decodeMs
            BatchdMs  = $batchdMs
            TotalMs   = $totalMs
            Fallbacks = $fallbacks
            Exit      = $code
        })
    }
}

$lines.Add("| Threads | Flash | Load (ms) | Encode (ms) | Decode (ms) | batchd (ms) | Total (ms) | Fallbacks | Status |")
$lines.Add("|---|---|---|---|---|---|---|---|---|")
foreach ($r in $rows) {
    if ($r.Exit -ne 0) {
        $lines.Add("| $($r.Threads) | $($r.Flash) | - | - | - | - | - | - | exit $($r.Exit) |")
    } else {
        $lines.Add(("| {0} | {1} | {2:F2} | {3:F2} | {4:F2} | {5:F2} | {6:F2} | {7} | ok |" -f `
            $r.Threads, $r.Flash, $r.LoadMs, $r.EncodeMs, $r.DecodeMs, `
            $r.BatchdMs, $r.TotalMs, $r.Fallbacks))
    }
}

$text = $lines -join "`n"
$text | Out-File -FilePath $Output -Encoding utf8
Write-Host ""
Write-Host $text
Write-Host ""
Write-Host "Wrote $Output"