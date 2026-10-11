$exe = "target\release\bench-asr-cpu.exe"
$mdl = "$env:LOCALAPPDATA\SvaraFlow\models\ggml-hinglish-swift-f16.bin"

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = (Resolve-Path $exe).Path
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true
foreach ($a in @("--model", $mdl, "--model-name", "Swift F16", "--tier", "light",
                 "--audio", "tests\audio\benchmark20s.wav", "--threads", "0")) {
    [void]$psi.ArgumentList.Add($a)
}
$p = [System.Diagnostics.Process]::Start($psi)
$out = $p.StandardOutput.ReadToEnd()
$err = $p.StandardError.ReadToEnd()
$p.WaitForExit()
$out
"--- exit: $($p.ExitCode) ---"
$err -split "`n" | Select-Object -Last 4