<#
.SYNOPSIS
    Verifies that the SvaraFlow development toolchain is installed.

.DESCRIPTION
    Probes each required tool, prints its version in green when present and
    working, in red when missing or broken. Exits non-zero if any tool failed
    so it can be used as a gate in CI or pre-commit hooks.

.EXAMPLE
    pwsh tools/check-env.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

# --- helpers ---------------------------------------------------------------

function Test-SvaraFlowTool {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string]   $Name,
        [Parameter(Mandatory)] [string[]] $CommandCandidates,
        [Parameter(Mandatory)] [string[]] $VersionArgs
    )

    $cmd = $null
    foreach ($candidate in $CommandCandidates) {
        $found = Get-Command -Name $candidate -ErrorAction SilentlyContinue
        if ($null -ne $found) {
            $cmd = $found
            break
        }
    }

    if ($null -eq $cmd) {
        Write-Host ("[RED]    {0,-10} missing (tried: {1})" -f $Name, ($CommandCandidates -join ', ')) -ForegroundColor Red
        return $false
    }

    $global:LASTEXITCODE = 0
    try {
        $raw = & $cmd.Source @VersionArgs 2>&1 | Out-String
        $exit = $LASTEXITCODE
    }
    catch {
        Write-Host ("[RED]    {0,-10} {1}" -f $Name, $_.Exception.Message) -ForegroundColor Red
        return $false
    }

    if ($exit -ne 0) {
        Write-Host ("[RED]    {0,-10} found at {1} but exited {2}" -f $Name, $cmd.Source, $exit) -ForegroundColor Red
        return $false
    }

    $firstLine = ($raw -split "`r?`n" | Where-Object { $_ -match '\S' } | Select-Object -First 1)
    if ($null -eq $firstLine) { $firstLine = '(no output)' }

    Write-Host ("[GREEN]  {0,-10} {1}" -f $Name, $firstLine.Trim()) -ForegroundColor Green
    return $true
}

# --- checks ----------------------------------------------------------------

$checks = @(
    @{ Name = 'git';        Cmds = @('git');                    Args = @('--version') }
    @{ Name = 'gh';         Cmds = @('gh');                     Args = @('--version') }
    @{ Name = 'rustc';      Cmds = @('rustc');                  Args = @('--version') }
    @{ Name = 'cargo';      Cmds = @('cargo');                  Args = @('--version') }
    @{ Name = 'cmake';      Cmds = @('cmake');                  Args = @('--version') }
    @{ Name = 'clang';      Cmds = @('clang');                  Args = @('--version') }
    @{ Name = 'node';       Cmds = @('node');                   Args = @('--version') }
    @{ Name = 'python';     Cmds = @('python','python3','py');  Args = @('--version') }
    @{ Name = 'ffmpeg';     Cmds = @('ffmpeg');                 Args = @('-version') }
    @{ Name = 'nvidia-smi'; Cmds = @('nvidia-smi');             Args = @('--version') }
    @{ Name = 'nvcc';       Cmds = @('nvcc');                   Args = @('--version') }
)

Write-Host 'SvaraFlow environment check'
Write-Host ('-' * 60)

$allOk = $true
foreach ($c in $checks) {
    $ok = Test-SvaraFlowTool -Name $c.Name -CommandCandidates $c.Cmds -VersionArgs $c.Args
    if (-not $ok) { $allOk = $false }
}

Write-Host ('-' * 60)
if ($allOk) {
    Write-Host 'All required tools are present.' -ForegroundColor Green
    exit 0
}

Write-Host 'One or more tools are missing or broken. Install them and re-run.' -ForegroundColor Red
exit 1