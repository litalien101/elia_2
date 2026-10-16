$ErrorActionPreference = "Stop"

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pythonExe = if ($pythonCommand) { $pythonCommand.Source } else { $null }

if (-not $pythonExe -or $pythonExe -like "$env:LOCALAPPDATA\Microsoft\WindowsApps\*") {
    $pythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
    $candidate = Get-ChildItem $pythonRoot -Directory -Filter "Python*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "python.exe" } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1
    if ($candidate) {
        $pythonExe = $candidate
    }
}

if (-not $pythonExe) {
    throw "Python 3.11 or newer is required. Install Python and reopen PowerShell."
}

$versionText = & $pythonExe --version
if ($versionText -notmatch "Python (\d+)\.(\d+)") {
    throw "Could not determine Python version from '$versionText'."
}
if ([int]$Matches[1] -lt 3 -or ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -lt 11)) {
    throw "Python 3.11 or newer is required. Found $versionText."
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
Set-Location $PSScriptRoot
& $pythonExe -m elia
if ($LASTEXITCODE -ne 0) {
    throw "ELIA stopped with exit code $LASTEXITCODE."
}
