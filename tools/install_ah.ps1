# Register ah launcher: user PATH, AH_HOME, optional .ah file association.
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File tools\install_ah.ps1
#   powershell -ExecutionPolicy Bypass -File tools\install_ah.ps1 -AssociateAhFiles
param(
    [switch]$AssociateAhFiles
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AhCmd = Join-Path $Root "ah.cmd"

if (-not (Test-Path $AhCmd)) {
    throw "ah.cmd not found: $AhCmd"
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$segments = @()
if ($userPath) {
    $segments = $userPath -split ";" | Where-Object { $_ -and ($_ -ne $Root) }
}
$segments = @($Root) + $segments
$newPath = ($segments | Select-Object -Unique) -join ";"
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")
[Environment]::SetEnvironmentVariable("AH_HOME", $Root, "User")

$env:Path = "$Root;$env:Path"
$env:AH_HOME = $Root

Write-Host "Added to user PATH: $Root" -ForegroundColor Green
Write-Host "Set AH_HOME=$Root" -ForegroundColor Green
Write-Host ""
Write-Host "Open a new terminal, then run:" -ForegroundColor Cyan
Write-Host "  ah path\to\script.ah"
Write-Host ""
Write-Host "Sessions are created at: <launch-dir>\sessions\<timestamp>_...\"

if ($AssociateAhFiles) {
    $handler = "`"$AhCmd`" `"%1`" %*"
    cmd /c "assoc .ah=AnthillScript" 2>$null
    cmd /c "ftype AnthillScript=$handler"
    Write-Host ""
    Write-Host "Registered .ah -> ah.cmd (double-click or run script.ah directly)" -ForegroundColor Green
}
