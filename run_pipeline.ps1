# One-click runner for the standard pipeline on Windows (PowerShell)
# Uses the workspace venv if present

$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $workspace "venv/Scripts/python.exe"
if (-Not (Test-Path $python)) {
  Write-Host "venv python not found; using system python"
  $python = "python"
}

Write-Host "[1/2] Fetch (preset standard)..." -ForegroundColor Cyan
& $python (Join-Path $workspace "fetch_history_pro.py") --preset standard

Write-Host "[2/2] Features (preset standard)..." -ForegroundColor Cyan
& $python (Join-Path $workspace "compute_features.py") --preset standard

Write-Host "Done." -ForegroundColor Green
