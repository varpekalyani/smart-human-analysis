# Run the app with the project venv (required for hand detection - no yellow banner)
Set-Location $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating venv and installing dependencies..." -ForegroundColor Yellow
    python -m venv venv
    & (Join-Path $PSScriptRoot "venv\Scripts\pip.exe") install -r requirements.txt
}
Write-Host "Starting app (use venv - hand detection will work)..." -ForegroundColor Green
& $venvPython app.py
