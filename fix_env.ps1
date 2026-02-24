# Fix NumPy + OpenCV in the PROJECT VENV (run with app closed)
# Run: .\fix_env.ps1

Set-Location $PSScriptRoot
$pip = ".\venv\Scripts\pip.exe"
if (-not (Test-Path $pip)) {
    Write-Host "Creating venv first..." -ForegroundColor Yellow
    python -m venv venv
}
Write-Host "Uninstalling opencv and numpy in venv..." -ForegroundColor Yellow
& $pip uninstall -y opencv-python opencv-contrib-python numpy 2>$null
Write-Host "Installing NumPy 1.x (required for hand detection)..." -ForegroundColor Cyan
& $pip install "numpy>=1.26.0,<2"
Write-Host "Installing OpenCV 4.9..." -ForegroundColor Cyan
& $pip install opencv-python==4.9.0.80
Write-Host "Done. Run the app with: .\run.ps1  or  .\run.bat" -ForegroundColor Green
