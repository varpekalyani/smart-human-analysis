# Download YOLOv4 weights for Car Colour Detection module
# Run this script from the project root, or adjust paths as needed.

$modelsDir = Join-Path $PSScriptRoot "models"
$weightsPath = Join-Path $modelsDir "yolov4.weights"
$url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights"

if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force
}

if (Test-Path $weightsPath) {
    Write-Host "yolov4.weights already exists. Skipping download."
    exit 0
}

Write-Host "Downloading yolov4.weights (~162 MB). This may take a few minutes..."
try {
    Invoke-WebRequest -Uri $url -OutFile $weightsPath -UseBasicParsing
    Write-Host "Download complete: $weightsPath"
} catch {
    Write-Host "Error: $_"
    Write-Host ""
    Write-Host "Manual download:"
    Write-Host "  1. Visit: $url"
    Write-Host "  2. Download yolov4.weights"
    Write-Host "  3. Place it in: $modelsDir"
    exit 1
}
