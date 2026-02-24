# Download YOLOv4-tiny weights for Car Colour Detection (smaller, faster)
$modelsDir = Join-Path $PSScriptRoot "models"
$weightsPath = Join-Path $modelsDir "yolov4-tiny.weights"
$url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"

if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir -Force }
if (Test-Path $weightsPath) { Write-Host "yolov4-tiny.weights exists. Skipping."; exit 0 }

Write-Host "Downloading yolov4-tiny.weights (~6 MB)..."
try {
    Invoke-WebRequest -Uri $url -OutFile $weightsPath -UseBasicParsing
    Write-Host "Done: $weightsPath"
} catch {
    Write-Host "Error: $_"
    Write-Host "Manual: $url -> $modelsDir"
    exit 1
}
