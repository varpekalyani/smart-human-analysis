#!/bin/bash
# Download YOLOv4 weights for Car Colour Detection module
# Run: chmod +x download_yolov4.sh && ./download_yolov4.sh

MODELS_DIR="$(dirname "$0")/models"
WEIGHTS_PATH="$MODELS_DIR/yolov4.weights"
URL="https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights"

mkdir -p "$MODELS_DIR"

if [ -f "$WEIGHTS_PATH" ]; then
    echo "yolov4.weights already exists. Skipping download."
    exit 0
fi

echo "Downloading yolov4.weights (~162 MB). This may take a few minutes..."
if command -v curl &> /dev/null; then
    curl -L -o "$WEIGHTS_PATH" "$URL" || { echo "Download failed."; exit 1; }
elif command -v wget &> /dev/null; then
    wget -O "$WEIGHTS_PATH" "$URL" || { echo "Download failed."; exit 1; }
else
    echo "Install curl or wget to download. Or manually download from:"
    echo "  $URL"
    echo "  Place file in: $MODELS_DIR"
    exit 1
fi

echo "Download complete: $WEIGHTS_PATH"
