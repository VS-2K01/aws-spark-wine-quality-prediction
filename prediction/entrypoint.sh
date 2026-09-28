#!/bin/bash
set -e

# -----------------------------
# entrypoint.sh
# Usage:
#   docker run image_name /data/ValidationDataset.csv
# -----------------------------

TEST_FILE_PATH="$1"

if [ -z "$TEST_FILE_PATH" ]; then
  echo "ERROR: No test CSV path provided."
  echo ""
  echo "Usage:"
  echo "  docker run <image> /path/inside/container/to/TestDataset.csv"
  echo ""
  echo "Example:"
  echo "  docker run --rm -v \$(pwd):/data wine-predictor:v3 /data/ValidationDataset.csv"
  exit 1
fi

echo "Test CSV File: $TEST_FILE_PATH"
echo "Model Path:    /app/winePrediction_model_ensemble"
echo ""

python3 /app/winePrediction_application.py \
  "$TEST_FILE_PATH" \
  "/app/winePrediction_model_ensemble"
