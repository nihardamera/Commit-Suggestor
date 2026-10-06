#!/bin/sh
# Download the trained LoRA adapter from the GitHub release into ./adapters.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
RELEASE=https://github.com/nihardamera/Commit-Suggestor/releases/download/v1.0
mkdir -p "$HERE/adapters"
for f in adapters.safetensors adapter_config.json; do
  curl -fsSL -o "$HERE/adapters/$f" "$RELEASE/$f"
done
echo "adapter saved to $HERE/adapters"
