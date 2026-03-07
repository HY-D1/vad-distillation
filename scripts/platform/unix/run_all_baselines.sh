#!/bin/bash
# Run all baseline methods for comparison

set -e

MANIFEST="manifests/torgo_pilot.csv"
OUTPUT_BASE="outputs/baselines"

echo "=========================================="
echo "Running VAD Baselines"
echo "=========================================="

# Create output directories
mkdir -p "$OUTPUT_BASE/energy/frame_probs"
mkdir -p "$OUTPUT_BASE/silero/frame_probs"
mkdir -p "$OUTPUT_BASE/speechbrain/frame_probs"

# 1. Energy Baseline
echo ""
echo "1. Running Energy baseline..."
python scripts/core/run_baseline.py \
    --method energy \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_BASE/energy/"

# 2. Silero Baseline (teacher)
echo ""
echo "2. Running Silero baseline..."
python scripts/core/run_baseline.py \
    --method silero \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_BASE/silero/"

# 3. SpeechBrain Baseline (optional - requires speechbrain)
echo ""
echo "3. Running SpeechBrain baseline..."
if python -c "import speechbrain" 2>/dev/null; then
    python scripts/core/run_baseline.py \
        --method speechbrain \
        --manifest "$MANIFEST" \
        --output-dir "$OUTPUT_BASE/speechbrain/"
else
    echo "   SpeechBrain not installed, skipping."
fi

echo ""
echo "=========================================="
echo "Baselines complete!"
echo "=========================================="
