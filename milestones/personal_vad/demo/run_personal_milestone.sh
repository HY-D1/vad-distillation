#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] Coverage audit + conditional ground-truth claims (F01 test)"
./.venv/bin/python milestones/personal_vad/scripts/evaluate_against_ground_truth.py \
  --fold F01 \
  --eval-split test \
  --max-utterances 30 \
  --output-dir outputs/personal_vad/ground_truth_eval

echo
echo "[2/6] Split coverage summary"
cat outputs/personal_vad/ground_truth_eval/split_coverage_summary.json

echo
echo "[3/6] Claim status"
if [[ -f outputs/personal_vad/ground_truth_eval/ground_truth_claims_blocked.txt ]]; then
  cat outputs/personal_vad/ground_truth_eval/ground_truth_claims_blocked.txt
else
  cat outputs/personal_vad/ground_truth_eval/summary.txt
fi

echo
echo "[4/6] Live comparison check from personal comparison output"
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/personal_vad/comparison_live_ready/summary.json")
data = json.loads(path.read_text())
rows = data["comparison"]["utterance_comparisons"]
best = max(rows, key=lambda x: x["correlation"])
worst = min(rows, key=lambda x: x["correlation"])
print("Live comparison check")
print(f"Best agreement: {best['utt_id']} | corr = {best['correlation']:.3f} | mse = {best['mse']:.4f}")
print(f"Worst agreement: {worst['utt_id']} | corr = {worst['correlation']:.3f} | mse = {worst['mse']:.4f}")
PY

echo
echo "[5/6] Existing milestone artifacts"
cat outputs/personal_vad/comparison_live_ready/summary.txt
cat outputs/personal_vad/energy_tuning_live_ready/summary.txt
ls -lh outputs/personal_vad/qualitative_plots_live_ready/

echo
echo "[6/6] Secondary analyses (non-blocking)"
./.venv/bin/python milestones/personal_vad/scripts/run_model_selection.py \
  --input outputs/personal_vad/ground_truth_eval/results.csv \
  --output-dir outputs/personal_vad/model_selection || true
./.venv/bin/python milestones/personal_vad/scripts/error_analysis.py \
  --input outputs/personal_vad/ground_truth_eval/results.csv \
  --output-dir outputs/personal_vad/error_analysis || true

echo
echo "Done. Primary demo artifacts:"
echo "  - outputs/personal_vad/ground_truth_eval/split_coverage_summary.json"
echo "  - outputs/personal_vad/ground_truth_eval/summary.txt (or ground_truth_claims_blocked.txt)"
echo "  - outputs/personal_vad/comparison_live_ready/summary.txt"
echo "  - outputs/personal_vad/energy_tuning_live_ready/summary.txt"
echo "  - outputs/personal_vad/qualitative_plots_live_ready/F01_Session1_0002.png"
