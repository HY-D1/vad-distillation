#!/bin/bash
#
# Quick verification script for Windows RTX 4080 training results on Mac
# 
# Loops through all 15 TORGO folds and prints key checkpoint and metrics info
# for quick validation without generating a full report.
#
# Usage:
#     chmod +x scripts/analysis/quick_verify.sh
#     ./scripts/analysis/quick_verify.sh [results_dir]
#
#     # Custom results directory
#     ./scripts/analysis/quick_verify.sh outputs/production_cuda
#
#     # Quick check of specific folds only
#     ./scripts/analysis/quick_verify.sh outputs/production_cuda F01 M01 FC01
#
# Expected directory structure:
#     outputs/production_cuda/
#     ├── checkpoints/
#     │   ├── fold_F01_best.pt
#     │   ├── fold_F02_best.pt
#     │   └── ...
#     └── logs/
#         ├── fold_F01_summary.json
#         ├── fold_F02_summary.json
#         └── ...

set -e

# Colors for output (if terminal supports it)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default results directory
RESULTS_DIR="${1:-outputs/production_cuda}"
shift || true  # Remove first arg if it exists, otherwise continue

# All 15 TORGO speaker folds
ALL_FOLDS=("F01" "F03" "F04" "M01" "M02" "M03" "M04" "M05" "FC01" "FC02" "FC03" "MC01" "MC02" "MC03" "MC04")

# Use provided folds or default to all
if [ $# -gt 0 ]; then
    FOLDS=("$@")
else
    FOLDS=("${ALL_FOLDS[@]}")
fi

CHECKPOINTS_DIR="$RESULTS_DIR/checkpoints"
LOGS_DIR="$RESULTS_DIR/logs"

echo "================================================================================"
echo "Quick Verification of Windows RTX 4080 Training Results"
echo "================================================================================"
echo ""
echo "Results directory:     $RESULTS_DIR"
echo "Checkpoints directory: $CHECKPOINTS_DIR"
echo "Logs directory:        $LOGS_DIR"
echo ""
echo "Verifying ${#FOLDS[@]} folds: ${FOLDS[*]}"
echo ""
echo "================================================================================"

# Check if directories exist
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}Error: Results directory not found: $RESULTS_DIR${NC}"
    exit 1
fi

if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo -e "${YELLOW}Warning: Checkpoints directory not found: $CHECKPOINTS_DIR${NC}"
fi

if [ ! -d "$LOGS_DIR" ]; then
    echo -e "${YELLOW}Warning: Logs directory not found: $LOGS_DIR${NC}"
fi

# Track statistics
TOTAL=${#FOLDS[@]}
FOUND=0
MISSING=0
WITH_ERRORS=0

echo ""
echo "Per-Fold Verification:"
echo "--------------------------------------------------------------------------------"
printf "%-8s %-12s %-10s %-10s %-10s %-10s %-10s\n" "Fold" "Checkpoint" "Epoch" "Best AUC" "Test AUC" "Test F1" "Status"
echo "--------------------------------------------------------------------------------"

for FOLD in "${FOLDS[@]}"; do
    CHECKPOINT="$CHECKPOINTS_DIR/fold_${FOLD}_best.pt"
    SUMMARY="$LOGS_DIR/fold_${FOLD}_summary.json"
    
    CKPT_STATUS="MISSING"
    EPOCH="N/A"
    BEST_AUC="N/A"
    TEST_AUC="N/A"
    TEST_F1="N/A"
    STATUS="FAIL"
    
    # Check checkpoint
    if [ -f "$CHECKPOINT" ]; then
        CKPT_STATUS="OK"
        ((FOUND++)) || true
        
        # Try to extract info using Python if available
        if command -v python3 &> /dev/null; then
            PYTHON_CHECK=$(python3 << EOF
import torch
try:
    ckpt = torch.load('$CHECKPOINT', map_location='cpu')
    epoch = ckpt.get('epoch', 'N/A')
    best_auc = ckpt.get('best_auc', 'N/A')
    has_model = 'model_state_dict' in ckpt
    print(f"{epoch},{best_auc},{has_model}")
except Exception as e:
    print(f"ERROR,{e},False")
EOF
)
            if [[ "$PYTHON_CHECK" == ERROR* ]]; then
                CKPT_STATUS="ERROR"
                ((WITH_ERRORS++)) || true
            else
                IFS=',' read -r EPOCH BEST_AUC HAS_MODEL <<< "$PYTHON_CHECK"
                if [ "$HAS_MODEL" != "True" ]; then
                    CKPT_STATUS="NO_MODEL"
                    ((WITH_ERRORS++)) || true
                fi
            fi
        else
            # Fallback: just check file exists and is non-empty
            if [ ! -s "$CHECKPOINT" ]; then
                CKPT_STATUS="EMPTY"
                ((WITH_ERRORS++)) || true
            fi
        fi
    else
        ((MISSING++)) || true
    fi
    
    # Check summary
    if [ -f "$SUMMARY" ]; then
        if command -v python3 &> /dev/null; then
            METRICS=$(python3 << EOF
import json
try:
    with open('$SUMMARY', 'r') as f:
        data = json.load(f)
    test_metrics = data.get('test_metrics', {})
    auc = test_metrics.get('auc', 'N/A')
    f1 = test_metrics.get('f1', 'N/A')
    print(f"{auc},{f1}")
except Exception as e:
    print("ERROR,ERROR")
EOF
)
            IFS=',' read -r TEST_AUC TEST_F1 <<< "$METRICS"
            if [ "$TEST_AUC" != "ERROR" ] && [ "$TEST_AUC" != "N/A" ]; then
                # Check if AUC is reasonable (> 0.5)
                if (( $(echo "$TEST_AUC > 0.5" | bc -l 2>/dev/null || echo "0") )); then
                    STATUS="PASS"
                else
                    STATUS="LOW_AUC"
                fi
            fi
        fi
    fi
    
    # Format output
    if [ "$STATUS" == "PASS" ]; then
        STATUS_COLOR="${GREEN}PASS${NC}"
    elif [ "$CKPT_STATUS" == "MISSING" ]; then
        STATUS_COLOR="${RED}MISSING${NC}"
    else
        STATUS_COLOR="${YELLOW}$STATUS${NC}"
    fi
    
    printf "%-8s %-12s %-10s %-10s %-10s %-10s " \
        "$FOLD" "$CKPT_STATUS" "$EPOCH" "$BEST_AUC" "$TEST_AUC" "$TEST_F1"
    echo -e "$STATUS_COLOR"
done

echo "--------------------------------------------------------------------------------"
echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo ""
echo "Total folds checked: $TOTAL"
echo "Checkpoints found:   $FOUND"
echo "Checkpoints missing: $MISSING"
echo "With errors:         $WITH_ERRORS"
echo ""

if [ $MISSING -eq 0 ] && [ $WITH_ERRORS -eq 0 ]; then
    echo -e "${GREEN}All folds verified successfully!${NC}"
    echo ""
    echo "================================================================================"
    exit 0
else
    echo -e "${YELLOW}Some folds have issues (missing or errors).${NC}"
    echo ""
    echo "Missing checkpoints: $MISSING"
    echo "Checkpoints with errors: $WITH_ERRORS"
    echo ""
    echo "================================================================================"
    exit 1
fi
