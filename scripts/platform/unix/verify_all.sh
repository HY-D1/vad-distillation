#!/bin/bash
# =============================================================================
# Automated verification script for Windows 4080 outputs on Mac
# =============================================================================
#
# Usage:
#   ./scripts/verify_all.sh [OPTIONS]
#
# Options:
#   -s, --source DIR      Source directory (default: outputs/production_4080)
#   -c, --config FILE     Config file (default: configs/production.yaml)
#   -f, --folds FOLDS     Comma-separated folds (default: F01,F02,F03,F04,F05)
#   -d, --device DEVICE   Device to use (cpu/mps)
#   -b, --batch-size N    Batch size for inference
#   -h, --help            Show this help message
#
# Examples:
#   # Verify all folds with defaults
#   ./scripts/verify_all.sh
#
#   # Verify specific folds
#   ./scripts/verify_all.sh -f "F01,F02,F03"
#
#   # Use CPU with smaller batch size
#   ./scripts/verify_all.sh -d cpu -b 4
#
# =============================================================================

set -e  # Exit on error

# Default values
SOURCE_DIR="outputs/production_4080"
CONFIG="configs/production.yaml"
FOLDS="F01 F02 F03 F04 F05"
DEVICE=""
BATCH_SIZE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source)
            SOURCE_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -f|--folds)
            FOLDS="${2//,/ }"  # Replace commas with spaces
            shift 2
            ;;
        -d|--device)
            DEVICE="--device $2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="--batch-size $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./scripts/verify_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -s, --source DIR      Source directory (default: outputs/production_4080)"
            echo "  -c, --config FILE     Config file (default: configs/production.yaml)"
            echo "  -f, --folds FOLDS     Comma-separated folds (default: F01,F02,F03,F04,F05)"
            echo "  -d, --device DEVICE   Device to use (cpu/mps)"
            echo "  -b, --batch-size N    Batch size for inference"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Setup
CHECKPOINT_DIR="$SOURCE_DIR/checkpoints"
VERIFICATION_DIR="$SOURCE_DIR/verification"
REPORT_PATH="$VERIFICATION_DIR/final_report.md"

# Create verification directory
mkdir -p "$VERIFICATION_DIR"

# Header
echo "======================================================================"
echo "Windows 4080 → Mac Verification"
echo "======================================================================"
echo "Source: $SOURCE_DIR"
echo "Config: $CONFIG"
echo "Folds:  $FOLDS"
echo "Output: $VERIFICATION_DIR"
echo "======================================================================"
echo ""

# Track results
FAILED_FOLDS=()
PASSED_FOLDS=()

# Verify each fold
for fold in $FOLDS; do
    echo "----------------------------------------------------------------------"
    echo "Verifying $fold..."
    echo "----------------------------------------------------------------------"

    CHECKPOINT="$CHECKPOINT_DIR/fold_${fold}_best.pt"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "Warning: Checkpoint not found: $CHECKPOINT"
        echo "Skipping $fold..."
        FAILED_FOLDS+=("$fold (checkpoint not found)")
        continue
    fi

    echo "Checkpoint: $CHECKPOINT"

    # Run verification
    if python scripts/core/verify_checkpoint.py \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --fold "$fold" \
        --output-dir "$VERIFICATION_DIR" \
        --compare-original \
        $DEVICE \
        $BATCH_SIZE; then

        echo "✓ $fold verification passed"
        PASSED_FOLDS+=("$fold")
    else
        echo "✗ $fold verification failed"
        FAILED_FOLDS+=("$fold (metrics differ)")
    fi

    echo ""
done

# Generate comparison report
echo "======================================================================"
echo "Generating Comparison Report"
echo "======================================================================"

python scripts/analysis/compare_verification.py \
    --windows-dir "$SOURCE_DIR/logs/" \
    --mac-dir "$VERIFICATION_DIR/" \
    --output "$REPORT_PATH"

echo ""

# Summary
echo "======================================================================"
echo "Verification Summary"
echo "======================================================================"
echo "Total folds:    $(echo $FOLDS | wc -w)"
echo "Passed:         ${#PASSED_FOLDS[@]}"
echo "Failed:         ${#FAILED_FOLDS[@]}"
echo ""

if [ ${#PASSED_FOLDS[@]} -gt 0 ]; then
    echo "Passed folds: ${PASSED_FOLDS[*]}"
fi

if [ ${#FAILED_FOLDS[@]} -gt 0 ]; then
    echo "Failed folds: ${FAILED_FOLDS[*]}"
fi

echo ""
echo "Report saved to: $REPORT_PATH"
echo "======================================================================"

# Exit with appropriate code
if [ ${#FAILED_FOLDS[@]} -eq 0 ]; then
    echo "✅ All verifications passed!"
    exit 0
else
    echo "⚠️  Some verifications failed"
    exit 1
fi
