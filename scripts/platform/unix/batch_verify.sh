#!/bin/bash
#
# Batch verification script for Mac to validate Windows RTX 4080 training results.
#
# This script loops through all 15 folds and verifies checkpoint integrity
# and metrics are within expected ranges.
#
# Usage:
#   chmod +x scripts/batch_verify.sh
#   ./scripts/batch_verify.sh outputs/production_cuda/
#
#   # With custom thresholds
#   ./scripts/batch_verify.sh outputs/production_cuda/ 0.80 0.65 0.25
#
#   # Verify specific folds only
#   FOLDS="F01 F02 F03" ./scripts/batch_verify.sh outputs/production_cuda/

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
RESULTS_DIR="${1:-outputs/production_cuda/}"
MIN_AUC="${2:-0.85}"
MIN_F1="${3:-0.70}"
MAX_MISS_RATE="${4:-0.20}"

CHECKPOINTS_DIR="${RESULTS_DIR}/checkpoints"
LOGS_DIR="${RESULTS_DIR}/logs"

# Default folds (15 speakers in TORGO)
DEFAULT_FOLDS=("F01" "F02" "F03" "F04" "F05" "M01" "M02" "M03" "M04" "M05" "M06" "M07" "M08" "M09" "M10")
FOLDS=(${FOLDS:-${DEFAULT_FOLDS[@]}})

echo "========================================"
echo "Windows RTX 4080 Results Verification"
echo "========================================"
echo ""
echo "Results directory:  ${RESULTS_DIR}"
echo "Checkpoints dir:    ${CHECKPOINTS_DIR}"
echo "Logs dir:           ${LOGS_DIR}"
echo ""
echo "Thresholds:"
echo "  Min AUC:          ${MIN_AUC}"
echo "  Min F1:           ${MIN_F1}"
echo "  Max Miss Rate:    ${MAX_MISS_RATE}"
echo ""
echo "Folds to verify:    ${#FOLDS[@]}"
echo "  ${FOLDS[@]}"
echo ""
echo "========================================"
echo ""

# Check if directories exist
if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo -e "${RED}Error: Results directory not found: ${RESULTS_DIR}${NC}"
    exit 1
fi

if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
    echo -e "${YELLOW}Warning: Checkpoints directory not found: ${CHECKPOINTS_DIR}${NC}"
fi

if [[ ! -d "${LOGS_DIR}" ]]; then
    echo -e "${YELLOW}Warning: Logs directory not found: ${LOGS_DIR}${NC}"
fi

# Counters
TOTAL=0
PASSED=0
WARNINGS=0
FAILED=0
MISSING=0

# Arrays to store results
declare -a RESULTS_AUC
declare -a RESULTS_F1
declare -a RESULTS_MISS
declare -a RESULTS_STATUS
declare -a RESULTS_ISSUES

echo "Starting verification..."
echo ""

for FOLD in "${FOLDS[@]}"; do
    TOTAL=$((TOTAL + 1))
    
    echo -n "Verifying ${FOLD}... "
    
    CHECKPOINT="${CHECKPOINTS_DIR}/fold_${FOLD}_best.pt"
    SUMMARY="${LOGS_DIR}/fold_${FOLD}_summary.json"
    
    ISSUES=""
    STATUS="PASS"
    
    # Check checkpoint exists
    if [[ ! -f "${CHECKPOINT}" ]]; then
        STATUS="MISSING"
        ISSUES="checkpoint missing; "
    fi
    
    # Check summary exists and extract metrics
    if [[ -f "${SUMMARY}" ]]; then
        # Extract metrics using Python (more reliable than jq)
        if command -v python3 &> /dev/null; then
            METRICS=$(python3 << EOF
import json
import sys
try:
    with open('${SUMMARY}', 'r') as f:
        data = json.load(f)
        test = data.get('test_metrics', {})
        best = data.get('best_val_auc', 0)
        print(f"{test.get('auc', 0):.4f},{test.get('f1', 0):.4f},{test.get('miss_rate', 1):.4f},{test.get('false_alarm_rate', 1):.4f},{best:.4f}")
except Exception as e:
    print("ERROR")
EOF
)
            
            if [[ "${METRICS}" == "ERROR" ]]; then
                STATUS="FAIL"
                ISSUES+="summary parse error; "
                AUC="0"
                F1="0"
                MISS="1"
            else
                IFS=',' read -r AUC F1 MISS FAR BEST <<< "${METRICS}"
                
                # Check thresholds
                if (( $(echo "${AUC} < ${MIN_AUC}" | bc -l) )); then
                    if [[ "${STATUS}" == "PASS" ]]; then
                        STATUS="WARN"
                    fi
                    ISSUES+="AUC ${AUC} < ${MIN_AUC}; "
                fi
                
                if (( $(echo "${F1} < ${MIN_F1}" | bc -l) )); then
                    if [[ "${STATUS}" == "PASS" ]]; then
                        STATUS="WARN"
                    fi
                    ISSUES+="F1 ${F1} < ${MIN_F1}; "
                fi
                
                if (( $(echo "${MISS} > ${MAX_MISS_RATE}" | bc -l) )); then
                    if [[ "${STATUS}" == "PASS" ]]; then
                        STATUS="WARN"
                    fi
                    ISSUES+="MissRate ${MISS} > ${MAX_MISS_RATE}; "
                fi
            fi
        else
            # Fallback: just check file exists
            AUC="?"
            F1="?"
            MISS="?"
        fi
    else
        if [[ "${STATUS}" == "PASS" ]]; then
            STATUS="MISSING"
        fi
        ISSUES+="summary missing; "
        AUC="N/A"
        F1="N/A"
        MISS="N/A"
    fi
    
    # Update counters
    case "${STATUS}" in
        "PASS")
            PASSED=$((PASSED + 1))
            echo -e "${GREEN}✓ PASS${NC} (AUC: ${AUC}, F1: ${F1}, Miss: ${MISS})"
            ;;
        "WARN")
            WARNINGS=$((WARNINGS + 1))
            echo -e "${YELLOW}⚠ WARN${NC} (AUC: ${AUC}, F1: ${F1}, Miss: ${MISS})"
            echo "         Issues: ${ISSUES}"
            ;;
        "MISSING")
            MISSING=$((MISSING + 1))
            echo -e "${RED}✗ MISS${NC} - ${ISSUES}"
            ;;
        *)
            FAILED=$((FAILED + 1))
            echo -e "${RED}✗ FAIL${NC} - ${ISSUES}"
            ;;
    esac
    
    # Store results for summary
    RESULTS_AUC+=("${AUC}")
    RESULTS_F1+=("${F1}")
    RESULTS_MISS+=("${MISS}")
    RESULTS_STATUS+=("${STATUS}")
    RESULTS_ISSUES+=("${ISSUES}")
done

echo ""
echo "========================================"
echo "Verification Summary"
echo "========================================"
echo ""
echo -e "Total folds:     ${BOLD}${TOTAL}${NC}"
echo -e "  ${GREEN}✓ Passed:      ${PASSED}${NC}"
echo -e "  ${YELLOW}⚠ Warnings:    ${WARNINGS}${NC}"
echo -e "  ${RED}✗ Failed:      ${FAILED}${NC}"
echo -e "  ${RED}✗ Missing:     ${MISSING}${NC}"
echo ""

# Print detailed table
echo "========================================"
echo "Detailed Results"
echo "========================================"
echo ""
printf "%-8s %-10s %-8s %-8s %-8s %s\n" "Fold" "Status" "AUC" "F1" "Miss%" "Issues"
echo "----------------------------------------"

for i in "${!FOLDS[@]}"; do
    FOLD="${FOLDS[$i]}"
    STATUS="${RESULTS_STATUS[$i]}"
    AUC="${RESULTS_AUC[$i]}"
    F1="${RESULTS_F1[$i]}"
    MISS="${RESULTS_MISS[$i]}"
    ISSUES="${RESULTS_ISSUES[$i]}"
    
    # Color status
    case "${STATUS}" in
        "PASS") STATUS_COLOR="${GREEN}${STATUS}${NC}" ;;
        "WARN") STATUS_COLOR="${YELLOW}${STATUS}${NC}" ;;
        *) STATUS_COLOR="${RED}${STATUS}${NC}" ;;
    esac
    
    printf "%-8s %-10b %-8s %-8s %-8s %s\n" "${FOLD}" "${STATUS_COLOR}" "${AUC}" "${F1}" "${MISS}" "${ISSUES}"
done

echo ""
echo "========================================"

# Final verdict
if [[ ${PASSED} -eq ${TOTAL} ]]; then
    echo -e "${GREEN}✅ ALL FOLDS PASSED VERIFICATION${NC}"
    echo "========================================"
    exit 0
elif [[ ${FAILED} -eq 0 && ${MISSING} -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  ${PASSED}/${TOTAL} FOLDS PASSED, ${WARNINGS} WITH WARNINGS${NC}"
    echo "========================================"
    exit 0
else
    echo -e "${RED}✗ ${PASSED}/${TOTAL} FOLDS PASSED${NC}"
    echo "========================================"
    exit 1
fi
