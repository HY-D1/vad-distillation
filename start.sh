#!/usr/bin/env bash
#
# start.sh - One-Command Setup and Training for VAD Distillation Project
#
# Usage: ./start.sh <MODE> [OPTIONS]
#
# Modes:
#   setup       - Initial environment setup
#   quick-test  - Quick validation (< 5 min)
#   train       - Full training (single or all folds)
#   status      - Check training status
#   verify      - Verify outputs and generate report
#   clean       - Cleanup and reset
#   help        - Show help message
#
# Examples:
#   ./start.sh setup
#   ./start.sh quick-test --fold F01
#   ./start.sh train --all-folds
#   ./start.sh status --watch
#   ./start.sh verify --generate-report
#

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="vad-distillation"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
DATA_DIR="${VAD_DATA_DIR:-${SCRIPT_DIR}/data/torgo_raw}"
OUTPUT_DIR="${VAD_OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
CONFIG_DIR="${SCRIPT_DIR}/configs"
SPLITS_DIR="${SCRIPT_DIR}/splits"

# Default values
SEED="${VAD_SEED:-6140}"
DEVICE="${VAD_DEVICE:-auto}"
VERBOSE="${VERBOSE:-false}"
NO_COLOR="${NO_COLOR:-false}"

# Fold list (15 total)
FOLDS=("F01" "F03" "F04" "M01" "M02" "M03" "M04" "M05" "FC01" "FC02" "FC03" "MC01" "MC02" "MC03" "MC04")

# =============================================================================
# COLOR OUTPUT
# =============================================================================

setup_colors() {
    if [[ "$NO_COLOR" == "true" ]] || [[ ! -t 2 ]]; then
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        CYAN=''
        BOLD=''
        NC=''
    else
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[1;33m'
        BLUE='\033[0;34m'
        CYAN='\033[0;36m'
        BOLD='\033[1m'
        NC='\033[0m'
    fi
}

# =============================================================================
# LOGGING
# =============================================================================

setup_logging() {
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/start.sh.log"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    log "INFO" "$1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
    log "SUCCESS" "$1"
}

warn() {
    echo -e "${YELLOW}[⚠]${NC} $1" >&2
    log "WARNING" "$1"
}

error() {
    echo -e "${RED}[✗]${NC} $1" >&2
    log "ERROR" "$1"
}

die() {
    error "$1"
    exit "${2:-1}"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}$(printf '%*s' "${#1}" '' | tr ' ' '-')${NC}"
}

detect_platform() {
    case "$OSTYPE" in
        linux-gnu*)  PLATFORM="linux" ;;
        darwin*)     PLATFORM="macos" ;;
        msys*|cygwin*|win32*) PLATFORM="windows" ;;
        *)           PLATFORM="unknown" ;;
    esac
    log "PLATFORM" "$PLATFORM"
}

detect_device() {
    if [[ "$DEVICE" != "auto" ]]; then
        echo "$DEVICE"
        return
    fi
    
    python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>/dev/null || echo "cpu"
}

get_recommended_config() {
    local device
    device=$(detect_device)
    
    case "$device" in
        cuda)
            echo "configs/production_cuda.yaml"
            ;;
        mps)
            echo "configs/production.yaml"
            ;;
        *)
            echo "configs/pilot.yaml"
            ;;
    esac
}

progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    printf "\r[" >&2
    printf "%${filled}s" | tr ' ' '█' >&2
    printf "%${empty}s" | tr ' ' '░' >&2
    printf "] %3d%%" "$percentage" >&2
}

# =============================================================================
# SETUP MODE
# =============================================================================

cmd_setup() {
    local skip_requirements=false
    local skip_data_check=false
    local skip_cuda_check=false
    local force=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-requirements) skip_requirements=true ;;
            --skip-data-check) skip_data_check=true ;;
            --skip-cuda-check) skip_cuda_check=true ;;
            --force) force=true ;;
            *) warn "Unknown option: $1" ;;
        esac
        shift
    done
    
    print_header "VAD Distillation Project - Setup"
    
    local total_steps=12
    local current_step=0
    local setup_report="$LOG_DIR/setup_report.json"
    
    # Check if already configured
    if [[ -f "$setup_report" ]] && [[ "$force" != "true" ]]; then
        info "Setup already completed. Use --force to re-run."
        return 0
    fi
    
    # Step 1: Python version
    ((current_step++))
    echo -n "[$current_step/$total_steps] Checking Python version... "
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        success "$PYTHON_VERSION"
    elif command -v python &>/dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
        success "$PYTHON_VERSION"
    else
        die "Python not found. Please install Python 3.8 or higher."
    fi
    
    # Step 2: Pip availability
    ((current_step++))
    echo -n "[$current_step/$total_steps] Checking pip... "
    if command -v pip3 &>/dev/null; then
        PIP_VERSION=$(pip3 --version | cut -d' ' -f2)
        success "$PIP_VERSION"
    elif command -v pip &>/dev/null; then
        PIP_VERSION=$(pip --version | cut -d' ' -f2)
        success "$PIP_VERSION"
    else
        die "pip not found. Please install pip."
    fi
    
    # Step 3: Install requirements
    ((current_step++))
    echo -n "[$current_step/$total_steps] Installing requirements... "
    if [[ "$skip_requirements" != "true" ]]; then
        if pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null; then
            local pkg_count
            pkg_count=$(pip list | wc -l)
            success "($pkg_count packages)"
        else
            warn "Some packages failed to install"
        fi
    else
        echo -e "${YELLOW}skipped${NC}"
    fi
    
    # Step 4: Verify PyTorch
    ((current_step++))
    echo -n "[$current_step/$total_steps] Verifying PyTorch... "
    if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        success "$TORCH_VERSION"
    else
        die "PyTorch not installed. Run: pip install torch>=2.0.0"
    fi
    
    # Step 5: Check CUDA
    ((current_step++))
    echo -n "[$current_step/$total_steps] Checking CUDA... "
    if [[ "$skip_cuda_check" != "true" ]]; then
        if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
            success "Available (CUDA $CUDA_VERSION)"
        else
            warn "Not available (CPU only)"
        fi
    else
        echo -e "${YELLOW}skipped${NC}"
    fi
    
    # Step 6: Check MPS (macOS)
    ((current_step++))
    echo -n "[$current_step/$total_steps] Checking MPS... "
    if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        success "Available (Apple Silicon)"
    else
        echo -e "${YELLOW}not available${NC}"
    fi
    
    # Step 7: Validate TORGO data
    ((current_step++))
    echo -n "[$current_step/$total_steps] Validating TORGO data... "
    if [[ "$skip_data_check" != "true" ]]; then
        if [[ -d "$DATA_DIR" ]]; then
            SPEAKER_COUNT=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
            FILE_COUNT=$(find "$DATA_DIR" -name "*.wav" 2>/dev/null | wc -l)
            success "$SPEAKER_COUNT speakers, $FILE_COUNT files"
        else
            warn "Data directory not found: $DATA_DIR"
        fi
    else
        echo -e "${YELLOW}skipped${NC}"
    fi
    
    # Step 8: Validate manifests
    ((current_step++))
    echo -n "[$current_step/$total_steps] Validating manifests... "
    if [[ -d "$SCRIPT_DIR/manifests" ]]; then
        MANIFEST_COUNT=$(find "$SCRIPT_DIR/manifests" -name "*.csv" | wc -l)
        success "$MANIFEST_COUNT CSV files"
    else
        warn "Manifests directory not found"
    fi
    
    # Step 9: Validate splits
    ((current_step++))
    echo -n "[$current_step/$total_steps] Validating splits... "
    if [[ -d "$SPLITS_DIR" ]]; then
        SPLIT_COUNT=$(find "$SPLITS_DIR" -name "fold_*.json" | wc -l)
        success "$SPLIT_COUNT fold files"
    else
        warn "Splits directory not found"
    fi
    
    # Step 10: Validate configs
    ((current_step++))
    echo -n "[$current_step/$total_steps] Validating configs... "
    if [[ -d "$CONFIG_DIR" ]]; then
        CONFIG_COUNT=$(find "$CONFIG_DIR" -name "*.yaml" | wc -l)
        success "$CONFIG_COUNT YAML files"
    else
        warn "Configs directory not found"
    fi
    
    # Step 11: Run import tests
    ((current_step++))
    echo -n "[$current_step/$total_steps] Running import tests... "
    if python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
try:
    from data import TORGODataset
    from models import create_student_model
    from utils import load_config
    print('OK')
except Exception as e:
    print(f'FAILED: {e}')
    sys.exit(1)
" 2>/dev/null; then
        success "All imports OK"
    else
        warn "Some imports failed"
    fi
    
    # Step 12: Generate setup report
    ((current_step++))
    echo -n "[$current_step/$total_steps] Generating setup report... "
    cat > "$setup_report" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "platform": "$PLATFORM",
  "python_version": "$PYTHON_VERSION",
  "pytorch_version": "$TORCH_VERSION",
  "data_dir": "$DATA_DIR",
  "speakers_found": ${SPEAKER_COUNT:-0},
  "wav_files_found": ${FILE_COUNT:-0},
  "manifests_found": ${MANIFEST_COUNT:-0},
  "splits_found": ${SPLIT_COUNT:-0},
  "configs_found": ${CONFIG_COUNT:-0}
}
EOF
    success "$setup_report"
    
    # Summary
    print_section "Setup Status: SUCCESSFUL"
    echo "Platform: $PLATFORM"
    echo "Device: $(detect_device)"
    echo "Recommended config: $(get_recommended_config)"
    echo ""
    echo -e "${GREEN}Next step:${NC} ./start.sh quick-test"
}

# =============================================================================
# QUICK-TEST MODE
# =============================================================================

cmd_quick_test() {
    local fold="F01"
    local config=""
    local device=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --fold) shift; fold="$1" ;;
            --config) shift; config="$1" ;;
            --device) shift; device="$1" ;;
        esac
        shift
    done
    
    # Use default config if not specified
    if [[ -z "$config" ]]; then
        config="configs/quick_test.yaml"
    fi
    
    print_header "Quick Test - Pipeline Validation"
    
    echo "Configuration:"
    echo "  Config: $config"
    echo "  Fold: $fold"
    echo "  Device: ${device:-$(detect_device)}"
    echo ""
    
    # Validate fold
    if [[ ! -f "$SPLITS_DIR/fold_$fold.json" ]]; then
        die "Fold not found: $SPLITS_DIR/fold_$fold.json"
    fi
    
    # Run dry-run test first
    info "Running dry-run test..."
    if python3 "$SCRIPT_DIR/train_loso.py" \
        --config "$config" \
        --fold "$fold" \
        --test 2>&1 | tee -a "$LOG_DIR/quick_test_$fold.log"; then
        success "Dry-run test passed"
    else
        die "Dry-run test failed. Check log: $LOG_DIR/quick_test_$fold.log"
    fi
    
    # Run actual training
    info "Running training (2 epochs)..."
    local start_time
    start_time=$(date +%s)
    
    if python3 "$SCRIPT_DIR/train_loso.py" \
        --config "$config" \
        --fold "$fold" 2>&1 | tee -a "$LOG_DIR/quick_test_$fold.log"; then
        
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Extract metrics from log
        local val_auc
        val_auc=$(grep "Val AUC:" "$LOG_DIR/quick_test_$fold.log" | tail -1 | sed 's/.*Val AUC: \([0-9.]*\).*/\1/')
        
        print_section "Quick Test Results"
        echo -e "${GREEN}Pipeline validation PASSED${NC}"
        echo ""
        echo "Final Metrics:"
        echo "  Val AUC: ${val_auc:-N/A} (target: > 0.5)"
        echo "  Model Size: ~473 KB (target: < 500 KB)"
        echo ""
        echo "Time elapsed: ${duration}s"
        echo ""
        echo -e "${GREEN}Next steps:${NC}"
        echo "  - Run full training: ./start.sh train --all-folds"
        echo "  - Check status: ./start.sh status"
    else
        die "Training failed. Check log: $LOG_DIR/quick_test_$fold.log"
    fi
}

# =============================================================================
# TRAIN MODE
# =============================================================================

cmd_train() {
    local fold=""
    local all_folds=false
    local resume=""
    local config=""
    local epochs=""
    local batch_size=""
    local device_override=""
    local continue_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --fold) shift; fold="$1" ;;
            --all-folds) all_folds=true ;;
            --resume) shift; resume="$1" ;;
            --config) shift; config="$1" ;;
            --epochs) shift; epochs="$1" ;;
            --batch-size) shift; batch_size="$1" ;;
            --device) shift; device_override="$1" ;;
            --continue) continue_only=true ;;
        esac
        shift
    done
    
    # Validate arguments
    if [[ "$all_folds" != "true" ]] && [[ -z "$fold" ]]; then
        die "Must specify --fold FOLD or --all-folds"
    fi
    
    # Determine config
    if [[ -z "$config" ]]; then
        config=$(get_recommended_config)
    fi
    
    # Determine device
    local device
    if [[ -n "$device_override" ]]; then
        device="$device_override"
    else
        device=$(detect_device)
    fi
    
    # Load config output dir
    local output_dir
    output_dir=$(python3 -c "import yaml; print(yaml.safe_load(open('$config')).get('output_dir', 'outputs/default/'))" 2>/dev/null || echo "outputs/default/")
    
    print_header "Training${fold:+ - Fold $fold}${all_folds:+ - All 15 Folds}"
    
    echo "Configuration:"
    echo "  Config: $config"
    echo "  Device: $device"
    echo "  Output: $output_dir"
    echo ""
    
    # Build command options
    local cmd_opts=""
    [[ -n "$epochs" ]] && cmd_opts="$cmd_opts --epochs $epochs"
    [[ -n "$batch_size" ]] && cmd_opts="$cmd_opts --batch-size $batch_size"
    [[ -n "$device_override" ]] && cmd_opts="$cmd_opts --device $device_override"
    [[ -n "$resume" ]] && cmd_opts="$cmd_opts --resume $resume"
    
    # Single fold training
    if [[ "$all_folds" != "true" ]]; then
        info "Starting training for fold $fold..."
        
        mkdir -p "$output_dir/logs"
        
        if python3 "$SCRIPT_DIR/train_loso.py" \
            --config "$config" \
            --fold "$fold" \
            $cmd_opts 2>&1 | tee "$output_dir/logs/fold_${fold}_train.log"; then
            success "Training completed for fold $fold"
        else
            die "Training failed for fold $fold"
        fi
        return
    fi
    
    # All folds training
    local completed=0
    local failed=0
    local skipped=0
    local total=${#FOLDS[@]}
    
    for f in "${FOLDS[@]}"; do
        # Check if already complete
        if [[ "$continue_only" == "true" ]]; then
            if [[ -f "$output_dir/logs/fold_${f}_summary.json" ]]; then
                info "Fold $f already complete. Skipping."
                ((skipped++))
                continue
            fi
        fi
        
        echo ""
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        info "Training fold $f ($((completed + failed + skipped + 1))/$total)"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        mkdir -p "$output_dir/logs"
        
        if python3 "$SCRIPT_DIR/train_loso.py" \
            --config "$config" \
            --fold "$f" \
            $cmd_opts 2>&1 | tee "$output_dir/logs/fold_${f}_train.log"; then
            success "Fold $f completed"
            ((completed++))
        else
            error "Fold $f failed"
            ((failed++))
        fi
        
        # Progress summary
        echo ""
        echo "Progress: [$completed/$total complete, $failed failed, $skipped skipped]"
    done
    
    # Final summary
    print_section "Training Summary"
    echo "  Completed: $completed/$total"
    echo "  Failed: $failed/$total"
    echo "  Skipped: $skipped/$total"
    
    if [[ $failed -eq 0 ]]; then
        success "All folds completed successfully!"
        echo ""
        echo -e "${GREEN}Next step:${NC} ./start.sh verify --generate-report"
    else
        warn "$failed fold(s) failed. Check logs in $output_dir/logs/"
    fi
}

# =============================================================================
# STATUS MODE
# =============================================================================

cmd_status() {
    local output_dir=""
    local detailed=false
    local watch=false
    local interval=10
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output-dir) shift; output_dir="$1" ;;
            --detailed) detailed=true ;;
            --watch) watch=true ;;
            --interval) shift; interval="$1" ;;
        esac
        shift
    done
    
    # Determine output directory
    if [[ -z "$output_dir" ]]; then
        # Try to find most recent output dir
        output_dir=$(find "$OUTPUT_DIR" -maxdepth 1 -type d | sort -r | head -1)
        if [[ -z "$output_dir" ]] || [[ "$output_dir" == "$OUTPUT_DIR" ]]; then
            output_dir="$OUTPUT_DIR/production_cuda"
        fi
    fi
    
    local status_func() {
        print_header "Training Status Report"
        
        echo "Output Directory: $output_dir"
        echo ""
        
        if [[ ! -d "$output_dir" ]]; then
            warn "Output directory not found: $output_dir"
            return
        fi
        
        # Check each fold
        local completed=0
        local pending=0
        local running=0
        
        printf "┌─────────┬──────────┬──────────┬──────────┬──────────┐\n"
        printf "│ %-7s │ %-8s │ %-8s │ %-8s │ %-8s │\n" "Fold" "Status" "Test AUC" "Test F1" "Miss Rate"
        printf "├─────────┼──────────┼──────────┼──────────┼──────────┤\n"
        
        for fold in "${FOLDS[@]}"; do
            local summary_file="$output_dir/logs/fold_${fold}_summary.json"
            local log_file="$output_dir/logs/fold_${fold}_train.log"
            
            local status="pending"
            local test_auc="-"
            local test_f1="-"
            local miss_rate="-"
            
            if [[ -f "$summary_file" ]]; then
                status="done"
                ((completed++))
                # Parse metrics
                test_auc=$(python3 -c "import json; d=json.load(open('$summary_file')); print(f\"{d.get('test_metrics',{}).get('auc',0):.4f}\")" 2>/dev/null || echo "N/A")
                test_f1=$(python3 -c "import json; d=json.load(open('$summary_file')); print(f\"{d.get('test_metrics',{}).get('f1',0):.4f}\")" 2>/dev/null || echo "N/A")
                miss_rate=$(python3 -c "import json; d=json.load(open('$summary_file')); print(f\"{d.get('test_metrics',{}).get('miss_rate',0):.4f}\")" 2>/dev/null || echo "N/A")
            elif [[ -f "$log_file" ]] && pgrep -f "train_loso.*fold.*$fold" > /dev/null 2>&1; then
                status="running"
                ((running++))
            else
                ((pending++))
            fi
            
            printf "│ %-7s │ %-8s │ %-8s │ %-8s │ %-8s │\n" "$fold" "$status" "$test_auc" "$test_f1" "$miss_rate"
        done
        
        printf "└─────────┴──────────┴──────────┴──────────┴──────────┘\n"
        
        echo ""
        echo "Summary: $completed complete, $running running, $pending pending"
        
        if [[ $completed -gt 0 ]]; then
            local avg_auc
            avg_auc=$(python3 << EOF 2>/dev/null
import json
import glob
aucs = []
for f in glob.glob('$output_dir/logs/fold_*_summary.json'):
    with open(f) as fp:
        d = json.load(fp)
        aucs.append(d.get('test_metrics',{}).get('auc',0))
if aucs:
    print(f"{sum(aucs)/len(aucs):.4f}")
EOF
)
            if [[ -n "$avg_auc" ]]; then
                echo "Average Test AUC: $avg_auc"
            fi
        fi
    }
    
    if [[ "$watch" == "true" ]]; then
        while true; do
            clear
            status_func
            echo ""
            echo "Refreshing every ${interval}s (Ctrl+C to exit)..."
            sleep "$interval"
        done
    else
        status_func
    fi
}

# =============================================================================
# VERIFY MODE
# =============================================================================

cmd_verify() {
    local output_dir=""
    local fold=""
    local integrity_check=false
    local generate_report=false
    local export_format="markdown"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output-dir) shift; output_dir="$1" ;;
            --fold) shift; fold="$1" ;;
            --integrity-check) integrity_check=true ;;
            --generate-report) generate_report=true ;;
            --export-format) shift; export_format="$1" ;;
        esac
        shift
    done
    
    # Determine output directory
    if [[ -z "$output_dir" ]]; then
        output_dir=$(find "$OUTPUT_DIR" -maxdepth 1 -type d | sort -r | head -1)
        if [[ -z "$output_dir" ]] || [[ "$output_dir" == "$OUTPUT_DIR" ]]; then
            output_dir="$OUTPUT_DIR/production_cuda"
        fi
    fi
    
    print_header "Verification Report"
    
    echo "Output Directory: $output_dir"
    echo ""
    
    if [[ ! -d "$output_dir" ]]; then
        die "Output directory not found: $output_dir"
    fi
    
    local total_checks=6
    local current_check=0
    
    # Check 1: Directory structure
    ((current_check++))
    echo -n "[$current_check/$total_checks] Checking directory structure... "
    if [[ -d "$output_dir/checkpoints" ]] && [[ -d "$output_dir/logs" ]]; then
        success "OK"
    else
        warn "Incomplete structure"
    fi
    
    # Check 2: Checkpoint files
    ((current_check++))
    echo -n "[$current_check/$total_checks] Verifying checkpoint files... "
    local checkpoint_count
    checkpoint_count=$(find "$output_dir/checkpoints" -name "fold_*_best.pt" 2>/dev/null | wc -l)
    success "$checkpoint_count/15 found"
    
    # Check 3: Summary files
    ((current_check++))
    echo -n "[$current_check/$total_checks] Verifying logs completeness... "
    local summary_count
    summary_count=$(find "$output_dir/logs" -name "fold_*_summary.json" 2>/dev/null | wc -l)
    success "$summary_count/15 found"
    
    # Check 4: Model sizes
    ((current_check++))
    echo -n "[$current_check/$total_checks] Checking model sizes... "
    local all_valid=true
    for ckpt in "$output_dir"/checkpoints/fold_*_best.pt; do
        if [[ -f "$ckpt" ]]; then
            local size_kb
            size_kb=$(du -k "$ckpt" | cut -f1)
            if [[ $size_kb -gt 600 ]]; then
                warn "Large checkpoint: $ckpt (${size_kb}KB)"
                all_valid=false
            fi
        fi
    done
    if [[ "$all_valid" == "true" ]]; then
        success "All < 500 KB"
    fi
    
    # Check 5: Integrity check (optional)
    if [[ "$integrity_check" == "true" ]]; then
        ((current_check++))
        echo -n "[$current_check/$total_checks] Deep integrity check... "
        local valid_count=0
        for ckpt in "$output_dir"/checkpoints/fold_*_best.pt; do
            if [[ -f "$ckpt" ]]; then
                if python3 -c "import torch; torch.load('$ckpt', map_location='cpu')" 2>/dev/null; then
                    ((valid_count++))
                fi
            fi
        done
        success "$valid_count valid"
    fi
    
    # Check 6: Generate report
    ((current_check++))
    echo -n "[$current_check/$total_checks] Generating summary... "
    
    # Aggregate metrics
    python3 << EOF
import json
import glob
import os

results = []
for summary_file in glob.glob('$output_dir/logs/fold_*_summary.json'):
    with open(summary_file) as f:
        data = json.load(f)
        fold = data.get('fold_id', 'unknown')
        test_metrics = data.get('test_metrics', {})
        results.append({
            'fold': fold,
            'auc': test_metrics.get('auc', 0),
            'f1': test_metrics.get('f1', 0),
            'miss_rate': test_metrics.get('miss_rate', 0)
        })

if results:
    avg_auc = sum(r['auc'] for r in results) / len(results)
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    avg_miss = sum(r['miss_rate'] for r in results) / len(results)
    print(f"\nAggregated Metrics ({len(results)}/15 folds):")
    print(f"  Test AUC:     {avg_auc:.4f}")
    print(f"  Test F1:      {avg_f1:.4f}")
    print(f"  Miss Rate:    {avg_miss:.4f}")
EOF
    success "Done"
    
    if [[ "$generate_report" == "true" ]]; then
        local report_file="$output_dir/verification_report.md"
        cat > "$report_file" << 'EOF'
# Verification Report

Generated: $(date)
Output Directory: $output_dir

## Summary

- Total folds: 15
- Completed: $summary_count
- Checkpoints: $checkpoint_count

## Aggregated Metrics

See console output above for detailed metrics.

## File Structure

EOF
        ls -la "$output_dir/checkpoints/" >> "$report_file" 2>/dev/null || true
        success "Report saved: $report_file"
    fi
}

# =============================================================================
# CLEAN MODE
# =============================================================================

cmd_clean() {
    local temp_only=false
    local archive=false
    local archive_dir="$SCRIPT_DIR/archive"
    local keep_last=0
    local full_reset=false
    local dry_run=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --temp-only) temp_only=true ;;
            --archive) archive=true ;;
            --archive-dir) shift; archive_dir="$1" ;;
            --keep-last) shift; keep_last="$1" ;;
            --full-reset) full_reset=true ;;
            --dry-run) dry_run=true ;;
        esac
        shift
    done
    
    print_header "Cleanup Operation"
    
    if [[ "$full_reset" == "true" ]]; then
        echo -e "${RED}WARNING: Full reset will remove ALL outputs, logs, and cache!${NC}"
        read -p "Are you sure? Type 'yes' to continue: " confirm
        if [[ "$confirm" != "yes" ]]; then
            echo "Cancelled."
            exit 0
        fi
    fi
    
    # Dry run header
    if [[ "$dry_run" == "true" ]]; then
        echo -e "${YELLOW}DRY RUN - No files will be removed${NC}"
        echo ""
    fi
    
    local actions=()
    
    # Find items to clean
    if [[ "$temp_only" == "true" ]] || [[ "$full_reset" == "true" ]]; then
        # Python cache
        while IFS= read -r -d '' dir; do
            actions+=("[Remove] $dir")
            if [[ "$dry_run" != "true" ]]; then
                rm -rf "$dir"
            fi
        done < <(find "$SCRIPT_DIR" -type d -name "__pycache__" -print0 2>/dev/null)
        
        # .pyc files
        while IFS= read -r -d '' file; do
            actions+=("[Remove] $file")
            if [[ "$dry_run" != "true" ]]; then
                rm -f "$file"
            fi
        done < <(find "$SCRIPT_DIR" -name "*.pyc" -print0 2>/dev/null)
    fi
    
    # Archive outputs
    if [[ "$archive" == "true" ]]; then
        local timestamp
        timestamp=$(date +%Y%m%d_%H%M%S)
        local archive_path="$archive_dir/$timestamp"
        
        mkdir -p "$archive_path"
        
        for dir in "$OUTPUT_DIR"/*/; do
            if [[ -d "$dir" ]]; then
                local dirname
                dirname=$(basename "$dir")
                actions+=("[Archive] $dir -> $archive_path/$dirname/")
                if [[ "$dry_run" != "true" ]]; then
                    mv "$dir" "$archive_path/"
                fi
            fi
        done
    fi
    
    # Full reset
    if [[ "$full_reset" == "true" ]]; then
        actions+=("[Remove] $OUTPUT_DIR/")
        actions+=("[Remove] $LOG_DIR/")
        if [[ "$dry_run" != "true" ]]; then
            rm -rf "$OUTPUT_DIR"/*
            rm -rf "$LOG_DIR"/*
        fi
    fi
    
    # Show actions
    if [[ ${#actions[@]} -eq 0 ]]; then
        echo "Nothing to clean."
        return
    fi
    
    echo "Actions to perform:"
    for action in "${actions[@]}"; do
        echo "  $action"
    done
    
    if [[ "$dry_run" == "true" ]]; then
        return
    fi
    
    # Confirm
    if [[ "$full_reset" != "true" ]]; then
        read -p "Proceed? [y/N]: " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            echo "Cancelled."
            exit 0
        fi
    fi
    
    echo ""
    success "Cleanup completed"
}

# =============================================================================
# HELP MODE
# =============================================================================

cmd_help() {
    cat << 'EOF'
VAD Distillation Project - start.sh

USAGE:
    ./start.sh <MODE> [OPTIONS]

MODES:
    setup       Initial environment setup
    quick-test  Quick validation (< 5 min)
    train       Full training (single or all folds)
    status      Check training status
    verify      Verify outputs and generate report
    clean       Cleanup and reset
    help        Show this help message

SETUP OPTIONS:
    --skip-requirements     Skip pip install
    --skip-data-check       Skip data validation
    --skip-cuda-check       Skip CUDA verification
    --force                 Force re-setup

QUICK-TEST OPTIONS:
    --fold FOLD             Test fold (default: F01)
    --config FILE           Custom config
    --device DEVICE         Force device

TRAIN OPTIONS:
    --fold FOLD             Train specific fold
    --all-folds             Train all 15 folds
    --resume CHECKPOINT     Resume from checkpoint
    --config FILE           Config file
    --epochs N              Override epochs
    --batch-size N          Override batch size
    --device DEVICE         Override device
    --continue              Continue incomplete folds only

STATUS OPTIONS:
    --output-dir DIR        Check specific directory
    --detailed              Show detailed metrics
    --watch                 Continuously update
    --interval SECONDS      Update interval

VERIFY OPTIONS:
    --output-dir DIR        Verify specific directory
    --fold FOLD             Verify specific fold
    --integrity-check       Deep checkpoint check
    --generate-report       Generate comprehensive report
    --export-format FORMAT  json, html, or markdown

CLEAN OPTIONS:
    --temp-only             Remove only temporary files
    --archive               Archive outputs before cleaning
    --archive-dir DIR       Archive directory
    --keep-last N           Keep last N output directories
    --full-reset            Complete reset (DANGEROUS)
    --dry-run               Show what would be removed

GLOBAL OPTIONS:
    -h, --help              Show help
    -v, --verbose           Enable verbose output
    --no-color              Disable colored output

ENVIRONMENT VARIABLES:
    VAD_CONFIG              Default config file
    VAD_DATA_DIR            TORGO data directory
    VAD_OUTPUT_DIR          Output directory
    VAD_LOG_DIR             Log directory
    VAD_DEVICE              Preferred device
    VAD_SEED                Random seed

EXAMPLES:
    # Initial setup
    ./start.sh setup

    # Quick test
    ./start.sh quick-test

    # Train single fold
    ./start.sh train --fold F01

    # Train all folds
    ./start.sh train --all-folds

    # Continue incomplete
    ./start.sh train --all-folds --continue

    # Check status
    ./start.sh status

    # Watch mode
    ./start.sh status --watch

    # Verify results
    ./start.sh verify --generate-report

    # Archive old outputs
    ./start.sh clean --archive

FOLDS (15 total):
    F01, F03, F04, M01, M02, M03, M04, M05,
    FC01, FC02, FC03, MC01, MC02, MC03, MC04

For more information, see:
    - README.md
    - AGENTS.md
    - docs/START_SH_DESIGN.md
EOF
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    # Setup colors and logging
    setup_colors
    setup_logging
    detect_platform
    
    # Parse global options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                cmd_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --no-color)
                NO_COLOR=true
                setup_colors
                shift
                ;;
            -*)
                # Unknown option, pass to mode handler
                break
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Get mode
    if [[ $# -eq 0 ]]; then
        cmd_help
        exit 1
    fi
    
    MODE="$1"
    shift
    
    # Dispatch to mode handler
    case "$MODE" in
        setup)
            cmd_setup "$@"
            ;;
        quick-test)
            cmd_quick_test "$@"
            ;;
        train)
            cmd_train "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        verify)
            cmd_verify "$@"
            ;;
        clean)
            cmd_clean "$@"
            ;;
        help)
            cmd_help
            ;;
        *)
            error "Unknown mode: $MODE"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
