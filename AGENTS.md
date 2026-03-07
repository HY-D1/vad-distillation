# AGENTS.md - VAD Distillation Project

This file contains essential information for AI coding agents working on the Compact VAD for Atypical Speech via Knowledge Distillation project.

## Project Overview

This project builds a compact Voice Activity Detection (VAD) model optimized for atypical speech (e.g., dysarthric/Parkinsonian) by distilling knowledge from Silero VAD into a smaller "student" model.

### Key Goals
- **Model Size**: ≤ 500 KB (student model) - **Achieved: ~473 KB**
- **AUC Drop** (vs Silero on atypical): ≤ 10%
- **CPU Latency**: ≤ 10 ms/frame
- **Dataset**: TORGO (dysarthric speech dataset) with 15 speaker folds
- **Random Seed**: 6140 (NOT 42) for reproducibility

### Target Deliverables

| Metric | Target | Achieved |
|--------|--------|----------|
| Student Model Size | ≤ 500 KB | ~473 KB |
| AUC Drop (vs Silero on atypical) | ≤ 10% | TBD |
| CPU Latency | ≤ 10 ms/frame | TBD |
| Atypical Miss Rate | Lower than Silero baseline | TBD |

## Technology Stack

### Core Dependencies (from requirements.txt)
- **PyTorch** ≥ 2.0.0 (deep learning framework)
- **torchaudio** ≥ 2.0.0 (audio I/O)
- **librosa** ≥ 0.10.0 (mel spectrogram computation)
- **soundfile** ≥ 0.12.0 (audio loading)
- **numpy** ≥ 1.24.0, **scipy** ≥ 1.10.0
- **pandas** ≥ 2.0.0 (data manipulation)
- **scikit-learn** ≥ 1.3.0 (metrics computation: AUC, F1, confusion matrix)
- **PyYAML** ≥ 6.0 (configuration files)
- **matplotlib** ≥ 3.7.0, **seaborn** ≥ 0.12.0 (visualization)
- **tqdm** (progress bars)
- **speechbrain** (baseline comparison)

### Hardware Support
- **CUDA**: GPU acceleration on NVIDIA devices (RTX 4080 optimized, batch_size=64)
- **MPS**: Apple Silicon (M1/M2/M3) acceleration (batch_size=16)
- **CPU**: Fallback for all platforms

## Project Structure

```
vad-distillation/
├── analysis/                 # Analysis outputs and visualizations
├── baselines/                # Baseline VAD implementations
│   ├── energy_vad.py         # Energy-based baseline
│   ├── speechbrain_vad.py    # SpeechBrain wrapper
│   └── __init__.py
├── cli/                      # Unified CLI implementation
│   ├── config.py             # Configuration management
│   ├── utils.py              # CLI utilities
│   ├── __init__.py
│   └── commands/             # CLI command implementations
│       ├── analyze.py
│       ├── baseline.py
│       ├── clean.py
│       ├── export.py
│       ├── setup.py
│       ├── status.py
│       ├── sweep.py
│       ├── train.py
│       ├── validate.py
│       └── __init__.py
├── configs/                  # Training configurations
│   ├── pilot.yaml           # Smoke test config (CPU/MPS, small dataset)
│   ├── pilot_cuda.yaml      # RTX 4080 smoke test config
│   ├── production.yaml      # Full training config (MPS/macOS)
│   ├── production_cuda.yaml # Full training config (CUDA/RTX 4080)
│   ├── quick_test.yaml      # Quick CUDA verification (< 5 min)
│   ├── baselines.yaml       # Baseline experiment config
│   ├── alpha_sweep.yaml     # Alpha hyperparameter sweep
│   ├── temperature_sweep.yaml # Temperature sweep
│   ├── week2_matrix.json    # 36-experiment matrix
│   └── templates/           # Config templates
│       └── example_template.yaml
├── data/                     # TORGO dataset (not in git)
│   ├── __init__.py
│   ├── torgo_dataset.py     # TORGO dataset class with caching
│   └── torgo_raw/           # Raw audio files
├── models/                   # Model architectures
│   ├── __init__.py
│   ├── tinyvad_student.py   # Main TinyVAD student model (~473KB)
│   └── losses.py            # Distillation loss functions
├── notebooks/                # Jupyter notebooks for analysis
│   ├── baseline_silero_metrics.ipynb
│   └── eda_torgo_sentences.ipynb
├── outputs/                  # Training outputs
│   ├── pilot/               # Pilot experiment outputs
│   ├── pilot_cuda/          # CUDA pilot outputs
│   ├── production_cuda/     # CUDA production outputs
│   └── baselines/           # Baseline outputs
├── pretrained_models/        # Downloaded baseline models
├── scripts/                  # Utility and execution scripts
│   ├── __init__.py
│   ├── export_model.py      # Model export utilities
│   ├── README.md            # Scripts documentation
│   ├── analysis/            # Analysis scripts
│   │   ├── analyze_week2.py
│   │   ├── compare_methods.py
│   │   ├── compare_platforms.py
│   │   ├── compute_baseline_metrics.py
│   │   ├── quick_verify.sh
│   │   └── verify_4080_results.py
│   ├── archive/             # Archived/deprecated scripts
│   │   ├── cleanup_project.py
│   │   ├── create_all_teacher_probs_fixed.py
│   │   └── SCRIPT_AUDIT.md
│   ├── core/                # Core training and experiment scripts
│   │   ├── run_baseline.py
│   │   ├── run_experiment.py
│   │   ├── run_sweep.py
│   │   ├── run_tests.py     # Main test runner
│   │   ├── test_all_imports.py
│   │   └── verify_checkpoint.py
│   ├── data/                # Data processing scripts
│   │   ├── build_torgo_manifest.py
│   │   ├── cache_features.py
│   │   ├── cache_teacher.py
│   │   ├── generate_loso_splits.py
│   │   ├── run_silero_teacher.py
│   │   └── validate_torgo_setup.py
│   └── deprecated/          # Deprecated scripts (do not use)
│       ├── cache_manager.py
│       ├── compare_verification.py
│       ├── extract_predictions.py
│       └── generate_experiment_matrix.py
├── splits/                   # LOSO splits (15 JSON files for 15 folds)
│   ├── fold_F01.json
│   ├── fold_F03.json
│   ├── fold_F04.json
│   ├── fold_FC01.json
│   ├── fold_FC02.json
│   ├── fold_FC03.json
│   ├── fold_M01.json
│   ├── fold_M02.json
│   ├── fold_M03.json
│   ├── fold_M04.json
│   ├── fold_M05.json
│   ├── fold_MC01.json
│   ├── fold_MC02.json
│   ├── fold_MC03.json
│   ├── fold_MC04.json
│   └── summary.json
├── teacher_probs/            # Cached Silero teacher outputs
│   ├── meta.json
│   └── meta 3.json
├── manifests/                # Dataset manifests (CSV)
│   ├── torgo_pilot.csv      # Small subset for testing
│   ├── torgo_sentences.csv  # Full dataset
│   └── torgo_f01_only.csv   # Single speaker for testing
├── logs/                     # Training logs
├── teacher_hard_labels/      # Thresholded teacher labels (optional)
├── train_loso.py            # Main LOSO training script
├── train.py                 # Simple wrapper for smoke tests
├── utils.py                 # Shared utilities
├── vad.py                   # Unified CLI entry point
├── verify_configs.py        # Config validation script
├── requirements.txt         # Python dependencies
├── start.sh                 # Bash setup/training script (Unix)
├── start.ps1                # PowerShell setup script (Windows)
├── start.bat                # Windows batch wrapper
├── README.md                # Human documentation
├── CLI_DESIGN.md            # CLI design specification
├── AGENTS.md                # This file
└── LICENSE                  # MIT License
```

## Key Concepts

### 1. Knowledge Distillation
The student model learns from both:
- **Hard labels**: Ground truth from transcripts (1=speech, 0=silence)
- **Soft labels**: Teacher (Silero VAD) probability outputs

Loss formula: `L = α * L_hard + (1-α) * L_soft`
- `α` (alpha): Weight for hard loss (0.5 = balanced, default: 0.6 in production)
- `T` (temperature): Softens probability distributions (default: 3.0)
- Temperature scaling: `soft_probs = sigmoid(logits / T)`

### 2. Leave-One-Speaker-Out (LOSO)
Speaker-independent evaluation with 15 TORGO folds:
- Train on N-1 speakers
- Validate on 1 speaker
- Test on held-out speaker
- Repeat for all 15 speakers

**Speaker IDs**: F01, F03, F04, M01, M02, M03, M04, M05, FC01, FC02, FC03, MC01, MC02, MC03, MC04

Fold configuration stored in `splits/fold_{speaker}.json` with keys:
- `train`: List of training speaker IDs
- `val`: Validation speaker ID
- `test`: Test speaker ID

### 3. TinyVAD Architecture
```
Input: Mel spectrogram (batch, time, n_mels=40)
  ↓
CNN Frontend: Conv2d → BatchNorm → ReLU → MaxPool (×2 layers)
  - Time downsampling: factor of 2^N (N=CNN layers)
  - Default: 2 layers → 4x downsampling
  ↓
GRU Backend: 2-layer GRU with hidden=32
  ↓
Output: Frame-level speech probability (sigmoid)
```

**Default Configuration (production_cuda.yaml)**:
- CNN channels: [14, 28]
- GRU hidden: 32
- GRU layers: 2
- Model size: ~473 KB

**Frame Rate Alignment**:
- Input: 125 fps (8ms frames) with hop_length=128 @ 16kHz
- After 2-layer CNN (4x downsampling): ~31.25 fps
- Teacher (Silero): 31.25 fps (32ms frames)
- Alignment handled via adaptive pooling in loss function

### 4. Model Size Variants

The TinyVAD architecture supports multiple size variants to trade off between model capacity and deployment constraints. All variants maintain the ≤ 500 KB target.

| Variant | CNN Channels | GRU Hidden | GRU Layers | Approx. Size | Use Case | Config File(s) |
|---------|-------------|------------|------------|--------------|----------|----------------|
| **Default** | [14, 28] | 32 | 2 | ~473 KB | Production training, best accuracy | `production.yaml`, `production_cuda.yaml`, `pilot_cuda.yaml`, `quick_test.yaml` |
| **Small** | [12, 24] | 20 | 2 | ~300-400 KB | Balanced accuracy/size for edge devices | *Custom configuration* |
| **Tiny** | [16] | 16 | 2 | ~100-200 KB | Ultra-compact, mobile/embedded | *Custom configuration* |
| **Micro** | [8] | 8 | 2 | ~50-100 KB | Minimal footprint, feasibility testing | *Custom configuration* |
| **Pilot*** | [8, 16] | 24 | 2 | ~200-300 KB | Fast smoke testing, CPU-friendly | `pilot.yaml` |

**Notes:**
- **Default** is the recommended variant for production use, achieving the best accuracy within the size constraint
- **Pilot*** variant uses a unique configuration optimized for quick smoke tests on CPU
- All variants use 2 GRU layers for temporal modeling consistency
- Model size can be further reduced by using single-layer GRU or removing CNN layers (set `cnn_channels: []`)
- To use a different variant, modify the `model` section in your config YAML:
  ```yaml
  model:
    n_mels: 40
    cnn_channels: [12, 24]  # Adjust for variant
    gru_hidden: 20          # Adjust for variant
    gru_layers: 2
    dropout: 0.1
  ```

## Build and Test Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install speechbrain for baselines
pip install speechbrain
```

### Data Validation
```bash
# Validate TORGO dataset setup
python scripts/data/validate_torgo_setup.py

# Build dataset manifest
python scripts/data/build_torgo_manifest.py \
    --data_dir data/torgo_raw \
    --output manifests/torgo_sentences.csv
```

### Unified CLI (vad.py)

The project provides a unified CLI via `vad.py`:

```bash
# Setup and validation
python vad.py setup
python vad.py validate

# Training
python vad.py train --fold F01
python vad.py train --all --parallel 2
python vad.py train --quick  # Quick test (2 epochs, F01 only)

# Baselines
python vad.py baseline silero
python vad.py baseline energy
python vad.py baseline speechbrain
python vad.py baseline all

# Hyperparameter sweep
python vad.py sweep --param alpha --values 0.3 0.5 --folds F01

# Analysis
python vad.py analyze
python vad.py status

# Model export
python vad.py export --format onnx
python vad.py export --format torchscript
```

### Legacy Training Commands (train_loso.py)

#### Smoke Test (Quick Validation)
```bash
# CPU smoke test
python train_loso.py --config configs/pilot.yaml --fold F01 --test

# CUDA smoke test (RTX 4080)
python train_loso.py --config configs/pilot_cuda.yaml --fold F01 --test
```

#### Full Training
```bash
# Single fold on MPS (macOS)
python train_loso.py --config configs/production.yaml --fold F01

# Single fold on CUDA (RTX 4080)
python train_loso.py --config configs/production_cuda.yaml --fold F01
```

#### Quick Test (CUDA Verification)
```bash
# Quick CUDA verification (< 5 minutes)
python train_loso.py --config configs/quick_test.yaml --fold F01
```

#### Hyperparameter Sweep
```bash
# Run grid search over alpha and temperature
python scripts/core/run_sweep.py \
    --param alpha --values 0.3 0.5 0.7 0.9 \
    --param temperature --values 1 2 3 5 \
    --folds F01 M01 FC01 \
    --base-config configs/pilot.yaml \
    --output-dir outputs/sweep \
    --epochs 50
```

### Baselines
```bash
# Energy-based VAD
python scripts/core/run_baseline.py \
    --method energy \
    --manifest manifests/torgo_pilot.csv \
    --output-dir outputs/baselines/energy/

# Silero VAD (teacher)
python scripts/core/run_baseline.py \
    --method silero \
    --manifest manifests/torgo_pilot.csv \
    --output-dir outputs/baselines/silero/

# SpeechBrain VAD
python scripts/core/run_baseline.py \
    --method speechbrain \
    --manifest manifests/torgo_pilot.csv \
    --output-dir outputs/baselines/speechbrain/
```

### Analysis
```bash
# Analyze sweep results
python scripts/analysis/analyze_week2.py \
    --results-dir outputs/sweep \
    --output-dir analysis/sweep

# Compare methods
python scripts/analysis/compare_methods.py \
    --results-dir outputs/production_cuda \
    --output-dir analysis/comparison
```

### Testing
```bash
# Run all tests
python scripts/core/run_tests.py

# Or run individual test modules
python -m models.tinyvad_student  # Model tests
python -m data.torgo_dataset      # Dataset tests
```

## Platform-Specific Configuration

### Windows RTX 4080 CUDA

| Parameter | Value |
|-----------|-------|
| `device` | `cuda` |
| `batch_size` | 64 |
| `num_workers` | 4 |
| Config file | `production_cuda.yaml` |

**Workflow**:
```bash
# 1. Quick verification (< 5 min)
python train_loso.py --config configs/quick_test.yaml --fold F01

# 2. Pilot test (10-15 min)
python train_loso.py --config configs/pilot_cuda.yaml --fold F01 --epochs 5

# 3. Full production training (per fold)
python train_loso.py --config configs/production_cuda.yaml --fold F01
```

### macOS MPS (Apple Silicon)

| Parameter | Value |
|-----------|-------|
| `device` | `mps` |
| `batch_size` | 16 |
| `num_workers` | 0 |
| Config file | `production.yaml` |

**Note**: MPS falls back to CPU automatically if not available. Some operations (adaptive pooling) may run on CPU even with MPS.

## Code Style Guidelines

### Python Style
- Follow PEP 8 conventions
- Maximum line length: 100 characters
- Use double quotes for docstrings, single quotes for strings
- Type hints for function signatures (preferred)

### Imports
```python
# Standard library first
import argparse
import json
from pathlib import Path
from typing import Dict, List

# Third-party packages
import numpy as np
import torch
import yaml

# Local modules
from data import TORGODataset
from models.tinyvad_student import create_student_model
```

### Documentation
All functions must have docstrings (Google style):

```python
def compute_metrics(predictions: np.ndarray, 
                    labels: np.ndarray,
                    probs: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Binary predictions [N]
        labels: Ground truth labels [N]
        probs: Probability of positive class [N]
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary of metrics (auc, f1, miss_rate, etc.)
    """
```

### Naming Conventions
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Private functions prefixed with `_`

## Configuration Files

### YAML Structure
Configurations are YAML files with the following sections:

```yaml
seed: 6140                    # Random seed for reproducibility
device: cuda                  # Device: cpu, cuda, or mps
num_workers: 4                # DataLoader workers (0 for MPS)

model:                        # Model architecture
  n_mels: 40
  cnn_channels:
  - 14
  - 28
  gru_hidden: 32
  gru_layers: 2
  dropout: 0.1

alpha: 0.6                    # Weight for hard loss
temperature: 3.0              # Temperature for softening

dataset: torgo_full
num_samples: 2000             # Samples per epoch
seq_len: 150                  # Frames per sequence
batch_size: 64
learning_rate: 0.001
num_epochs: 20

lr_scheduler:                 # Optional LR scheduler
  type: ReduceLROnPlateau
  factor: 0.5
  patience: 3
  min_lr: 0.0001

manifest: manifests/torgo_sentences.csv
teacher_probs_dir: teacher_probs/
output_dir: outputs/production_cuda/

log_interval: 50
save_interval: 5

early_stopping:               # Optional early stopping
  enabled: true
  patience: 5
  metric: val_loss
```

### Configuration Hierarchy
The configuration system follows this priority (highest to lowest):
1. CLI arguments
2. Environment variables (VAD_CONFIG, VAD_DEVICE, etc.)
3. Config file (YAML)
4. Default values

## Testing Strategy

### Test Types
1. **Unit Tests**: Individual components (dataset, model, losses)
2. **Integration Tests**: Full training pipeline
3. **Smoke Tests**: Quick validation with minimal data

### Running Tests
```bash
# Run all tests via test runner
python scripts/core/run_tests.py

# Run test mode (processes only 5 samples)
python train_loso.py --config configs/pilot.yaml --fold F01 --test

# Test model variants
python models/tinyvad_student.py  # Runs comprehensive tests

# Verify CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify MPS setup
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Model Tests (in tinyvad_student.py)
- Forward pass validation
- Size check (≤ 500 KB)
- ONNX export
- TorchScript export
- Predict method with audio

## Output Directory Structure

After training completes:
```
outputs/production_cuda/
├── checkpoints/
│   ├── fold_F01_latest_best.pt         # Best model by validation AUC
│   ├── fold_F01_latest.pt       # Latest checkpoint
│   └── fold_F01_epoch_10.pt     # Periodic checkpoints
├── logs/
│   ├── fold_F01.csv             # Per-epoch metrics
│   ├── fold_F01_summary.json    # Final summary
│   └── fold_F01_predictions.npz # Test predictions
└── config.yaml                  # Effective config used
```

### Key Metrics to Monitor
- **Val AUC**: Primary metric for model selection (> 0.85 good)
- **Test AUC**: Should be within 5% of Val AUC
- **Miss Rate**: Target < 0.20 for atypical speech
- **Model Size**: Must be < 500 KB

## Common Issues and Solutions

### NaN in Loss
- **Cause**: Learning rate too high
- **Fix**: Reduce learning rate (0.001 → 0.0001)

### AUC = 0.5 (Random)
- **Cause**: Model not learning
- **Fix**: Check data loading, increase model capacity

### Out of Memory (CUDA)
- **Cause**: Batch size too large for RTX 4080
- **Fix**: Reduce `batch_size` from 64 to 32

### Out of Memory (MPS)
- **Cause**: Batch size too large for Apple Silicon
- **Fix**: Reduce `batch_size` from 16 to 8

### MPS Device Issues (macOS)
- **Cause**: MPS doesn't support all operations (e.g., adaptive pooling with non-divisible sizes)
- **Fix**: Code includes CPU fallback for adaptive pooling in `models/losses.py`

### Audio Loading Errors
- **Cause**: Missing torchaudio or corrupt files
- **Fix**: Run `scripts/data/validate_torgo_setup.py`

## Data Requirements

### TORGO Dataset Structure
```
data/torgo_raw/
├── F01/
│   ├── Session1/
│   │   ├── wav_headMic/     # Audio files (.wav)
│   │   └── prompts/          # Transcripts (.txt)
│   └── Session2/
├── M01/
└── ... (15 speakers total)
```

### Download
1. Visit: https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
2. Complete license agreement
3. Extract to `data/torgo_raw/`

### Manifest Format
CSV with columns:
- `utterance_id`: Unique identifier
- `audio_path`: Path to audio file
- `speaker_id`: Speaker identifier (F01, M01, FC01, MC01, etc.)
- `transcript`: Text transcription
- `duration`: Audio duration in seconds

### Teacher Probabilities
Pre-computed Silero outputs stored in `teacher_probs/`:
- One `.npz` file per utterance
- Contains: `probs` (frame-level probabilities), `frame_rate`, `threshold`
- Generated by `scripts/data/cache_teacher.py`

## Model Export

### TorchScript
```python
from models import create_student_model
model = create_student_model()
model.export_torchscript('deploy/model.pt', method='trace')
```

### ONNX
```python
model.export_onnx('deploy/model.onnx', input_shape=(1, 100, 40))
```

### Expected Sizes
- PyTorch checkpoint: ~1,400 KB (includes optimizer state)
- TorchScript: ~473 KB (model only)
- ONNX: ~473 KB (model only)

## Security Considerations

1. **Data Privacy**: TORGO contains medical speech data
   - Do not commit audio files to git
   - Keep data in `data/torgo_raw/` (gitignored)
   
2. **Model Files**: Checkpoint files are binary
   - Verify source before loading unknown checkpoints
   - Use `map_location='cpu'` when loading on different devices

3. **Dependencies**: Keep dependencies updated
   - Regularly run `pip install --upgrade -r requirements.txt`
   - Check for security advisories in PyTorch, librosa

## Development Workflow

### New Feature Development
1. Create feature branch
2. Test with pilot config: `python train_loso.py --config configs/pilot.yaml --fold F01 --test`
3. Run small experiment: `python train_loso.py --config configs/pilot.yaml --fold F01 --epochs 5`
4. Verify metrics and model size
5. Merge to main

### Adding New Scripts
- Place in appropriate `scripts/` subdirectory
- Add module docstring with usage examples
- Follow existing argparse patterns
- Update this AGENTS.md if adding new commands

### Configuration Changes
- Add new configs to `configs/` directory
- Document in config YAML comments
- Test on both CPU and target hardware
- Update example commands in this file

### CLI Implementation Status

The project includes a unified CLI (`vad.py`) that wraps legacy scripts. Below is the current implementation status:

| Command | Status | Implementation | Notes |
|---------|--------|----------------|-------|
| setup | ✅ Working | Native | Full implementation with dependency check, install, and directory setup |
| validate | ✅ Working | Native | Validates environment, data, configs, folds, and teacher probabilities |
| train | ⚠️ Partial | Wrapper | Calls `train_loso.py` via subprocess; supports --fold, --all, --quick, --resume |
| baseline | ⚠️ Partial | Wrapper | Calls `scripts/core/run_baseline.py` for silero/energy/speechbrain methods |
| sweep | ⚠️ Partial | Wrapper | Calls `scripts/core/run_sweep.py` for hyperparameter grid search |
| analyze | ⚠️ Partial | Hybrid | Comparison works via `compare_methods.py`; report generation is stubbed (TODO) |
| status | ✅ Working | Native | Shows training status table with metrics; supports --watch and --json modes |
| clean | ✅ Working | Native | Full implementation for checkpoints, logs, cache cleanup with --dry-run |
| export | ✅ Working | Native | Native ONNX and TorchScript export using model's export methods |

**Legend:**
- ✅ Working - Fully functional native implementation
- ⚠️ Partial - Functional but delegates to legacy scripts (wrapper) or has partial features
- ❌ Planned - Not yet implemented

**Notes:**
- Wrapper commands provide a unified interface but still depend on legacy scripts in `scripts/core/`
- The `analyze` command's `report` subcommand has a TODO marker for full report generation with plots
- All commands include proper argument parsing, help text, and colored output via `cli.utils`

## Helper Scripts

### start.sh (Unix/Linux/macOS)
Bash script for setup, training, and verification:
```bash
./start.sh setup       # Environment setup
./start.sh quick-test  # Quick validation
./start.sh train       # Full training
./start.sh status      # Check training status
./start.sh verify      # Verify outputs
./start.sh clean       # Cleanup
```

### start.ps1 (Windows)
PowerShell equivalent of start.sh for Windows users.

## Additional Resources

- **README.md**: Human-oriented documentation with platform-specific guides
- **CLI_DESIGN.md**: Detailed CLI design specification
- **Scripts**: All scripts in `scripts/` have `--help` for usage
- **Notebooks**: EDA and analysis in `notebooks/`

## Phase 1 Final Project Workflow (Demo Video 1)

### Prerequisites
- All 15 LOSO folds trained and checkpoints in `outputs/production_cuda/checkpoints/`
- Training logs in `outputs/production_cuda/logs/`

### Step 1: Verify Training Results
```bash
# Verify all 15 folds completed successfully
python scripts/analysis/verify_4080_results.py \
    --results-dir outputs/production_cuda \
    --max-miss-rate 0.75 --min-f1 0.4
```

### Step 2: Extract Frame Predictions
```bash
# Extract all fold predictions to frame_probs format for comparison
python scripts/core/extract_all_predictions.py \
    --results-dir outputs/production_cuda \
    --output-dir outputs/our_model
```

This creates:
```
outputs/our_model/
├── extraction_summary.json
├── fold_F01/frame_probs/*.npy
├── fold_F03/frame_probs/*.npy
└── ... (all 15 folds)
```

### Step 3: Run Baselines (if not already done)
```bash
# Energy baseline
python scripts/core/run_baseline.py \
    --method energy \
    --manifest manifests/torgo_sentences.csv \
    --output-dir outputs/baselines/energy/

# Silero baseline (teacher)
python scripts/core/run_baseline.py \
    --method silero \
    --manifest manifests/torgo_sentences.csv \
    --output-dir outputs/baselines/silero/

# SpeechBrain baseline (optional)
python scripts/core/run_baseline.py \
    --method speechbrain \
    --manifest manifests/torgo_sentences.csv \
    --output-dir outputs/baselines/speechbrain/
```

### Step 4: Run Full Comparison Pipeline
```bash
# Run complete comparison (auto-detects existing baselines)
python scripts/core/run_full_comparison.py

# Or with specific options
python scripts/core/run_full_comparison.py \
    --skip-baselines \
    --methods energy,silero,our_model
```

Generates:
- `analysis/comparison/comparison_table.csv`
- `analysis/comparison/comparison_table.md`
- `analysis/comparison/plots/*.png`

### Step 5: Run Latency Benchmark
```bash
# Benchmark student model latency
python scripts/analysis/benchmark_latency.py \
    --compare-baselines
```

Generates:
- `analysis/benchmark_results.json`
- `analysis/benchmark_report.md`
- `analysis/latency_comparison.png`

### Alternative: One-Command Full Pipeline
```bash
# Run everything (extract, baselines, compare, benchmark)
python scripts/core/run_full_comparison.py && \
python scripts/analysis/benchmark_latency.py --compare-baselines
```

## New Scripts Reference

| Script | Purpose | Location |
|--------|---------|----------|
| `run_all_folds.py` | Train all 15 LOSO folds | `scripts/core/` |
| `extract_all_predictions.py` | Extract frame_probs from predictions.npz | `scripts/core/` |
| `run_full_comparison.py` | Master comparison pipeline | `scripts/core/` |
| `benchmark_latency.py` | Latency + engineering targets | `scripts/analysis/` |
| `extract_predictions.py` | Per-fold prediction extraction | `scripts/analysis/` |
| `run_baseline.py` (wrapper) | Backward compatibility wrapper | `scripts/` |
| `compare_methods.py` (wrapper) | Backward compatibility wrapper | `scripts/` |
| `run_all_folds.bat` | Windows batch for training | `scripts/platform/windows/` |

## Training on Windows RTX 4080

### Option 1: Individual Folds
```bash
python train_loso.py --config configs/production_cuda.yaml --fold F01
```

### Option 2: All Folds (Sequential)
```bash
python scripts/core/run_all_folds.py --config configs/production_cuda.yaml
```

### Option 3: All Folds (Parallel - 2 at a time)
```bash
python scripts/core/run_all_folds.py --config configs/production_cuda.yaml --parallel 2
```

### Option 4: Resume Interrupted Training
```bash
python scripts/core/run_all_folds.py --config configs/production_cuda.yaml --resume
```

---

*Last updated: 2026-03-07*
*Seed: 6140 | Model Size Target: ≤ 500 KB | Folds: 15* | *Phase 1: Complete*
