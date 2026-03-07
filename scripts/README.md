# Scripts Directory

This directory contains utility scripts organized by category for the VAD Distillation project.

## Directory Structure

```
scripts/
├── README.md              # This file
├── __init__.py            # Package marker
├── core/                  # Essential training/validation scripts
├── data/                  # Data preparation and caching
├── analysis/              # Analysis and comparison tools
├── platform/              # Platform-specific scripts
│   ├── windows/           # Windows PowerShell/Batch scripts
│   └── unix/              # Mac/Linux shell scripts
└── archive/               # Deprecated/one-off scripts
```

## Quick Reference

| Task | Script Location |
|------|----------------|
| Train single fold | `python train_loso.py --config configs/pilot.yaml --fold F01` |
| Run experiment | `python scripts/core/run_experiment.py --help` |
| Run hyperparameter sweep | `python scripts/core/run_sweep.py --help` |
| Cache teacher outputs | `python scripts/data/cache_teacher.py --manifest manifests/torgo_sentences.csv` |
| Cache features | `python scripts/data/cache_features.py --manifest manifests/torgo_sentences.csv` |
| Compare methods | `python scripts/analysis/compare_methods.py --help` |
| Verify checkpoint | `python scripts/core/verify_checkpoint.py --help` |

---

## Core Scripts (`scripts/core/`)

Essential scripts for training, validation, and running experiments.

### Main Scripts

- **run_experiment.py** - Unified experiment runner supporting single experiments, batch experiments from JSON matrix, and resume functionality.
  ```bash
  python scripts/core/run_experiment.py --config configs/pilot.yaml --fold F01
  ```

- **run_sweep.py** - Hyperparameter sweep runner for grid search and random search over alpha and temperature parameters.
  ```bash
  python scripts/core/run_sweep.py --param alpha --values 0.3 0.5 0.7 --folds F01 --base-config configs/pilot.yaml
  ```

- **run_baseline.py** - Unified baseline execution script for Energy-based VAD, SpeechBrain VAD, and Silero VAD.
  ```bash
  python scripts/core/run_baseline.py --method energy --manifest manifests/torgo_sentences.csv --output-dir outputs/baseline_energy/
  ```

- **verify_checkpoint.py** - Verify trained checkpoint integrity and metrics.
  ```bash
  python scripts/core/verify_checkpoint.py --checkpoint outputs/production_cuda/checkpoints/fold_F01_latest_best.pt --fold F01
  ```

### Testing Scripts

- **run_tests.py** - Run project tests.
- **test_all_imports.py** - Test that all imports work correctly.

---

## Data Scripts (`scripts/data/`)

Scripts for data preparation, caching, and management.

### Caching Scripts

- **cache_teacher.py** - Enhanced teacher (Silero VAD) output caching with parallel batch processing and resume capability.
  ```bash
  python scripts/data/cache_teacher.py --manifest manifests/torgo_sentences.csv --device cuda --batch-size 16
  ```

- **cache_features.py** - Enhanced feature caching for mel spectrograms, MFCCs, and raw audio.
  ```bash
  python scripts/data/cache_features.py --manifest manifests/torgo_sentences.csv --features mel mfcc --parallel 4
  ```

### Data Preparation

- **build_torgo_manifest.py** - Build TORGO dataset manifest for sentence-level audio files.
  ```bash
  python scripts/data/build_torgo_manifest.py --data_dir data/torgo_raw --output manifests/torgo_sentences.csv
  ```

- **generate_loso_splits.py** - Generate Leave-One-Speaker-Out (LOSO) train/val/test splits.
  ```bash
  python scripts/data/generate_loso_splits.py --manifest manifests/torgo_sentences.csv --output_dir splits/
  ```

- **validate_torgo_setup.py** - Validate TORGO dataset setup and check for missing files.
  ```bash
  python scripts/data/validate_torgo_setup.py --data_dir data/torgo_raw
  ```

- **run_silero_teacher.py** - Run Silero VAD as teacher on audio files.

---

## Analysis Scripts (`scripts/analysis/`)

Scripts for analyzing results, comparing methods, and generating reports.

### Comparison Scripts

- **compare_methods.py** - Unified comparison script for multiple VAD methods.
  ```bash
  python scripts/analysis/compare_methods.py \
      --manifest manifests/torgo_pilot.csv \
      --methods outputs/exp1/frame_probs,outputs/exp2/frame_probs \
      --method-names "Baseline,Our Method" \
      --output-dir outputs/comparison
  ```

- **compare_platforms.py** - Compare results between Windows and Mac platforms.
### Analysis Scripts

- **analyze_week2.py** - Analyze Week 2 hyperparameter sweep results for alpha (α) and temperature (T).
  ```bash
  python scripts/analysis/analyze_week2.py --results-dir outputs/week2/ --output-dir analysis/week2/
  ```

- **verify_4080_results.py** - Verify RTX 4080 training results.

---

## Platform Scripts (`scripts/platform/`)

Platform-specific automation scripts.

### Windows (`scripts/platform/windows/`)

PowerShell and Batch scripts for Windows systems.

- **train_all_folds.ps1** - Train all 15 TORGO folds sequentially on Windows.
  ```powershell
  .\scripts\platform\windows\train_all_folds.ps1
  ```

- **train_all_folds.bat** - Batch file version for training all folds.
  ```batch
  scripts\platform\windows\train_all_folds.bat
  ```

- **quick_verify.ps1** - Quick verification script that tests F01 fold with --test flag.
  ```powershell
  .\scripts\platform\windows\quick_verify.ps1
  ```

- **extract_for_mac.ps1** - Extract and package trained outputs for transfer to Mac.
  ```powershell
  .\scripts\platform\windows\extract_for_mac.ps1
  ```

### Unix/Mac (`scripts/platform/unix/`)

Shell scripts for Mac and Linux systems.

- **batch_verify.sh** - Batch verification script for Mac to validate Windows RTX 4080 training results.
  ```bash
  chmod +x scripts/platform/unix/batch_verify.sh
  ./scripts/platform/unix/batch_verify.sh outputs/production_cuda/
  ```

- **verify_all.sh** - Automated verification script for Windows 4080 outputs on Mac.
  ```bash
  ./scripts/platform/unix/verify_all.sh -f "F01,F02,F03"
  ```

- **run_all_baselines.sh** - Run all baseline methods.
  ```bash
  ./scripts/platform/unix/run_all_baselines.sh
  ```

---

## Archive (`scripts/archive/`)

Deprecated or one-off scripts kept for reference.

- **create_all_teacher_probs_fixed.py** - One-off script for creating teacher probability symlinks (deprecated).
- **cleanup_project.py** - Clean up project by removing junk files (.DS_Store, __pycache__, etc.).
  ```bash
  python scripts/archive/cleanup_project.py --exec
  ```

---

## Usage Examples

### Complete Workflow

1. **Prepare Data**
   ```bash
   # Build manifest
   python scripts/data/build_torgo_manifest.py --data_dir data/torgo_raw --output manifests/torgo_sentences.csv
   
   # Generate LOSO splits
   python scripts/data/generate_loso_splits.py --manifest manifests/torgo_sentences.csv --output_dir splits/
   ```

2. **Cache Data**
   ```bash
   # Cache teacher outputs
   python scripts/data/cache_teacher.py --manifest manifests/torgo_sentences.csv --device cuda
   
   # Cache features
   python scripts/data/cache_features.py --manifest manifests/torgo_sentences.csv --features mel --parallel 4
   ```

3. **Train Model**
   ```bash
   # Single fold
   python train_loso.py --config configs/pilot.yaml --fold F01
   
   # Or all folds (Windows)
   .\scripts\platform\windows\train_all_folds.ps1
   ```

4. **Analyze Results**
   ```bash
   # Compare methods
   python scripts/analysis/compare_methods.py --manifest manifests/torgo_pilot.csv --methods outputs/exp1/frame_probs --method-names "Our Model"
   
   # Verify checkpoint
   python scripts/core/verify_checkpoint.py --checkpoint outputs/production_cuda/checkpoints/fold_F01_latest_best.pt --fold F01
   ```

---

## Notes

- Scripts in `core/`, `data/`, and `analysis/` are Python scripts and should work on any platform.
- Scripts in `platform/` are platform-specific (PowerShell for Windows, Bash for Unix/Mac).
- All scripts support `--help` flag for detailed usage information.
- Most scripts are standalone and can be run directly without additional setup.
