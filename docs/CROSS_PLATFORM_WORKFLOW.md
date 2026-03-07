# Cross-Platform Workflow Guide

**Mac M1 Pro (Source) → Windows RTX 4080 (Training) → Mac M1 Pro (Verification)**

This guide covers the complete data transfer and workflow for training VAD models on Windows RTX 4080 and verifying results on Mac M1 Pro.

---

## Tracking Training Outputs with Git LFS

### Overview
Training outputs (checkpoints, logs) are now tracked in the repository using Git LFS, enabling seamless cross-device workflows.

### What's Tracked
- **Model checkpoints** (*.pt) - ~1.4 MB each, stored in LFS
- **Training logs** (*.csv) - Small text files
- **Summary metrics** (*.json) - Small text files
- **Predictions** (*.npz) - Large binary, stored in LFS

### Setup (One-time)
```bash
# Install Git LFS
git lfs install

# Pull LFS files
git lfs pull
```

### Workflow: Windows → macOS
```bash
# On Windows (after training)
git add outputs/production_cuda/
git commit -m "Add training results for folds F01, F03, ..."
git push

# On macOS
git pull
git lfs pull
# Now all checkpoints and logs are available locally
```

### Storage Quota
- GitHub LFS free tier: 1 GB storage, 1 GB bandwidth/month
- Current project size: ~45 MB (checkpoints) + ~140 MB (predictions)
- Well within free tier limits

### Selective Tracking
Only `outputs/production_cuda/` is tracked.
Baseline outputs and archives remain gitignored (can be regenerated).

---

## Prerequisites

### Environment Setup

**On Mac (Source):**
- TORGO dataset extracted to `data/torgo_raw/`
- Teacher probabilities cached in `teacher_probs/`
- LOSO splits generated in `splits/`
- Dataset manifests in `manifests/`

**On Windows (Training):**
- TORGO dataset extracted to `data/torgo_raw/` (same structure as Mac)
- Python environment with PyTorch CUDA installed
- RTX 4080 with updated drivers

### Directory Structure (Both Platforms)

```
vad-distillation/
├── data/torgo_raw/          # EXCLUDED from transfer (on both machines)
├── teacher_probs/             # Mac → Windows
├── splits/                    # Mac → Windows
├── manifests/                 # Mac → Windows
├── configs/                   # Mac → Windows
├── outputs/                   # Windows → Mac
└── scripts/                   # Git-tracked (no transfer needed)
```

---

## Section 1: Mac → Windows (Before Training)

### Files to Copy from Mac to Windows

| Directory | Size | Description |
|-----------|------|-------------|
| `teacher_probs/` | ~8 MB (~15,086 files) | Cached Silero VAD outputs per utterance |
| `splits/` | ~2 MB (16 files) | LOSO fold definitions (15 folds + all) |
| `manifests/` | ~1 MB (3 files) | Dataset manifests (CSV) |
| `configs/*.yaml` | ~10 KB (8 files) | Training configuration files |

**Total transfer size: ~11 MB**

### What to Exclude

| Directory | Reason |
|-----------|--------|
| `data/torgo_raw/` | Too large (~2 GB); must exist on both machines with identical structure |
| `outputs/` | Will be generated on Windows |
| `pretrained_models/` | Will be auto-downloaded if needed |
| `.git/` | Git-tracked files |

### Transfer Methods

#### Option A: rsync over SSH (Recommended)

**From Mac to Windows:**

```bash
# 1. Navigate to project root on Mac
cd ~/dev/vad-distillation

# 2. Transfer teacher_probs (cached Silero outputs)
rsync -avz --progress \
    teacher_probs/ \
    user@windows-host:vad-distillation/teacher_probs/

# 3. Transfer splits (LOSO fold definitions)
rsync -avz --progress \
    splits/ \
    user@windows-host:vad-distillation/splits/

# 4. Transfer manifests (dataset CSVs)
rsync -avz --progress \
    manifests/ \
    user@windows-host:vad-distillation/manifests/

# 5. Transfer config files
rsync -avz --progress \
    configs/*.yaml \
    user@windows-host:vad-distillation/configs/

# Or do it all in one command:
rsync -avz --progress \
    --include='teacher_probs/***' \
    --include='splits/***' \
    --include='manifests/***' \
    --include='configs/*.yaml' \
    --exclude='*' \
    . user@windows-host:vad-distillation/
```

**Note:** On Windows, you need an SSH server (OpenSSH server) installed and running.

#### Option B: External Drive / USB

```bash
# On Mac - copy to external drive
cp -r teacher_probs splits manifests configs /Volumes/ExternalDrive/vad-transfer/

# On Windows - copy from external drive
robocopy E:\vad-transfer\teacher_probs teacher_probs /E
robocopy E:\vad-transfer\splits splits /E
robocopy E:\vad-transfer\manifests manifests /E
copy E:\vad-transfer\configs\*.yaml configs\
```

#### Option C: Cloud Storage (OneDrive/Dropbox/Google Drive)

```bash
# On Mac
cp -r teacher_probs splits manifests configs ~/Dropbox/vad-transfer/

# On Windows (after sync)
robocopy %USERPROFILE%\Dropbox\vad-transfer\teacher_probs teacher_probs /E
robocopy %USERPROFILE%\Dropbox\vad-transfer\splits splits /E
robocopy %USERPROFILE%\Dropbox\vad-transfer\manifests manifests /E
copy %USERPROFILE%\Dropbox\vad-transfer\configs\*.yaml configs\
```

#### Option D: Local Network (SMB/Samba)

```bash
# On Mac - mount Windows share
mkdir -p /mnt/windows
mount_smbfs //user@windows-host/shared /mnt/windows

# Copy files
rsync -avz teacher_probs/ /mnt/windows/vad-distillation/teacher_probs/
rsync -avz splits/ /mnt/windows/vad-distillation/splits/
rsync -avz manifests/ /mnt/windows/vad-distillation/manifests/
rsync -avz configs/*.yaml /mnt/windows/vad-distillation/configs/
```

### Post-Transfer Verification (On Windows)

```powershell
# Verify file counts
(Get-ChildItem teacher_probs -File).Count  # Should be ~15086
(Get-ChildItem splits -File).Count         # Should be 16
(Get-ChildItem manifests -File).Count      # Should be 3
(Get-ChildItem configs\*.yaml).Count       # Should be 8

# Verify TORGO data exists on Windows
Test-Path data/torgo_raw/F01/Session1/wav_headMic
```

---

## Section 2: Windows Training

### Pre-Training Checklist

```powershell
# 1. Activate conda environment
conda activate vad

# 2. Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# 3. Verify data structure
python scripts/data/validate_torgo_setup.py

# Expected output:
# ✓ TORGO data structure validated
# ✓ Found 15 speakers
# ✓ Manifests loaded successfully
```

### Training Steps

#### Step 1: Quick Test (< 5 minutes)

Verifies the entire pipeline works without errors.

```powershell
python train_loso.py --config configs/quick_test.yaml --fold F01
```

**What to check:**
- No CUDA out-of-memory errors
- Loss decreases (not NaN)
- AUC > 0.5 (model is learning)
- Completes without errors

#### Step 2: Pilot Test (10-15 minutes)

Tests with pilot configuration on a single fold.

```powershell
python train_loso.py --config configs/pilot_cuda.yaml --fold F01 --epochs 5
```

**Configuration:** `configs/pilot_cuda.yaml`
- Batch size: 64 (RTX 4080 optimized)
- Num workers: 4
- Epochs: 5 (pilot)
- Device: cuda

**Expected results:**
- Val AUC: > 0.80
- Training time: ~2-3 min per epoch
- Model size: ~473 KB

#### Step 3: Full Training (All 15 Folds)

**Option A: Using PowerShell Script (Recommended)**

Create `scripts/platform/windows/train_all_folds.ps1`:

```powershell
# scripts/platform/windows/train_all_folds.ps1
$folds = @("F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05", "FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04")
$config = "configs/production_cuda.yaml"

foreach ($fold in $folds) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Training fold: $fold" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    python train_loso.py --config $config --fold $fold
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Training failed for fold $fold" -ForegroundColor Red
        exit 1
    }
}

Write-Host "All folds completed successfully!" -ForegroundColor Green
```

Run the script:

```powershell
# Allow script execution (if not already enabled)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run training
.\scripts\train_all_folds.ps1
```

**Option B: PowerShell Loop (All Folds)**

```powershell
# Train all remaining folds automatically
$folds = @('F04','FC01','FC02','FC03','M01','M02','M03','M04','M05','MC01','MC02','MC03','MC04')
foreach ($fold in $folds) {
    Write-Host "Training fold: $fold" -ForegroundColor Green
    python train_loso.py --config configs/production_cuda.yaml --fold $fold
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed for fold: $fold" -ForegroundColor Red
        break
    }
}
```

**Option C: Train Specific Folds**

```powershell
# Train only specific folds (modify the array as needed)
$folds = @('F04','M01','M02')
foreach ($fold in $folds) {
    Write-Host "Training fold: $fold" -ForegroundColor Green
    python train_loso.py --config configs/production_cuda.yaml --fold $fold
}
```

**Option D: Parallel Training (Advanced)**

```powershell
# Train multiple folds in parallel (if VRAM allows)
# Note: RTX 4080 (16GB) can typically handle 2 concurrent trainings

Start-Process powershell -ArgumentList "python train_loso.py --config configs/production_cuda.yaml --fold F01"
Start-Process powershell -ArgumentList "python train_loso.py --config configs/production_cuda.yaml --fold F03"
# ... etc
```

### Training Monitoring

**Watch training logs:**

```powershell
# View live training log
tail -f outputs/production_cuda/logs/fold_F01.csv

# Or use Get-Content on Windows
Get-Content outputs/production_cuda/logs/fold_F01.csv -Wait
```

**Expected output structure after training:**

```
outputs/production_cuda/
├── checkpoints/
│   ├── fold_F01_latest_best.pt
│   ├── fold_F01_latest.pt
│   ├── fold_F03_best.pt
│   ├── fold_F03_latest.pt
│   └── ... (30 files total: best + latest per fold)
├── logs/
│   ├── fold_F01.csv          # Per-epoch metrics
│   ├── fold_F01_summary.json # Final summary
│   ├── fold_F01_predictions.npz
│   ├── fold_F03.csv
│   ├── fold_F03_summary.json
│   ├── fold_F03_predictions.npz
│   └── ... (45 files total: csv + json + npz per fold)
└── config.yaml               # Copy of effective config
```

### Expected Training Time

| Configuration | Time per Fold | Total Time (15 folds) |
|--------------|---------------|----------------------|
| quick_test.yaml | ~5 min | ~75 min |
| pilot_cuda.yaml | ~15 min | ~4 hours |
| production_cuda.yaml | ~45 min | ~11 hours |

---

## Section 3: Windows → Mac (After Training)

### Files to Copy Back to Mac

| Directory/Pattern | Approx. Size | Description |
|-------------------|--------------|-------------|
| `outputs/production_cuda/checkpoints/*_best.pt` | ~750 KB × 15 | Best model per fold |
| `outputs/production_cuda/checkpoints/*_latest.pt` | ~750 KB × 15 | Latest checkpoint per fold |
| `outputs/production_cuda/logs/*.csv` | ~50 KB × 15 | Training history |
| `outputs/production_cuda/logs/*.json` | ~2 KB × 15 | Summary metrics |
| `outputs/production_cuda/logs/*.npz` | ~500 KB × 15 | Test predictions |

**Total transfer size: ~20-30 MB**

### Transfer Commands

#### Option A: Git LFS (Recommended)

```bash
# On Windows - add outputs to git tracking
git add outputs/production_cuda/
git commit -m "Add training results for all 15 folds"
git push

# On Mac - pull the updates
git pull
git lfs pull
```

#### Option B: rsync: Windows → Mac

```powershell
# On Windows (using WSL or Git Bash)
rsync -avz --progress \
    outputs/production_cuda/ \
    user@mac-host:vad-distillation/outputs/production_cuda/
```

**Or from Mac:**

```bash
# Pull from Windows to Mac
rsync -avz --progress \
    user@windows-host:vad-distillation/outputs/production_cuda/ \
    outputs/production_cuda/
```

#### Option C: Robocopy + External Drive

```powershell
# On Windows - copy to external drive
robocopy outputs\production_cuda\checkpoints E:\vad-results\checkpoints *.pt /E
robocopy outputs\production_cuda\logs E:\vad-results\logs /E
copy outputs\production_cuda\config.yaml E:\vad-results\

# On Mac - copy from external drive
cp -r /Volumes/ExternalDrive/vad-results/* outputs/production_cuda/
```

#### Option D: Compress and Transfer

```powershell
# On Windows - create archive
Compress-Archive -Path outputs/production_cuda -DestinationPath vad-results.zip

# Transfer via any method (SCP, cloud, USB)
scp vad-results.zip user@mac-host:vad-distillation/

# On Mac - extract
unzip vad-results.zip -d outputs/
```

### Post-Transfer Verification (On Mac)

```bash
# Verify checkpoint count
ls outputs/production_cuda/checkpoints/*_best.pt | wc -l  # Should be 15

# Verify log files
ls outputs/production_cuda/logs/*.json | wc -l  # Should be 15

# Check file sizes
find outputs/production_cuda/checkpoints -name "*.pt" -exec ls -lh {} \;

# Verify one summary file
cat outputs/production_cuda/logs/fold_F01_summary.json | python -m json.tool
```

---

## Section 4: Mac Verification

### Prerequisites

```bash
# Ensure you have the analysis script
ls scripts/analysis/verify_4080_results.py

# Or use the compare_methods.py script
ls scripts/analysis/compare_methods.py
```

### Verification Commands

#### Step 1: Run Verification Script

```bash
python scripts/analysis/verify_4080_results.py \
    --results-dir outputs/production_cuda
```

**Expected output:**
```
========================================
RTX 4080 Training Results Verification
========================================
Fold    Test AUC    Val AUC    Model Size
F01     0.9234      0.9312     473 KB
F03     0.9187      0.9256     473 KB
...     ...         ...        ...
----------------------------------------
Mean    0.9212      0.9284     473 KB
Std     0.0089      0.0091     0 KB

========================================
Cross-Platform Consistency Check
========================================
✓ All 15 folds present
✓ Model sizes consistent (473 KB)
✓ Metrics within tolerance
```

#### Step 2: Compare with Baselines

```bash
python scripts/analysis/compare_methods.py \
    --manifest manifests/torgo_pilot.csv \
    --methods outputs/baselines/silero/,outputs/production_cuda/ \
    --method-names "Silero,Student" \
    --output-dir analysis/comparison
```

#### Step 3: Generate Analysis Plots

```bash
# Create comparison visualizations
python notebooks/analyze_results.py \
    --results-dir outputs/production_cuda \
    --output-dir analysis/figures
```

### Expected Metric Tolerances

When comparing results across platforms or rerunning experiments, use these tolerances:

| Metric | Tolerance | Notes |
|--------|-----------|-------|
| AUC | ±0.005 | Area Under ROC Curve |
| F1 Score | ±0.01 | Harmonic mean of precision/recall |
| Accuracy | ±0.01 | Overall classification accuracy |
| Miss Rate | ±0.02 | False negative rate (atypical speech) |
| Model Size | ±1 KB | Should be exactly 473 KB |
| Training Loss | ±0.05 | Final training loss |
| Val Loss | ±0.02 | Final validation loss |

**Cross-Platform Consistency:**
- Same random seed (6140) should produce **identical** results on both platforms
- Minor differences (< 0.001) may occur due to:
  - Different CUDA/cuDNN versions
  - Different PyTorch builds
  - Hardware-specific optimizations

### Verification Checklist

```bash
# 1. Check all folds completed
python -c "
import json, glob, sys
files = glob.glob('outputs/production_cuda/logs/*_summary.json')
print(f'Folds completed: {len(files)}/15')
sys.exit(0 if len(files) == 15 else 1)
"

# 2. Check model sizes
python -c "
import os, glob
for f in glob.glob('outputs/production_cuda/checkpoints/*_best.pt'):
    size = os.path.getsize(f) / 1024
    if abs(size - 473) > 1:
        print(f'WARNING: {f} has unexpected size: {size:.0f} KB')
    else:
        print(f'✓ {os.path.basename(f)}: {size:.0f} KB')
"

# 3. Check metrics are reasonable
python -c "
import json, glob
for f in sorted(glob.glob('outputs/production_cuda/logs/*_summary.json')):
    with open(f) as fp:
        data = json.load(fp)
    fold = data.get('fold', 'unknown')
    test_auc = data.get('test_auc', 0)
    val_auc = data.get('val_auc', 0)
    status = '✓' if test_auc > 0.85 and val_auc > 0.85 else '✗'
    print(f'{status} {fold}: Test AUC={test_auc:.4f}, Val AUC={val_auc:.4f}')
"
```

---

## Quick Reference: One-Page Summary

### Before Training (Mac → Windows)

```bash
# Mac side
rsync -avz teacher_probs/ splits/ manifests/ configs/*.yaml user@windows:vad-distillation/
```

### Training (Windows)

```powershell
# Quick test
python train_loso.py --config configs/quick_test.yaml --fold F01

# Pilot test
python train_loso.py --config configs/pilot_cuda.yaml --fold F01 --epochs 5

# Full training (all folds)
$folds = @('F01','F03','F04','M01','M02','M03','M04','M05','FC01','FC02','FC03','MC01','MC02','MC03','MC04')
foreach ($fold in $folds) {
    Write-Host "Training fold: $fold" -ForegroundColor Green
    python train_loso.py --config configs/production_cuda.yaml --fold $fold
}
```

### After Training (Windows → Mac)

```bash
# Option A: Git LFS (recommended)
git pull && git lfs pull

# Option B: rsync
rsync -avz user@windows:vad-distillation/outputs/production_cuda/ outputs/production_cuda/
```

### Verification (Mac)

```bash
# Verify results
python scripts/analysis/verify_4080_results.py --results-dir outputs/production_cuda
```

---

## Troubleshooting

### Transfer Issues

| Issue | Solution |
|-------|----------|
| SSH connection refused | Enable OpenSSH server on Windows |
| Permission denied | Check SSH key authentication |
| Partial transfer | Use `rsync --partial` for resume capability |
| Slow transfer | Use compression `-z` flag with rsync |

### Git LFS Issues

| Issue | Solution |
|-------|----------|
| LFS files not downloaded | Run `git lfs pull` after `git pull` |
| LFS quota exceeded | Check quota with `git lfs quota` |
| LFS not installed | Run `git lfs install` first |
| Large files not tracked | Check `.gitattributes` for proper patterns |

### Training Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch_size to 32 in config |
| Data not found | Verify `data/torgo_raw/` exists with correct structure |
| Teacher probs missing | Re-run transfer from Mac |
| NaN loss | Reduce learning rate in config |

### Verification Issues

| Issue | Solution |
|-------|----------|
| Missing folds | Check Windows training completed all 15 folds |
| Wrong model size | Verify student model architecture unchanged |
| Low AUC (< 0.85) | Check if using correct config; may need more epochs |
| MPS errors | Ensure PyTorch MPS backend is available |

---

## Appendix A: Directory Size Reference

```bash
# Mac side (before transfer)
du -sh teacher_probs/  # ~8 MB
du -sh splits/         # ~2 MB
du -sh manifests/      # ~1 MB
du -sh configs/        # ~10 KB

# Windows side (after training)
du -sh outputs/production_cuda/checkpoints/  # ~22 MB
du -sh outputs/production_cuda/logs/         # ~8 MB
```

## Appendix B: Speaker ID Reference

| Fold | Speaker Type | Gender |
|------|--------------|--------|
| F01  | Dysarthric   | Female |
| F03  | Dysarthric   | Female |
| F04  | Dysarthric   | Female |
| M01  | Dysarthric   | Male   |
| M02  | Dysarthric   | Male   |
| M03  | Dysarthric   | Male   |
| M04  | Dysarthric   | Male   |
| M05  | Dysarthric   | Male   |
| FC01 | Control      | Female |
| FC02 | Control      | Female |
| FC03 | Control      | Female |
| MC01 | Control      | Male   |
| MC02 | Control      | Male   |
| MC03 | Control      | Male   |
| MC04 | Control      | Male   |

---

*Last updated: 2026-03-07*
*Target Model Size: 473 KB | Random Seed: 6140 | Total Folds: 15*
