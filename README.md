# Compact VAD for Atypical Speech via Knowledge Distillation

This project builds a compact Voice Activity Detection (VAD) model optimized for atypical speech (e.g., dysarthric/Parkinsonian) by distilling knowledge from Silero VAD into a smaller "student" model.

## Project Goal

Develop a lightweight VAD (≤ 500 KB) that maintains competitive accuracy on atypical speech with low latency (≤ 10 ms/frame on CPU) by training on continuous speech from the TORGO dataset.

## Key Deliverables

| Metric | Target | Achieved |
|--------|--------|----------|
| Student Model Size | ≤ 500 KB | **473 KB** |
| AUC Drop (vs Silero on atypical) | ≤ 10% | TBD |
| CPU Latency | ≤ 10 ms/frame | TBD |
| Atypical Miss Rate | Lower than Silero baseline | TBD |

## Quick Start

```bash
# 1. Validate TORGO data setup
python scripts/data/validate_torgo_setup.py

# 2. Build dataset manifest
python scripts/data/build_torgo_manifest.py \
    --data_dir data/torgo_raw \
    --output manifests/torgo_sentences.csv

# 3. Train student (single fold smoke test)
python train_loso.py --config configs/pilot.yaml --fold F01
```

---

## Usage (macOS with MPS)

### Prerequisites

```bash
# Check Python version (should be 3.9+)
python3 --version

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon (M1/M2/M3) - MPS is auto-detected
# For Intel Macs - uses CPU
```

### Quick Start

```bash
# Validate setup
python scripts/data/validate_torgo_setup.py

# Run smoke test (1-2 minutes)
python train_loso.py --config configs/pilot.yaml --fold F01 --test

# Train first model (30 min on Apple Silicon, 60 min on Intel)
python train_loso.py --config configs/pilot.yaml --fold F01 --epochs 10
```

### Running Baselines

```bash
# Energy baseline
python scripts/core/run_baseline.py \
  --method energy \
  --manifest manifests/torgo_pilot.csv \
  --output-dir outputs/baselines/energy/

# SpeechBrain baseline
pip install speechbrain
python scripts/core/run_baseline.py \
  --method speechbrain \
  --manifest manifests/torgo_pilot.csv \
  --output-dir outputs/baselines/speechbrain/
```

### Troubleshooting macOS

| Issue | Solution |
|-------|----------|
| MPS not available | Falls back to CPU automatically |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Permission denied | Run `chmod +x scripts/**/*.py` |

---

## Usage (Windows with RTX 4080 CUDA)

### Prerequisites

```cmd
# Check Python version
python --version

# Install dependencies (includes CUDA-enabled PyTorch)
pip install -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note**: On Windows, use `python` not `python3`. RTX 4080 uses CUDA with optimized batch size of 64.

### Quick Start

```cmd
# Validate setup
python scripts/data/validate_torgo_setup.py

# Quick test on CUDA (RTX 4080)
python train_loso.py --config configs/quick_test.yaml --fold F01

# Smoke test with CUDA
python train_loso.py --config configs/pilot_cuda.yaml --fold F01 --test

# Train first model on CUDA (10-15 min on RTX 4080)
python train_loso.py --config configs/pilot_cuda.yaml --fold F01 --epochs 10

# Full production training on CUDA
python train_loso.py --config configs/production_cuda.yaml --fold F01

# Train all folds (PowerShell)
$folds = @('F01','F03','F04','M01','M02','M03','M04','M05','FC01','FC02','FC03','MC01','MC02','MC03','MC04')
foreach ($fold in $folds) {
    python train_loso.py --config configs/production_cuda.yaml --fold $fold
}
```

### Running Baselines

```cmd
# Energy baseline
python scripts/core/run_baseline.py ^
  --method energy ^
  --manifest manifests/torgo_pilot.csv ^
  --output-dir outputs/baselines/energy/

# SpeechBrain baseline
pip install speechbrain
python scripts/core/run_baseline.py ^
  --method speechbrain ^
  --manifest manifests/torgo_pilot.csv ^
  --output-dir outputs/baselines/speechbrain/
```

### Troubleshooting Windows

| Issue | Solution |
|-------|----------|
| `python` not found | Use `py` or add Python to PATH |
| torchaudio DLL error | Reinstall: `pip uninstall torchaudio && pip install torchaudio` |
| CUDA out of memory | Reduce `batch_size` in config (default is 64 for RTX 4080) |
| Antivirus blocks Python | Add Python to antivirus exclusions |
| CSV has blank lines | Use `newline=''` (already fixed in code) |

---

## Cross-Platform Verification Workflow

This project supports training on Windows (RTX 4080 CUDA) and verification on macOS (MPS):

```
Windows RTX 4080          →            macOS MPS
(CUDA training)                       (Verification)
     │                                      ▲
     │ 1. Train all 15 folds                │
     │    (configs/production_cuda.yaml)    │
     │                                      │
     ▼                                      │
outputs/production_cuda/                   │
     │                                      │
     │ 2. Copy outputs to Mac               │
     │    (or shared cloud storage)         │
     │                                      │
     └──────────────────────────────────────┘
                                            │
     3. Verify on Mac                        │
        python scripts/analysis/compare_methods.py   │
        python notebooks/analyze_results.ipynb
```

### Key Differences by Platform

| Setting | macOS (MPS) | Windows (CUDA RTX 4080) |
|---------|-------------|-------------------------|
| Device | `mps` | `cuda` |
| Batch Size | 16 | 64 |
| Workers | 0 | 4 |
| Config | `production.yaml` | `production_cuda.yaml` |

---

## Baselines

We compare our TinyVAD student against three baselines:

### 1. Energy-based VAD (Course Requirement)
Simple energy thresholding with hysteresis and smoothing.
```bash
python scripts/core/run_baseline.py \
  --method energy \
  --manifest manifests/torgo_pilot.csv \
  --output-dir outputs/baselines/energy/
```

### 2. SpeechBrain CRDNN VAD (Reference Baseline)
Pretrained CRDNN model trained on LibriParty (F1=0.9477 on test set).
```bash
pip install speechbrain
python scripts/core/run_baseline.py \
  --method speechbrain \
  --manifest manifests/torgo_pilot.csv \
  --output-dir outputs/baselines/speechbrain/
```

### 3. Silero VAD (Teacher / Additional Reference)
Our distillation teacher.
```bash
python scripts/core/run_baseline.py \
  --method silero \
  --manifest manifests/torgo_pilot.csv \
  --output-dir outputs/baselines/silero/
```

### Compare All Methods
```bash
python scripts/analysis/compare_methods.py \
  --manifest manifests/torgo_pilot.csv \
  --methods outputs/baselines/energy/,outputs/baselines/speechbrain/,outputs/pilot/ \
  --method-names "Energy,SpeechBrain,Our Model" \
  --output-dir analysis/comparison \
  --proxy-labels teacher
```

---

## Hyperparameter Sweep

Run the full 36-experiment sweep (3 α × 4 T × 3 folds):

```bash
python scripts/core/run_sweep.py \
  --param alpha --values 0.5 0.7 0.9 \
  --param temperature --values 1 2 3 5 \
  --folds F01 M01 FC01 \
  --base-config configs/pilot.yaml \
  --output-dir outputs/week2_full \
  --epochs 50 \
  --patience 10
```

Analyze results:
```bash
python scripts/analysis/analyze_week2.py \
  --results-dir outputs/week2_full \
  --output-dir analysis/week2
```

---

## File Structure

```
├── analysis/               # Analysis outputs and visualizations
├── baselines/              # Baseline VAD implementations
│   ├── energy_vad.py
│   └── speechbrain_vad.py
├── configs/                # Training configurations
│   ├── pilot.yaml         # Base config for smoke tests (CPU/MPS)
│   ├── pilot_cuda.yaml    # RTX 4080 optimized pilot config
│   ├── production.yaml    # Full training config (MPS)
│   ├── production_cuda.yaml # Full training config (CUDA)
│   ├── quick_test.yaml    # Quick verification config
│   ├── baselines.yaml     # Baseline experiment config
│   ├── alpha_sweep.yaml   # Alpha hyperparameter sweep
│   ├── temperature_sweep.yaml # Temperature sweep
│   ├── week2_matrix.json  # 36-experiment matrix
│   └── templates/         # Config templates
├── data/                   # TORGO dataset (not in git)
│   ├── __init__.py
│   ├── torgo_dataset.py   # TORGO dataset class
│   └── torgo_raw/         # Raw audio files
├── models/                 # Model architectures
│   ├── __init__.py
│   ├── tinyvad_student.py # Main student model (~473KB)
│   └── losses.py          # Distillation loss functions
├── notebooks/              # Jupyter notebooks for analysis
├── outputs/                # Training outputs
├── pretrained_models/      # Downloaded baseline models
├── scripts/                # Utility and execution scripts
│   ├── core/               # Core training and experiment scripts
│   │   ├── run_baseline.py
│   │   ├── run_sweep.py
│   │   └── run_experiment.py
│   ├── data/               # Data processing scripts
│   │   ├── validate_torgo_setup.py
│   │   ├── build_torgo_manifest.py
│   │   └── cache_teacher.py
│   └── analysis/           # Analysis and comparison scripts
│       ├── compare_methods.py
│       └── analyze_week2.py
├── splits/                 # LOSO splits (JSON files for 15 folds)
├── teacher_probs/          # Cached Silero outputs
├── train_loso.py          # Main training script (LOSO)
├── train.py               # Simple wrapper for smoke tests
├── utils.py               # Shared utilities
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── AGENTS.md              # AI agent documentation
└── LICENSE                # MIT License
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_loso.py` | Main training script with LOSO support |
| `scripts/core/run_sweep.py` | Hyperparameter sweep runner |
| `scripts/analysis/analyze_week2.py` | Results analysis and visualization |
| `scripts/data/cache_manager.py` | Cache status, verify, clean |
| `scripts/data/build_torgo_manifest.py` | Generate dataset manifest |
| `scripts/data/validate_torgo_setup.py` | Validate TORGO installation |

## Design Decisions

1. **Dataset**: TORGO sentences (continuous speech) as primary dataset
2. **Evaluation**: Speaker-independent via Leave-One-Speaker-Out (LOSO) with 15 folds
3. **Distillation**: Soft labels with temperature T, loss = α×BCE(hard) + (1-α)×BCE(soft)
4. **Architecture**: CNN + GRU style student (TinyVAD-inspired), ~473KB
5. **Seed**: 6140 (for reproducibility)

## Checking Outputs

After training completes, verify that everything worked correctly by checking the outputs.

### Training Output Directory Structure

Training outputs are organized under the `output_dir` specified in your config:

```
outputs/pilot/                    outputs/production_cuda/
├── checkpoints/                  ├── checkpoints/
│   ├── fold_F01_latest_best.pt          │   ├── fold_F01_latest_best.pt
│   ├── fold_F01_latest.pt        │   ├── fold_F01_latest.pt
│   └── fold_F01_epoch_10.pt      │   └── ...
├── logs/                         ├── logs/
│   ├── fold_F01.csv              │   ├── fold_F01.csv
│   ├── fold_F01_summary.json     │   ├── fold_F01_summary.json
│   └── fold_F01_predictions.npz  │   └── ...
└── config.yaml                   └── config.yaml
```

### Key Metrics to Check

| Metric | Target | Notes |
|--------|--------|-------|
| **Val AUC** | > 0.85 (good), > 0.95 (excellent) | Primary metric for model selection |
| **Test AUC** | Similar to Val AUC | Large gap indicates overfitting |
| **Train Loss** | Decreasing over epochs | Should trend downward |
| **Model Size** | < 500 KB | Check summary JSON (target: ~473KB) |
| **Miss Rate** | < 0.20 (target) | Lower is better for atypical speech |

**Quick validation checklist:**
- Val AUC improved over baseline (0.5 = random)
- Test AUC within 5% of Val AUC
- No NaN values in loss
- Model size under 500 KB limit (~473KB expected)

### How to View Results

**View CSV logs (per-epoch metrics):**

```bash
# macOS/Linux
cat outputs/pilot/logs/fold_F01.csv

# Windows (Command Prompt)
type outputs\pilot\logs\fold_F01.csv

# Pretty-print with column alignment (requires csvkit)
csvlook outputs/pilot/logs/fold_F01.csv
```

CSV columns: `epoch`, `train_loss`, `train_hard_loss`, `train_soft_loss`, `val_auc`, `val_f1`, `val_miss_rate`, `val_false_alarm_rate`, `val_accuracy`, `learning_rate`, `time`

**View summary JSON (final results):**

```bash
# Pretty-print JSON
python -m json.tool outputs/pilot/logs/fold_F01_summary.json

# Extract specific metrics (requires jq)
cat outputs/pilot/logs/fold_F01_summary.json | jq '.best_val_auc, .test_metrics.auc'
```

Summary JSON contains:
- `fold_id`: Speaker fold identifier
- `train_speakers` / `val_speaker` / `test_speaker`: Data split info
- `num_parameters`: Model parameter count
- `model_size_mb`: Model size in megabytes
- `best_val_auc`: Best validation AUC achieved
- `test_metrics`: Full test set metrics (auc, f1, miss_rate, false_alarm_rate, accuracy)
- `config`: Full configuration used

**Load and inspect predictions:**

```python
import numpy as np

# Load predictions
data = np.load('outputs/pilot/logs/fold_F01_predictions.npz')

# Available arrays:
# - 'predictions': Binary predictions (0/1)
# - 'labels': Ground truth labels
# - 'probs': Predicted probabilities
# - 'utt_ids': Utterance identifiers

print(f"Predictions shape: {data['predictions'].shape}")
print(f"Test AUC: {compute_auc(data['labels'], data['probs']):.4f}")
```

**Check model size:**

```bash
# File size on disk
ls -lh outputs/pilot/checkpoints/fold_F01_latest_best.pt

# Or in Python
import torch
checkpoint = torch.load('outputs/pilot/checkpoints/fold_F01_latest_best.pt', weights_only=True)
print(f"Parameters: {checkpoint['num_parameters']:,}")
print(f"Size: {checkpoint['model_size_mb']:.2f} MB")
```

### Common Output Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **NaN in loss** | Loss shows `nan` in CSV | Reduce learning rate (e.g., 0.001 → 0.0001) |
| **AUC = 0.5** | Model not learning; random predictions | Check data loading, increase model capacity, check labels |
| **Out of memory** | CUDA OOM error | Reduce `batch_size` (RTX 4080 default is 64) |
| **Missing files** | FileNotFoundError during evaluation | Check `manifest`, `teacher_probs_dir`, and `splits/` paths |
| **Val AUC << Train** | Large generalization gap | Add dropout, reduce model size, early stopping |
| **Slow training** | Epoch time > 10 min | Reduce `num_samples`, use GPU (CUDA/MPS), reduce `seq_len` |

**Debugging tips:**

```bash
# Run test mode to verify setup
python train_loso.py --config configs/pilot.yaml --fold F01 --test

# Check data manifest
head manifests/torgo_pilot.csv

# Verify teacher probabilities exist
ls teacher_probs/ | head

# Check fold configuration
cat splits/fold_F01.json
```

### Comparing Multiple Runs

Use the comparison script to evaluate and compare different models or baselines:

```bash
# Compare your model against baselines
python scripts/analysis/compare_methods.py \
  --manifest manifests/torgo_pilot.csv \
  --methods outputs/baselines/energy/,outputs/baselines/speechbrain/,outputs/pilot/ \
  --method-names "Energy,SpeechBrain,Our Model" \
  --output-dir analysis/comparison \
  --proxy-labels teacher
```

**View comparison table:**

```bash
# Markdown table
cat analysis/comparison/comparison_table.md

# CSV for spreadsheet
cat analysis/comparison/comparison_table.csv
```

**Generated plots:**

```bash
# AUC comparison bar chart
open analysis/comparison/plots/auc_comparison.png

# Miss Rate vs False Alarm Rate scatter
open analysis/comparison/plots/miss_rate_far.png
```

**Example comparison output:**

```
| Method      | Size   | Latency | AUC    | F1     | Miss Rate | FAR    |
|-------------|--------|---------|--------|--------|-----------|--------|
| Energy      | N/A    | N/A     | 0.7234 | 0.6812 | 0.3124    | 0.1823 |
| SpeechBrain | 153 MB | 45 ms   | 0.8912 | 0.8543 | 0.1234    | 0.1421 |
| Our Model   | 473 KB | 8 ms    | 0.9234 | 0.8912 | 0.0891    | 0.1123 |
```

---

## Inspecting the Model

After training, you'll want to load, inspect, and verify your model. This section covers everything from basic loading to deployment-ready export.

### 1. Loading a Trained Model

Load a checkpoint and prepare it for inference:

```python
from models.tinyvad_student import create_student_model
import torch

# Create model architecture
model = create_student_model()

# Load trained weights
checkpoint_path = 'outputs/pilot/checkpoints/fold_F01_latest_best.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode
model.eval()

# Optional: Check training metadata
print(f"Training epochs: {checkpoint.get('epoch', 'N/A')}")
print(f"Best validation AUC: {checkpoint.get('best_auc', 'N/A'):.4f}")
print(f"Config: {checkpoint.get('config', {})}")
```

**Checkpoint contents:**
- `model_state_dict`: Model weights
- `epoch`: Training epoch when saved
- `best_auc`: Best validation AUC achieved
- `config`: Training configuration used

### 2. Checking Model Size

Verify your model meets the ≤ 500 KB target:

```python
# Get model size in KB
size_kb = model.get_model_size_kb()
print(f"Model size: {size_kb:.2f} KB")  # Should be ~473 KB

# Count trainable parameters
params = model.count_parameters()
print(f"Parameters: {params:,}")

# Get comprehensive model info
info = model.get_model_info()
print(f"""
Model Information:
  - Parameters: {info['parameters']:,}
  - Size: {info['size_kb']:.2f} KB ({info['size_mb']:.3f} MB)
  - CNN layers: {info['cnn_layers']} (time stride: {info['cnn_time_stride']}x)
  - GRU: {info['gru_layers']} layer(s), {info['gru_hidden']} hidden units
  - Mel bins: {info['n_mels']}
""")
```

**Expected output:**
```
Model size: ~473 KB
Parameters: ~118,000
```

### 3. Running Inference on New Audio

#### Option A: Using the Built-in `predict()` Method (Recommended)

The model includes a convenient `predict()` method that handles preprocessing:

```python
import numpy as np

# Load audio (any format supported by your audio library)
# Example: 16kHz mono audio
sample_rate = 16000
duration = 2.0  # seconds
audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

# Run inference
probs = model.predict(audio, return_numpy=True)

print(f"Input audio: {len(audio)/sample_rate:.2f}s")
print(f"Output frames: {len(probs)}")
print(f"Frame rate: {len(probs)/(len(audio)/sample_rate):.1f} fps")
print(f"Speech probability range: [{probs.min():.3f}, {probs.max():.3f}]")

# Apply threshold for binary decisions
threshold = 0.5
speech_frames = probs > threshold
print(f"Speech detected in {speech_frames.sum()}/{len(probs)} frames")
```

#### Option B: Manual Preprocessing (Full Control)

For custom preprocessing or batch processing:

```python
import torch
import torchaudio
import librosa
import numpy as np

# Load audio file
audio_path = 'audio.wav'
audio, sr = torchaudio.load(audio_path)

# Convert to mono if stereo
if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=True)

# Resample to 16kHz if needed
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(audio)
    sr = 16000

# Compute mel spectrogram
n_mels = 40
hop_length = 128  # 8ms @ 16kHz for 125 fps input

mel = librosa.feature.melspectrogram(
    y=audio.numpy().squeeze(),
    sr=sr,
    n_fft=512,
    hop_length=hop_length,
    n_mels=n_mels,
    fmin=0,
    fmax=8000,
    power=2.0
)

# Convert to dB and normalize
mel_db = librosa.power_to_db(mel, ref=np.max)
mel_norm = (mel_db + 80) / 80  # Normalize to [0, 1]
mel_norm = np.clip(mel_norm, 0, 1)

# Transpose to (time, n_mels)
mel_input = mel_norm.T  # Shape: (time_frames, n_mels)

# Run inference
model.eval()
with torch.no_grad():
    mel_tensor = torch.tensor(mel_input).unsqueeze(0).float()  # Add batch dim
    probs = model(mel_tensor)

print(f"Input shape: {mel_tensor.shape}")
print(f"Output shape: {probs.shape}")
print(f"Speech probabilities: {probs[0][:10]}")  # First 10 frames
```

#### Option C: Batch Processing Multiple Files

```python
import glob
from pathlib import Path

audio_files = glob.glob('data/audio/*.wav')
all_results = []

for audio_path in audio_files:
    # Load and preprocess
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    # Predict
    probs = model.predict(audio.numpy().squeeze())
    
    # Store results
    all_results.append({
        'file': Path(audio_path).name,
        'duration': len(audio) / 16000,
        'frames': len(probs),
        'speech_ratio': (probs > 0.5).mean(),
        'avg_confidence': probs.mean()
    })

# Print summary
for r in all_results:
    print(f"{r['file']}: {r['speech_ratio']*100:.1f}% speech, "
          f"avg conf: {r['avg_confidence']:.3f}")
```

### 4. Exporting for Deployment

Export your trained model to formats suitable for production deployment.

#### Export to TorchScript

TorchScript allows running the model in C++ or without Python dependencies:

```python
# Export using tracing (recommended for this model)
output_path = 'deploy/tinyvad_model.pt'
model.export_torchscript(output_path, method='trace')

print(f"TorchScript model saved to: {output_path}")

# Verify the exported model
import os
file_size = os.path.getsize(output_path) / 1024
print(f"File size: {file_size:.2f} KB")

# Test loading the TorchScript model
loaded_model = torch.jit.load(output_path)
test_input = torch.randn(1, 100, 40)
with torch.no_grad():
    output = loaded_model(test_input)
print(f"Verification output shape: {output.shape}")
```

#### Export to ONNX

ONNX enables cross-platform deployment (TensorRT, OpenVINO, ONNX Runtime):

```python
# Export to ONNX
onnx_path = 'deploy/tinyvad_model.onnx'
model.export_onnx(onnx_path, input_shape=(1, 100, 40))

print(f"ONNX model saved to: {onnx_path}")

# Verify with ONNX Runtime (optional)
try:
    import onnxruntime as ort
    
    # Create inference session
    session = ort.InferenceSession(onnx_path)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Test inference
    test_input = np.random.randn(1, 100, 40).astype(np.float32)
    outputs = session.run([output_name], {input_name: test_input})
    
    print(f"ONNX Runtime verification: output shape = {outputs[0].shape}")
    print("✓ ONNX export verified successfully!")
    
except ImportError:
    print("⚠ onnxruntime not installed, skipping verification")
```

#### Compare Export Formats

```python
import os
import time

formats = {
    'PyTorch (checkpoint)': 'outputs/pilot/checkpoints/fold_F01_latest_best.pt',
    'TorchScript': 'deploy/tinyvad_model.pt',
    'ONNX': 'deploy/tinyvad_model.onnx',
}

print(f"{'Format':<20} {'Size (KB)':>12} {'Status':>10}")
print("-" * 45)

for name, path in formats.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        status = "✓ Ready"
    else:
        size = 0
        status = "✗ Missing"
    print(f"{name:<20} {size:>12.2f} {status:>10}")
```

**Expected sizes:**
- PyTorch checkpoint: ~1,400 KB (includes optimizer state)
- TorchScript: ~473 KB (model only)
- ONNX: ~473 KB (model only)

### 5. Model Architecture Inspection

#### Visualize the Full Architecture

```python
# Print complete architecture
print(model)
```

**Example output:**
```
TinyVAD(
  (conv): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
  )
  (gru): GRU(960, 24, num_layers=2, batch_first=True)
  (fc): Linear(in_features=24, out_features=1, bias=True)
)
```

#### Detailed Layer Analysis

```python
# Inspect each layer's parameters
print("Layer-by-layer parameter count:")
print("-" * 60)

total_params = 0
for name, param in model.named_parameters():
    param_count = param.numel()
    total_params += param_count
    print(f"{name:<40} {param_count:>10,}")

print("-" * 60)
print(f"{'Total':<40} {total_params:>10,}")

# Calculate model size breakdown
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
print(f"\nSize breakdown:")
print(f"  Parameters: {param_size / 1024:.2f} KB")
print(f"  Buffers:    {buffer_size / 1024:.2f} KB")
print(f"  Total:      {(param_size + buffer_size) / 1024:.2f} KB")
```

#### Compute FLOPs Estimation

```python
# Get FLOPs estimate for performance analysis
flops = model.get_flops(input_shape=(1, 100, 40))

print(f"""
FLOPs Analysis (input: {flops['input_shape']}):
  CNN FLOPs:    {flops['cnn_flops']:>15,}
  GRU FLOPs:    {flops['gru_flops']:>15,}
  Linear FLOPs: {flops['linear_flops']:>15,}
  ─────────────────────────────────
  Total FLOPs:  {flops['total_flops']:>15,}
  Total MACs:   {flops['total_macs']:>15,}
  
Output frames: {flops['output_frames']}
""")

# Estimate latency (rough approximation)
# Assume 1 GFLOP/s = 1e9 FLOPs per second
estimated_time_ms = (flops['total_flops'] / 1e9) * 1000
print(f"Estimated inference time (1 GFLOP/s CPU): {estimated_time_ms:.3f} ms")
```

### Complete Inspection Script

Here's a complete script that performs all inspections at once:

```python
#!/usr/bin/env python3
"""Complete model inspection script."""

import torch
from models.tinyvad_student import create_student_model

def inspect_model(checkpoint_path: str):
    """Run comprehensive model inspection."""
    
    print("=" * 70)
    print("TinyVAD Model Inspection")
    print("=" * 70)
    
    # 1. Create and load model
    print("\n1. Loading Model")
    print("-" * 40)
    model = create_student_model()
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best AUC: {checkpoint.get('best_auc', 'N/A')}")
    else:
        print("✓ Using untrained model (random weights)")
    
    model.eval()
    
    # 2. Model info
    print("\n2. Model Information")
    print("-" * 40)
    info = model.get_model_info()
    print(f"Parameters:     {info['parameters']:,}")
    print(f"Size:           {info['size_kb']:.2f} KB ({info['size_mb']:.3f} MB)")
    print(f"CNN layers:     {info['cnn_layers']} (stride: {info['cnn_time_stride']}x)")
    print(f"GRU layers:     {info['gru_layers']}")
    print(f"GRU hidden:     {info['gru_hidden']}")
    print(f"Mel bins:       {info['n_mels']}")
    
    # Size check
    target_kb = 500
    if info['size_kb'] <= target_kb:
        print(f"✓ Size target met: {info['size_kb']:.2f} KB ≤ {target_kb} KB")
    else:
        print(f"✗ Size target NOT met: {info['size_kb']:.2f} KB > {target_kb} KB")
    
    # 3. FLOPs
    print("\n3. Computational Complexity")
    print("-" * 40)
    flops = model.get_flops(input_shape=(1, 100, 40))
    print(f"Total FLOPs:    {flops['total_flops']:,}")
    print(f"Total MACs:     {flops['total_macs']:,}")
    print(f"CNN FLOPs:      {flops['cnn_flops']:,}")
    print(f"GRU FLOPs:      {flops['gru_flops']:,}")
    
    # 4. Test forward pass
    print("\n4. Forward Pass Test")
    print("-" * 40)
    test_input = torch.randn(1, 100, 40)
    with torch.no_grad():
        output = model(test_input)
    print(f"Input shape:    {test_input.shape}")
    print(f"Output shape:   {output.shape}")
    print(f"Output range:   [{output.min():.4f}, {output.max():.4f}]")
    print("✓ Forward pass successful")
    
    # 5. Export tests
    print("\n5. Export Tests")
    print("-" * 40)
    
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # TorchScript
        try:
            ts_path = model.export_torchscript(tmpdir / 'test.pt')
            loaded = torch.jit.load(str(ts_path))
            print(f"✓ TorchScript export: {ts_path.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print(f"✗ TorchScript export failed: {e}")
        
        # ONNX
        try:
            onnx_path = model.export_onnx(tmpdir / 'test.onnx')
            print(f"✓ ONNX export: {onnx_path.stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
    
    print("\n" + "=" * 70)
    print("Inspection complete!")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    inspect_model(checkpoint)
```

**Usage:**
```bash
# Inspect untrained model
python inspect_model.py

# Inspect trained checkpoint
python inspect_model.py outputs/pilot/checkpoints/fold_F01_latest_best.pt
```

---

## Model Variants

The TinyVAD architecture supports multiple size variants to trade off between model capacity and deployment constraints. All variants maintain the ≤ 500 KB target size.

| Variant | CNN Channels | GRU Hidden | GRU Layers | Approx. Size | Use Case | Default Config |
|---------|-------------|------------|------------|--------------|----------|----------------|
| **Default** | [14, 28] | 32 | 2 | ~473 KB | Production training, best accuracy | `production_cuda.yaml` |
| **Small** | [12, 24] | 20 | 2 | ~300-400 KB | Balanced accuracy/size for edge devices | *Custom* |
| **Tiny** | [16] | 16 | 2 | ~100-200 KB | Ultra-compact, mobile/embedded | *Custom* |
| **Micro** | [8] | 8 | 2 | ~50-100 KB | Minimal footprint, feasibility testing | *Custom* |
| **Pilot** | [8, 16] | 24 | 2 | ~200-300 KB | Fast smoke testing, CPU-friendly | `pilot.yaml` |

### Selecting a Variant

**For production use**: Use the **Default** variant (`cnn_channels: [14, 28]`, `gru_hidden: 32`) which provides the best accuracy while meeting the ≤ 500 KB constraint. This is used in:
- `configs/production_cuda.yaml` (RTX 4080)
- `configs/production.yaml` (MPS/macOS)

**For quick testing**: Use the **Pilot** variant which trains faster on CPU:
```bash
python train_loso.py --config configs/pilot.yaml --fold F01 --test
```

**For custom deployments**: Create your own config by adjusting the `model` section:
```yaml
model:
  n_mels: 40
  cnn_channels: [12, 24]  # Try [16] for Tiny, [8] for Micro
  gru_hidden: 20          # Try 16 for Tiny, 8 for Micro
  gru_layers: 2
  dropout: 0.1
```

### Size vs. Accuracy Trade-offs

| Variant | Expected AUC | Latency (CPU) | Best For |
|---------|-------------|---------------|----------|
| Default | > 0.90 | ~8-10 ms/frame | Production, accuracy-critical |
| Small | > 0.85 | ~6-8 ms/frame | Edge devices, balanced |
| Tiny | > 0.80 | ~4-6 ms/frame | Mobile apps, real-time |
| Micro | > 0.75 | ~2-4 ms/frame | Feasibility, ultra-low latency |

---

## License

MIT License - See [LICENSE](LICENSE)
