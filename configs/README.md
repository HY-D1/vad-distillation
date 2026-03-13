# Configuration Files

This directory contains all training and experiment configuration files for the VAD distillation project.

## Quick Reference

| Config | Purpose | Device | Batch Size | When to Use |
|--------|---------|--------|------------|-------------|
| `quick_test.yaml` | Fast verification | CUDA | 16 | RTX 4080 quick sanity check (< 5 min) |
| `pilot.yaml` | Smoke testing | CPU/MPS | 8 | General smoke tests, macOS |
| `pilot_cuda.yaml` | Smoke testing | CUDA | 32 | RTX 4080 smoke tests |
| `production.yaml` | Full training | MPS | 16 | macOS full training |
| `production_cuda.yaml` | Full training | CUDA | 64 | RTX 4080 full training |
| `alpha_sweep.yaml` | Hyperparameter sweep | CPU | 8 | Testing different alpha values |
| `temperature_sweep.yaml` | Hyperparameter sweep | CPU | 8 | Testing different temperatures |
| `baselines.yaml` | Baseline evaluation | Auto | N/A | Running baseline methods |

## Configuration Details

### quick_test.yaml
**Purpose**: Fastest configuration for quick CUDA verification on RTX 4080

```yaml
seed: 6140
device: cuda
batch_size: 16
num_samples: 50
num_epochs: 2
```

**Use when**:
- Verifying CUDA setup on RTX 4080
- Quick debugging runs
- Testing code changes (< 5 minutes per fold)

---

### pilot.yaml
**Purpose**: General smoke testing on CPU or MPS (macOS)

```yaml
seed: 6140
device: cpu        # Can override with --device mps
batch_size: 8
num_samples: 100
num_epochs: 1
model:
  cnn_channels: [8, 16]
  gru_hidden: 24
```

**Use when**:
- First-time setup verification
- macOS smoke tests (use `--device mps`)
- CPU-only machines
- Quick functionality tests

---

### pilot_cuda.yaml
**Purpose**: RTX 4080 optimized smoke testing

```yaml
seed: 6140
device: cuda
num_workers: 2
batch_size: 32     # Larger for GPU efficiency
num_samples: 100
num_epochs: 1
model:
  cnn_channels: [8, 16]
  gru_hidden: 24
```

**Use when**:
- RTX 4080 smoke tests
- Verifying CUDA training pipeline
- Pre-production testing on Windows

---

### production.yaml
**Purpose**: Full training on macOS with MPS

```yaml
seed: 6140
device: mps
batch_size: 16     # Limited by MPS memory
num_samples: 2000
num_epochs: 20
model:
  cnn_channels: [14, 28]
  gru_hidden: 32
  dropout: 0.1
alpha: 0.6
temperature: 3.0
lr_scheduler:
  type: ReduceLROnPlateau
  factor: 0.5
  patience: 3
early_stopping:
  enabled: true
  patience: 5
```

**Use when**:
- Full macOS training
- Final model training on Apple Silicon
- Cross-platform verification of CUDA-trained models

---

### production_cuda.yaml
**Purpose**: Full training on RTX 4080 with CUDA

```yaml
seed: 6140
device: cuda
num_workers: 4     # Parallel data loading
batch_size: 64     # Larger batches for GPU efficiency
num_samples: 2000
num_epochs: 20
model:
  cnn_channels: [14, 28]
  gru_hidden: 32
  dropout: 0.1
alpha: 0.6
temperature: 3.0
lr_scheduler:
  type: ReduceLROnPlateau
  factor: 0.5
  patience: 3
early_stopping:
  enabled: true
  patience: 5
```

**Use when**:
- Full RTX 4080 training (primary training platform)
- Training all 15 TORGO folds
- Production model training

---

### alpha_sweep.yaml
**Purpose**: Test different alpha values for knowledge distillation

```yaml
seed: 6140
device: cpu
batch_size: 8
num_samples: 200
num_epochs: 3
alpha: 0.5         # Override with --alpha <value>
temperature: 3.0   # Fixed for alpha sweep
```

**Usage**:
```bash
python train.py --config configs/alpha_sweep.yaml --alpha 0.3
python train.py --config configs/alpha_sweep.yaml --alpha 0.5
python train.py --config configs/alpha_sweep.yaml --alpha 0.7
```

**Alpha values to sweep**:
| Alpha | Soft Loss Weight | Use Case |
|-------|------------------|----------|
| 0.0   | 0%               | Baseline (no distillation) |
| 0.3   | 30%              | Conservative distillation |
| 0.5   | 50%              | Balanced |
| 0.7   | 70%              | Aggressive distillation |
| 1.0   | 100%             | Pure distillation |

---

### temperature_sweep.yaml
**Purpose**: Test different temperature values for softening teacher outputs

```yaml
seed: 6140
device: cpu
batch_size: 8
num_samples: 200
num_epochs: 3
alpha: 0.5         # Fixed for temperature sweep
temperature: 3.0   # Override with --temperature <value>
```

**Usage**:
```bash
python train.py --config configs/temperature_sweep.yaml --temperature 1.0
python train.py --config configs/temperature_sweep.yaml --temperature 2.0
python train.py --config configs/temperature_sweep.yaml --temperature 3.0
python train.py --config configs/temperature_sweep.yaml --temperature 5.0
```

**Temperature values to sweep**:
| Temperature | Softening | Use Case |
|-------------|-----------|----------|
| 1.0         | None      | Standard training |
| 2.0         | Mild      | Conservative softening |
| 3.0         | Moderate  | Balanced (default) |
| 5.0         | Heavy     | Confident teacher |
| 10.0        | Extreme   | Highly calibrated teacher |

---

### baselines.yaml
**Purpose**: Configuration for baseline VAD methods comparison

```yaml
baselines:
  enabled: [energy, speechbrain, silero]
  
  energy:
    frame_hop_ms: 10
    frame_length_ms: 25
    threshold: 0.5
    hysteresis_high: 0.6
    hysteresis_low: 0.4
    min_speech_dur: 0.25
    min_silence_dur: 0.25
    smoothing_window: 3
    
  speechbrain:
    source: "speechbrain/vad-crdnn-libriparty"
    savedir: "pretrained_models/vad-crdnn-libriparty"
    
  silero:
    model: "silero-vad"
    threshold: 0.5

eval:
  metrics: [auc, f1, accuracy, precision, recall, miss_rate, far]
```

**Use when**:
- Running baseline comparisons
- Evaluating energy-based VAD
- Comparing against SpeechBrain and Silero

---

### week2_matrix.json
**Purpose**: Defines the 36-experiment matrix (3 α × 4 T × 3 folds)

**Structure**:
```json
{
  "experiments": [
    {
      "id": "alpha0.5_T1_F01",
      "alpha": 0.5,
      "temperature": 1.0,
      "fold": "F01"
    },
    ...
  ]
}
```

**Use with**:
```bash
python scripts/run_sweep.py \
    --param alpha --values 0.5 0.7 0.9 \
    --param temperature --values 1 2 3 5 \
    --folds F01 M01 FC01 \
    --base-config configs/pilot.yaml
```

---

## Configuration Parameters Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 6140 | Random seed for reproducibility |
| `device` | str | `cpu` | Compute device: `cpu`, `cuda`, `mps` |
| `num_workers` | int | 0 | DataLoader workers (4 for CUDA, 0 for MPS) |

### Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model.n_mels` | int | 40 | Number of mel filterbanks |
| `model.cnn_channels` | list | [8, 16] | CNN layer output channels |
| `model.gru_hidden` | int | 24 | GRU hidden dimension |
| `model.gru_layers` | int | 2 | Number of GRU layers |
| `model.dropout` | float | 0.0 | Dropout rate (0.1 in production) |

### Distillation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.5 | Weight for soft loss (0-1) |
| `temperature` | float | 3.0 | Softmax temperature for softening |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | `torgo_pilot` | Dataset identifier |
| `num_samples` | int | 100 | Samples per epoch (2000 for full) |
| `seq_len` | int | 100 | Sequence length in frames |
| `batch_size` | int | 8 | Training batch size |
| `learning_rate` | float | 0.001 | Initial learning rate |
| `num_epochs` | int | 1 | Training epochs (20 for full) |

### Data Paths

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `manifest` | str | `manifests/torgo_sentences.csv` | Dataset manifest |
| `teacher_probs_dir` | str | `teacher_probs/` | Cached teacher outputs |
| `output_dir` | str | `outputs/pilot/` | Training outputs |

### Learning Rate Scheduler

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_scheduler.type` | str | `ReduceLROnPlateau` | Scheduler type |
| `lr_scheduler.factor` | float | 0.5 | LR reduction factor |
| `lr_scheduler.patience` | int | 3 | Epochs before reducing |
| `lr_scheduler.min_lr` | float | 0.0001 | Minimum learning rate |

### Early Stopping

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `early_stopping.enabled` | bool | false | Enable early stopping |
| `early_stopping.patience` | int | 5 | Epochs before stopping |
| `early_stopping.metric` | str | `val_loss` | Metric to monitor |

### Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_interval` | int | 10 | Steps between logs |
| `save_interval` | int | 1 | Epochs between checkpoints |

---

## Platform-Specific Recommendations

### Windows RTX 4080 (CUDA)

**Recommended configs**:
- Quick test: `quick_test.yaml` (5 min)
- Smoke test: `pilot_cuda.yaml` (10-15 min)
- Full training: `production_cuda.yaml` (per fold)

**Key settings**:
- `device: cuda`
- `batch_size: 64`
- `num_workers: 4`

### macOS (MPS)

**Recommended configs**:
- Smoke test: `pilot.yaml` with `--device mps` (10-15 min)
- Full training: `production.yaml` (per fold)

**Key settings**:
- `device: mps`
- `batch_size: 16`
- `num_workers: 0`

### CPU-only

**Recommended configs**:
- Smoke test: `pilot.yaml` (20-30 min)
- Full training: Not recommended (very slow)

**Key settings**:
- `device: cpu`
- `batch_size: 8`
- `num_workers: 0`

---

## Creating Custom Configurations

Use the template in `templates/example_template.yaml`:

```yaml
# Copy and modify
seed: 6140
device: cuda

model:
  n_mels: 40
  cnn_channels: [14, 28]
  gru_hidden: 32
  gru_layers: 2
  dropout: 0.1

alpha: 0.6
temperature: 3.0

num_epochs: 20
batch_size: 64
learning_rate: 0.001

manifest: manifests/torgo_sentences.csv
teacher_probs_dir: teacher_probs/
output_dir: outputs/custom_experiment/
```

---

## Config Overrides

Override any config value via command line:

```bash
# Override device
python train_loso.py --config configs/pilot.yaml --device cuda

# Override batch size
python train_loso.py --config configs/pilot.yaml --batch_size 32

# Override alpha for sweep
python train_loso.py --config configs/alpha_sweep.yaml --alpha 0.7

# Override temperature
python train_loso.py --config configs/temperature_sweep.yaml --temperature 5.0
```

---

*Last updated: 2026-03-06*
*Seed: 6140 | Model Size: 473 KB | Folds: 15*
