# Compact VAD for Atypical Speech via Knowledge Distillation

This project builds a compact Voice Activity Detection (VAD) model optimized for atypical speech (e.g., dysarthric/Parkinsonian) by distilling knowledge from Silero VAD into a smaller "student" model.

## Project Goal

Develop a lightweight VAD (≤ 500 KB) that maintains competitive accuracy on atypical speech with low latency (≤ 10 ms/frame on CPU) by training on continuous speech from the TORGO dataset.

## Key Deliverables

| Metric | Target |
|--------|--------|
| Student Model Size | ≤ 500 KB |
| AUC Drop (vs Silero on atypical) | ≤ 10% |
| CPU Latency | ≤ 10 ms/frame |
| Atypical Miss Rate | Lower than Silero baseline |

## Project Structure

```
├── data/                   # Dataset storage (local only, not in git)
│   └── torgo_raw/         # TORGO audio files
│   └── README.md          # Data acquisition instructions
├── manifests/              # Dataset manifests (CSV)
├── splits/                 # Train/test splits (LOSO folds)
├── teacher_probs/          # Cached Silero outputs
├── teacher_hard_labels/    # Thresholded teacher labels
├── scripts/                # Utility scripts
│   ├── build_torgo_manifest.py
│   ├── run_silero_teacher.py
│   ├── cache_features.py
│   └── cache_teacher.py
├── notebooks/              # Analysis notebooks
│   ├── eda_torgo_sentences.ipynb
│   └── baseline_silero_metrics.ipynb
├── models/                 # Model architectures
│   └── tinyvad_student.py
├── configs/                # Training configurations
│   └── pilot.yaml
├── train.py                # Main training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Quick Start

### 1. Data Setup

See `data/README.md` for TORGO dataset acquisition and placement.

### 2. Build Manifest

```bash
python scripts/build_torgo_manifest.py \
    --data_dir data/torgo_raw \
    --output manifests/torgo_sentences.csv
```

### 3. Create LOSO Splits

Splits are generated as JSON files in `splits/` directory, one per held-out speaker.

### 4. Run Baseline (Silero Teacher)

```bash
python scripts/run_silero_teacher.py \
    --manifest manifests/torgo_sentences.csv \
    --output_dir teacher_probs/
```

### 5. Train Student (Smoke Test)

```bash
python train.py --config configs/pilot.yaml
```

## Key Design Decisions

1. **Dataset Priority**: TORGO sentences (continuous speech) as primary dataset. Saarbruecken Voice Database (SVD) is secondary due to isolated vowels limitation.

2. **Evaluation**: Speaker-independent via Leave-One-Speaker-Out (LOSO) cross-validation.

3. **Distillation**: Soft labels with temperature T (configurable), loss = (1-α)*BCE(hard) + α*BCE(soft).

4. **Architecture**: CNN + GRU style student (TinyVAD-inspired).

## Project Configuration

Edit `configs/pilot.yaml` to adjust:
- `alpha`: Weight for soft-label distillation loss (default: 0.5)
- `temperature`: Temperature for softening teacher outputs (default: 3.0)
- `model`: Student architecture parameters (channels, hidden dim, layers)

## License

MIT License - See LICENSE
