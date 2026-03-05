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

## Quick Start

```bash
# 1. Validate TORGO data setup
python scripts/validate_torgo_setup.py

# 2. Build dataset manifest
python scripts/build_torgo_manifest.py \
    --data_dir data/torgo_raw \
    --output manifests/torgo_sentences.csv

# 3. Train student (single fold smoke test)
python train_loso.py --config configs/pilot.yaml --fold F01
```

## Hyperparameter Sweep

Run the full 36-experiment sweep (3 α × 4 T × 3 folds):

```bash
python scripts/run_sweep.py \
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
python scripts/analyze_week2.py \
  --results-dir outputs/week2_full \
  --output-dir analysis/week2
```

## File Structure

```
├── configs/               # Training configurations
│   ├── pilot.yaml        # Base config for experiments
│   └── week2_matrix.json # 36 experiment definitions
├── data/                  # TORGO dataset (not in git)
│   └── torgo_raw/        # Raw audio files
├── local/                 # 📚 Documentation
│   ├── INDEX.md          # Master documentation index
│   ├── week1_scope_and_eval.md
│   ├── week2_execution_plan.md
│   ├── data_setup.md     # TORGO setup guide
│   └── CACHING.md        # Cache management
├── manifests/             # Dataset manifests (CSV)
├── models/                # Model architectures
│   └── tinyvad_student.py
├── notebooks/             # Analysis notebooks
├── outputs/               # Training outputs
├── scripts/               # Utility scripts
│   ├── build_torgo_manifest.py
│   ├── cache_features.py
│   ├── cache_manager.py
│   ├── cache_teacher.py
│   ├── run_sweep.py      # Week 2 sweep runner
│   └── analyze_week2.py  # Results analysis
├── splits/                # LOSO splits (JSON)
├── teacher_probs/         # Cached Silero outputs
├── teacher_hard_labels/   # Thresholded teacher labels
├── train_loso.py          # 🎯 Main training script
└── requirements.txt
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_loso.py` | Main training script with LOSO support |
| `scripts/run_sweep.py` | Hyperparameter sweep runner |
| `scripts/analyze_week2.py` | Results analysis and visualization |
| `scripts/cache_manager.py` | Cache status, verify, clean |
| `scripts/build_torgo_manifest.py` | Generate dataset manifest |
| `scripts/validate_torgo_setup.py` | Validate TORGO installation |

## Design Decisions

1. **Dataset**: TORGO sentences (continuous speech) as primary dataset
2. **Evaluation**: Speaker-independent via Leave-One-Speaker-Out (LOSO)
3. **Distillation**: Soft labels with temperature T, loss = (1-α)×BCE(hard) + α×BCE(soft)
4. **Architecture**: CNN + GRU style student (TinyVAD-inspired)

## License

MIT License - See [LICENSE](LICENSE)
