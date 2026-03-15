# Personal VAD Milestone: TinyVAD vs Energy Baseline Comparison

A personal milestone project comparing a TinyVAD student model against an energy-based VAD baseline on atypical speech audio from the TORGO dataset.

## Project Overview

This repository contains a complete workflow for comparing two Voice Activity Detection approaches:
- **TinyVAD Student Model**: A lightweight neural network (~473 KB) trained via knowledge distillation
- **Energy Baseline**: A classical rule-based VAD using signal energy and hysteresis thresholding

The project demonstrates loading a trained checkpoint, running inference, comparing predictions, tuning baseline parameters, and generating visual comparisons.

## Project Goal

Compare TinyVAD student model predictions against energy-based VAD baseline outputs on TORGO dataset audio, analyzing differences in prediction behavior, confidence levels, and temporal alignment.

## Repository Purpose

This repository serves as an individual milestone project demonstrating:
- Model inference and evaluation workflows
- Comparison methodology for VAD systems
- Parameter sensitivity analysis
- Visualization of model behaviors

**Note**: This milestone focuses on comparison and analysis rather than model training from scratch.

## Personal Contributions

The following scripts represent original personal work for this milestone:

### `scripts/personal/run_my_experiment.py`
Loads a trained TinyVAD checkpoint, runs inference on audio files, and compares frame-level predictions against energy baseline outputs. Handles frame rate mismatch through linear interpolation and computes MSE and correlation metrics.

### `scripts/personal/tune_energy_vad.py`
Performs a systematic sweep of 90 energy VAD parameter configurations (threshold, hysteresis, smoothing) to analyze which parameters affect frame-level probability outputs versus segment detection.

### `scripts/personal/plot_model_vs_energy.py`
Generates qualitative comparison plots showing student model predictions, energy baseline predictions, and audio waveforms for selected utterances with best, worst, and representative agreement.

## Starter / Reused Infrastructure

This repository builds upon existing infrastructure:

- **Model Architecture**: `models/tinyvad_student.py` - Existing TinyVAD implementation
- **Training Pipeline**: `train_loso.py` - Existing LOSO training script
- **Baseline Implementation**: `baselines/energy_vad.py` - Existing energy-based VAD
- **Baseline Runner**: `scripts/core/run_baseline.py` - Existing baseline execution
- **Dataset**: TORGO dataset (not included, paths configured separately)

The personal contribution is the **comparison, tuning, and visualization workflow** built on top of this infrastructure.

## Repository Structure

```
├── scripts/personal/              # Personal milestone scripts
│   ├── run_my_experiment.py       # Model comparison workflow
│   ├── tune_energy_vad.py         # Parameter tuning analysis
│   └── plot_model_vs_energy.py    # Visualization generation
│
├── outputs/personal/              # Personal milestone outputs
│   ├── comparison/                # Model vs baseline comparison
│   │   ├── summary.txt            # MSE and correlation metrics
│   │   ├── summary.json           # Detailed per-utterance results
│   │   └── student_frame_probs/   # Student model predictions (.npy)
│   ├── energy_tuning/             # Parameter sweep results
│   │   ├── summary.txt            # Best settings found
│   │   ├── results.csv            # All 90 settings evaluated
│   │   └── results.json           # Detailed results
│   └── qualitative_plots/         # Visual comparisons
│       ├── F01_Session1_0006.png  # Best agreement example
│       ├── F01_Session1_0009.png  # Worst agreement example
│       ├── F01_Session1_0003.png  # Representative example
│       └── summary.txt            # Plot descriptions
│
├── outputs/quick_test/            # Training artifacts
│   ├── checkpoints/               # Model checkpoint
│   │   └── fold_F01_latest_best.pt
│   └── logs/                      # Training metrics
│       └── fold_F01.csv
│
├── scripts/core/                  # Existing baseline scripts
├── baselines/                     # Existing baseline implementations
├── models/                        # Existing model architectures
├── configs/                       # Training configurations
└── README.md                      # This file
```

## Key Outputs

### Model Comparison (`outputs/personal/comparison/`)
- Frame-level prediction comparison for 20 utterances
- MSE and correlation metrics between methods
- Model size verification (472 KB within 500 KB target)

### Parameter Tuning (`outputs/personal/energy_tuning/`)
- 90 parameter configurations evaluated
- Key finding: smoothing_window affects frame probabilities; threshold/hysteresis affect segment detection only

### Visual Comparisons (`outputs/personal/qualitative_plots/`)
- 3 comparison plots showing predictions vs audio
- Examples selected by agreement quality (best, worst, representative)

### Training Artifacts (`outputs/quick_test/`)
- Trained checkpoint (1 epoch, validation AUC 0.9997)
- Training log with per-epoch metrics

## How to Reproduce the Main Personal Workflow

### Prerequisites
```bash
pip install -r requirements.txt  # PyTorch, NumPy, matplotlib, etc.
```

### Quick Reproduction (~2 seconds)
Run the comparison on 5 utterances:

```bash
python scripts/personal/run_my_experiment.py --max-utterances 5
```

**Expected output**:
- Model loaded: 120,933 parameters
- Model size: 472.7 KB
- Comparison summary saved to `outputs/personal/comparison/`

### Full Reproduction

1. **Run comparison on full pilot set**:
```bash
python scripts/personal/run_my_experiment.py --max-utterances 50
```

2. **Run parameter tuning** (~2 minutes):
```bash
python scripts/personal/tune_energy_vad.py --max-utterances 20
```

3. **Generate plots** (~5 seconds):
```bash
python scripts/personal/plot_model_vs_energy.py
```

## Known Limitations

1. **Proxy-Based Comparison**: The comparison uses the student model as a reference rather than ground-truth labels. This is appropriate for relative analysis but means results are proxy-based, not definitive ground-truth evaluation.

2. **Frame Rate Alignment**: The student model outputs ~31 fps while the baseline outputs ~100 fps. Linear interpolation is used to align frames, which preserves temporal relationships but may smooth fine-grained details.

3. **Checkpoint Reuse**: The project uses an existing trained checkpoint rather than training from scratch. The contribution is the comparison and analysis workflow, not the training pipeline.

4. **Subset Evaluation**: Analysis is performed on a subset of TORGO speaker F01. Broader evaluation would strengthen conclusions.

5. **No Superiority Claim**: The project does not claim the student model is definitively better than the baseline—it demonstrates different characteristics and behaviors.

## Milestone Scope

This repository is organized for an **individual project milestone** rather than a final polished research release. The focus is on demonstrating:

- Understanding of VAD model inference and evaluation
- Ability to compare different VAD approaches
- Systematic parameter analysis
- Clear documentation and presentation

## Technical Details

| Aspect | Details |
|--------|---------|
| Model Size | 472 KB (target: < 500 KB) |
| Parameters | 120,933 |
| Student Frame Rate | ~31 fps (CNN downsampling) |
| Baseline Frame Rate | ~100 fps (10ms hop) |
| Tuning Settings | 90 configurations evaluated |
| Best Correlation | 0.297 |
| Worst Correlation | -0.422 |

## Citation / Attribution

This milestone builds upon:
- TORGO dataset for dysarthric speech
- TinyVAD model architecture (existing implementation)
- Silero VAD as teacher model (for original training)