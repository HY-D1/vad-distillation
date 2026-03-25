#!/usr/bin/env python3
"""
Personal VAD Milestone: Qualitative Comparison Plots

This script generates visual plots comparing student-model predictions
against energy-baseline predictions for selected utterances.

Features:
- Selects 3 utterances: best agreement, worst agreement, representative
- Aligns frame rates using linear interpolation
- Plots frame probabilities vs time for both methods
- Includes audio waveform for context
- Saves high-quality PNG figures

Alignment Method:
- Student model: ~31 fps (CNN downsampling)
- Energy baseline: ~100 fps (10ms hop)
- Alignment: Linear interpolation to common time grid

Usage:
    python milestones/personal_vad/scripts/plot_model_vs_energy.py \
        --student-preds outputs/personal_vad/comparison/student_frame_probs \
        --baseline-preds outputs/personal_vad/baseline_energy_50/frame_probs \
        --summary outputs/personal_vad/comparison/summary.json \
        --manifest manifests/torgo_pilot.csv \
        --output-dir outputs/personal_vad/qualitative_plots
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try importing matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Plots will not be generated.")

from audio_loader import load_audio_mono


def load_summary(summary_path: str) -> Dict:
    """Load summary JSON from comparison outputs."""
    print(f"Loading summary: {summary_path}")
    with open(summary_path, 'r') as f:
        return json.load(f)


def select_examples(summary: Dict) -> List[Dict]:
    """
    Select 3 representative utterances for plotting.
    
    Returns list of dicts with utt_id, correlation, mse, and reason for selection.
    """
    comparisons = summary['comparison']['utterance_comparisons']
    
    # Sort by correlation
    sorted_by_corr = sorted(comparisons, key=lambda x: x['correlation'], reverse=True)
    
    examples = []
    
    # Best agreement (highest correlation)
    best = sorted_by_corr[0]
    examples.append({
        'utt_id': best['utt_id'],
        'correlation': best['correlation'],
        'mse': best['mse'],
        'reason': 'best_agreement',
        'description': 'Highest correlation with student model'
    })
    
    # Worst agreement (lowest correlation)
    worst = sorted_by_corr[-1]
    examples.append({
        'utt_id': worst['utt_id'],
        'correlation': worst['correlation'],
        'mse': worst['mse'],
        'reason': 'worst_agreement',
        'description': 'Lowest correlation (most disagreement)'
    })
    
    # Representative (middle correlation)
    mid_idx = len(sorted_by_corr) // 2
    mid = sorted_by_corr[mid_idx]
    examples.append({
        'utt_id': mid['utt_id'],
        'correlation': mid['correlation'],
        'mse': mid['mse'],
        'reason': 'representative',
        'description': 'Median correlation (typical case)'
    })
    
    return examples


def load_predictions(utt_id: str, student_dir: Path, baseline_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load student and baseline predictions for an utterance."""
    student_file = student_dir / f"{utt_id}.npy"
    baseline_file = baseline_dir / f"{utt_id}.npy"
    
    if not student_file.exists():
        raise FileNotFoundError(f"Student prediction not found: {student_file}")
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline prediction not found: {baseline_file}")
    
    student_probs = np.load(student_file)
    baseline_probs = np.load(baseline_file)
    
    return student_probs, baseline_probs


def load_audio(utt_id: str, manifest_path: str) -> Tuple[np.ndarray, int]:
    """Load audio waveform for an utterance."""
    # Find utterance in manifest
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            check_id = f"{row['speaker_id']}_{row['session']}_{int(row['utt_id']):04d}"
            if check_id == utt_id:
                audio_path = row['path']
                try:
                    waveform, sr = load_audio_mono(audio_path, target_sr=16000)
                    return waveform, sr
                except Exception as e:
                    print(f"Warning: Could not load audio for {utt_id}: {e}")
                    return None, 16000
    
    return None, 16000


def align_predictions(
    student_probs: np.ndarray,
    baseline_probs: np.ndarray,
    audio_duration: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align student and baseline predictions to common time grid.
    
    Args:
        student_probs: Student model frame probabilities
        baseline_probs: Energy baseline frame probabilities
        audio_duration: Audio duration in seconds
    
    Returns:
        time_grid: Common time axis
        student_aligned: Student probabilities interpolated to time_grid
        baseline_aligned: Baseline probabilities interpolated to time_grid
    """
    # Create time axes for each prediction
    student_time = np.linspace(0, audio_duration, len(student_probs))
    baseline_time = np.linspace(0, audio_duration, len(baseline_probs))
    
    # Create common time grid (use higher resolution = baseline resolution)
    time_grid = baseline_time
    
    # Interpolate student to baseline time grid
    student_aligned = np.interp(time_grid, student_time, student_probs)
    baseline_aligned = baseline_probs  # Already on target grid
    
    return time_grid, student_aligned, baseline_aligned


def plot_comparison(
    example: Dict,
    time_grid: np.ndarray,
    student_probs: np.ndarray,
    baseline_probs: np.ndarray,
    audio: np.ndarray,
    sr: int,
    output_path: Path
):
    """Generate comparison plot for one utterance."""
    
    if not MATPLOTLIB_AVAILABLE:
        print(f"Skipping plot for {example['utt_id']} (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 2])
    
    # Plot 1: Audio waveform (if available)
    ax1 = axes[0]
    if audio is not None:
        audio_time = np.linspace(0, len(audio) / sr, len(audio))
        ax1.plot(audio_time, audio, color='gray', linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Audio Waveform - {example["utt_id"]}')
    else:
        ax1.text(0.5, 0.5, 'Audio not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Audio - {example["utt_id"]} (not available)')
    
    ax1.set_xlim(0, time_grid[-1])
    ax1.set_xlabel('Time (seconds)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frame probabilities comparison
    ax2 = axes[1]
    ax2.plot(time_grid, student_probs, label='TinyVAD (Student)', color='blue', linewidth=2)
    ax2.plot(time_grid, baseline_probs, label='Energy Baseline', color='red', linewidth=2, alpha=0.7)
    
    # Add threshold line
    ax2.axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Threshold (0.5)', alpha=0.5)
    
    # Fill areas where each method predicts speech
    ax2.fill_between(time_grid, 0, 1, where=(student_probs > 0.5), 
                     alpha=0.1, color='blue', label='Student speech')
    ax2.fill_between(time_grid, 0, 1, where=(baseline_probs > 0.5), 
                     alpha=0.1, color='red', label='Baseline speech')
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Speech Probability', fontsize=12)
    ax2.set_title(f'VAD Predictions - {example["utt_id"]}\n'
                  f'Correlation: {example["correlation"]:.3f} | MSE: {example["mse"]:.4f} | '
                  f'Selected: {example["reason"].replace("_", " ").title()}',
                  fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(0, time_grid[-1])
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_path}")


def generate_summary(examples: List[Dict], output_dir: Path):
    """Generate summary file describing selected examples."""
    
    summary = {
        'examples': examples,
        'selection_method': 'Based on correlation with student model (TASK-003 comparison)',
        'alignment_method': 'Linear interpolation to baseline time grid',
        'frame_rates': {
            'student': '~31 fps (CNN downsampling)',
            'baseline': '~100 fps (10ms hop)'
        }
    }
    
    # Save JSON
    json_path = output_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text
    text_path = output_dir / 'summary.txt'
    with open(text_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("QUALITATIVE COMPARISON PLOTS - SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SELECTION METHOD\n")
        f.write("-" * 70 + "\n")
        f.write("Examples selected based on correlation with student model\n")
        f.write("from TASK-003 comparison results.\n\n")
        
        f.write("ALIGNMENT METHOD\n")
        f.write("-" * 70 + "\n")
        f.write("Student model: ~31 fps (due to CNN 4x downsampling)\n")
        f.write("Energy baseline: ~100 fps (10ms hop)\n")
        f.write("Alignment: Linear interpolation to baseline time grid\n\n")
        
        f.write("EXAMPLES SELECTED\n")
        f.write("-" * 70 + "\n\n")
        
        for i, ex in enumerate(examples, 1):
            f.write(f"{i}. {ex['utt_id']}\n")
            f.write(f"   Reason: {ex['reason'].replace('_', ' ').title()}\n")
            f.write(f"   Description: {ex['description']}\n")
            f.write(f"   Correlation: {ex['correlation']:.3f}\n")
            f.write(f"   MSE: {ex['mse']:.4f}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("PLOT FILES\n")
        f.write("=" * 70 + "\n\n")
        
        for i, ex in enumerate(examples, 1):
            f.write(f"{i}. {ex['utt_id']}.png\n")
            f.write(f"   Shows: Audio waveform + VAD predictions comparison\n\n")
    
    print(f"\nSummary saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {text_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative comparison plots for VAD predictions"
    )
    parser.add_argument(
        '--student-preds',
        type=str,
        default='outputs/personal_vad/comparison/student_frame_probs',
        help='Directory with student model predictions (.npy)'
    )
    parser.add_argument(
        '--baseline-preds',
        type=str,
        default='outputs/personal_vad/baseline_energy_50/frame_probs',
        help='Directory with baseline predictions (.npy)'
    )
    parser.add_argument(
        '--summary',
        type=str,
        default='outputs/personal_vad/comparison/summary.json',
        help='Comparison summary JSON'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='manifests/torgo_pilot.csv',
        help='Manifest CSV for audio file paths'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/personal_vad/qualitative_plots',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required for plotting.")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    student_dir = Path(args.student_preds)
    baseline_dir = Path(args.baseline_preds)
    
    print("=" * 70)
    print("QUALITATIVE COMPARISON PLOTS")
    print("=" * 70)
    print(f"\nStudent predictions: {student_dir}")
    print(f"Baseline predictions: {baseline_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Load summary and select examples
    summary = load_summary(args.summary)
    examples = select_examples(summary)
    
    print("Selected examples:")
    for ex in examples:
        print(f"  - {ex['utt_id']}: {ex['reason']} (corr={ex['correlation']:.3f})")
    print()
    
    # Generate plots for each example
    for i, example in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] Processing {example['utt_id']}...")
        
        try:
            # Load predictions
            student_probs, baseline_probs = load_predictions(
                example['utt_id'], student_dir, baseline_dir
            )
            print(f"  Student frames: {len(student_probs)}, Baseline frames: {len(baseline_probs)}")
            
            # Estimate audio duration from baseline (100 fps = 10ms per frame)
            audio_duration = len(baseline_probs) * 0.01  # 10ms per frame
            
            # Align predictions
            time_grid, student_aligned, baseline_aligned = align_predictions(
                student_probs, baseline_probs, audio_duration
            )
            print(f"  Aligned to {len(time_grid)} frames")
            
            # Load audio (optional)
            audio, sr = load_audio(example['utt_id'], args.manifest)
            if audio is not None:
                print(f"  Audio loaded: {len(audio)/sr:.2f}s")
            
            # Generate plot
            output_path = output_dir / f"{example['utt_id']}.png"
            plot_comparison(
                example, time_grid, student_aligned, baseline_aligned,
                audio, sr, output_path
            )
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Generate summary
    generate_summary(examples, output_dir)
    
    print("\n" + "=" * 70)
    print("Plotting complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
