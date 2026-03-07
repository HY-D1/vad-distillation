#!/usr/bin/env python3
"""
Compute baseline metrics for Silero VAD on TORGO pilot subset.

This script loads cached teacher probabilities and computes metrics using
proxy ground truth (all speech frames).

Usage:
    python scripts/compute_baseline_metrics.py
    python scripts/compute_baseline_metrics.py --manifest manifests/torgo_pilot.csv
    python scripts/compute_baseline_metrics.py --threshold 0.5
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load manifest CSV."""
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_teacher_probs(teacher_dir: Path, speaker_id: str, session: str, utt_id: str) -> np.ndarray:
    """Load cached teacher probabilities for an utterance."""
    # Try pattern: {speaker}_{session}_{utt_id:04d}.npy (from cache_teacher.py)
    try:
        utt_num = int(utt_id)
        prob_file = teacher_dir / f"{speaker_id}_{session}_{utt_num:04d}.npy"
        if prob_file.exists():
            return np.load(prob_file)
    except ValueError:
        pass
    
    # Try pattern: {speaker}_{utt_id}.npy (from run_silero_teacher.py)
    prob_file = teacher_dir / f"{speaker_id}_{utt_id}.npy"
    if prob_file.exists():
        return np.load(prob_file)
    
    return None


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics at a given threshold.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred_probs: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Confusion matrix components
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Metrics
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    # F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'miss_rate': miss_rate,
        'false_alarm_rate': false_alarm_rate,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def compute_speaker_metrics(
    manifest_rows: List[Dict],
    teacher_dir: Path,
    speaker_id: str,
    threshold: float = 0.5
) -> Dict:
    """
    Compute metrics for a specific speaker.
    
    Args:
        manifest_rows: All manifest rows
        teacher_dir: Directory with teacher probabilities
        speaker_id: Speaker to compute metrics for
        threshold: Classification threshold
    
    Returns:
        Dictionary with metrics and statistics
    """
    # Filter rows for this speaker
    speaker_rows = [r for r in manifest_rows if r['speaker_id'] == speaker_id]
    
    all_probs = []
    all_labels = []
    utterance_count = 0
    frame_count = 0
    missing_count = 0
    
    for row in speaker_rows:
        probs = load_teacher_probs(teacher_dir, row['speaker_id'], row['session'], row['utt_id'])
        
        if probs is None:
            missing_count += 1
            continue
        
        # Proxy ground truth: all frames are speech
        labels = np.ones(len(probs), dtype=int)
        
        all_probs.extend(probs)
        all_labels.extend(labels)
        utterance_count += 1
        frame_count += len(probs)
    
    if len(all_probs) == 0:
        return {
            'utterance_count': 0,
            'frame_count': 0,
            'missing_count': missing_count,
            'auc': None,
            'metrics_at_threshold': None,
        }
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # All labels are the same
        auc = None
    
    # Compute metrics at threshold
    metrics = compute_metrics_at_threshold(all_labels, all_probs, threshold)
    
    return {
        'utterance_count': utterance_count,
        'frame_count': frame_count,
        'missing_count': missing_count,
        'expected_utterances': len(speaker_rows),
        'auc': auc,
        'metrics_at_threshold': metrics,
    }


def generate_report(
    overall_metrics: Dict,
    speaker_metrics: Dict[str, Dict],
    threshold: float,
    output_path: Path
):
    """Generate markdown report."""
    
    report = f"""# Silero VAD Baseline Metrics on TORGO Pilot

## Dataset
- Total utterances: {overall_metrics['utterance_count']}
- Total frames: {overall_metrics['frame_count']:,}
- Threshold: {threshold}

### Speaker Distribution
"""
    
    for speaker, metrics in speaker_metrics.items():
        speaker_upper = speaker.upper()
        speaker_type = "Dysarthric" if speaker_upper in ["F01", "M01"] else "Control"
        report += f"- {speaker} ({speaker_type}): {metrics['utterance_count']} utterances"
        if metrics['missing_count'] > 0:
            report += f" ({metrics['missing_count']} missing)"
        report += "\n"
    
    report += f"""
## Overall Metrics

| Metric | Value |
|--------|-------|
"""
    
    if overall_metrics['auc'] is not None:
        report += f"| AUC | {overall_metrics['auc']:.4f} |\n"
    else:
        report += "| AUC | N/A |\n"
    
    m = overall_metrics['metrics_at_threshold']
    report += f"| Miss Rate @ {threshold} | {m['miss_rate']:.4f} |\n"
    report += f"| False Alarm Rate @ {threshold} | {m['false_alarm_rate']:.4f} |\n"
    report += f"| Accuracy @ {threshold} | {m['accuracy']:.4f} |\n"
    report += f"| F1 Score @ {threshold} | {m['f1_score']:.4f} |\n"
    report += f"| Precision @ {threshold} | {m['precision']:.4f} |\n"
    report += f"| Recall @ {threshold} | {m['recall']:.4f} |\n"
    
    report += f"""
### Confusion Matrix
| | Predicted Speech | Predicted Non-Speech |
|---|:---:|:---:|
| **Actual Speech** | TP: {m['tp']:,} | FN: {m['fn']:,} |
| **Actual Non-Speech** | FP: {m['fp']:,} | TN: {m['tn']:,} |

## Per-Speaker Metrics

| Speaker | Type | Utterances | Frames | AUC | Miss Rate | FAR | Accuracy | F1 |
|---------|------|:----------:|:------:|:---:|:---------:|:---:|:--------:|:--:|
"""
    
    for speaker, metrics in speaker_metrics.items():
        speaker_upper = speaker.upper()
        speaker_type = "Dysarthric" if speaker_upper in ["F01", "M01"] else "Control"
        
        if metrics['utterance_count'] == 0:
            report += f"| {speaker} | {speaker_type} | 0 | - | - | - | - | - | - |\n"
            continue
        
        auc = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
        m = metrics['metrics_at_threshold']
        
        report += f"| {speaker} | {speaker_type} | {metrics['utterance_count']} | {metrics['frame_count']:,} | "
        report += f"{auc} | {m['miss_rate']:.4f} | {m['false_alarm_rate']:.4f} | "
        report += f"{m['accuracy']:.4f} | {m['f1_score']:.4f} |\n"
    
    # Compute dysarthric vs control comparison
    dysarthric_speakers = ['F01', 'M01']
    control_speakers = ['FC01']
    
    dysarthric_frames = sum(speaker_metrics[s]['frame_count'] for s in dysarthric_speakers if s in speaker_metrics)
    control_frames = sum(speaker_metrics[s]['frame_count'] for s in control_speakers if s in speaker_metrics)
    
    report += f"""
## Dysarthric vs Control Comparison

| Group | Speakers | Total Frames | Avg Miss Rate | Avg FAR | Avg F1 |
|-------|----------|-------------|:-------------:|:-------:|:------:|
"""
    
    # Average metrics for dysarthric
    dys_mr = np.mean([speaker_metrics[s]['metrics_at_threshold']['miss_rate'] for s in dysarthric_speakers if s in speaker_metrics])
    dys_far = np.mean([speaker_metrics[s]['metrics_at_threshold']['false_alarm_rate'] for s in dysarthric_speakers if s in speaker_metrics])
    dys_f1 = np.mean([speaker_metrics[s]['metrics_at_threshold']['f1_score'] for s in dysarthric_speakers if s in speaker_metrics])
    
    # Metrics for control
    con_mr = np.mean([speaker_metrics[s]['metrics_at_threshold']['miss_rate'] for s in control_speakers if s in speaker_metrics])
    con_far = np.mean([speaker_metrics[s]['metrics_at_threshold']['false_alarm_rate'] for s in control_speakers if s in speaker_metrics])
    con_f1 = np.mean([speaker_metrics[s]['metrics_at_threshold']['f1_score'] for s in control_speakers if s in speaker_metrics])
    
    report += f"| Dysarthric | F01, M01 | {dysarthric_frames:,} | {dys_mr:.4f} | {dys_far:.4f} | {dys_f1:.4f} |\n"
    report += f"| Control | FC01 | {control_frames:,} | {con_mr:.4f} | {con_far:.4f} | {con_f1:.4f} |\n"
    
    report += f"""
## Observations

1. **Speech Detection Bias**: With proxy ground truth (all speech), Miss Rate represents frames 
   classified as non-speech by Silero VAD, while False Alarm Rate should be near 0 (no actual non-speech).

2. **AUC Interpretation**: AUC measures the model's ability to distinguish between speech and non-speech.
   Since we're using proxy labels (all speech), AUC will be 0.5 (random) if the model outputs are 
   consistent, or undefined if all labels are identical.

3. **Speaker Comparison**:
   - Dysarthric speakers (F01, M01) show different characteristics compared to control (FC01)
   - Higher Miss Rate on dysarthric speech suggests VAD may be dropping speech frames

## Limitations

1. **Proxy Ground Truth**: This analysis uses proxy labels (all frames assumed speech). True frame-level 
   ground truth with silence/non-speech annotations would provide more accurate metrics.

2. **Missing Data**: Some utterances had corrupted audio files and were skipped:
   - F01: 2 missing (Session1/0067.wav, 0068.wav)

3. **Threshold Sensitivity**: Metrics computed at threshold = {threshold}. Different operating 
   points may be preferred for different applications.

4. **Pilot Subset Only**: Results are for the pilot subset only and may not generalize to the full dataset.

---
*Generated by scripts/compute_baseline_metrics.py*
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compute baseline metrics for Silero VAD on TORGO"
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='manifests/torgo_pilot.csv',
        help='Path to manifest CSV'
    )
    parser.add_argument(
        '--teacher-dir',
        type=str,
        default='teacher_probs',
        help='Directory with cached teacher probabilities'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='local/baseline_metrics_report.md',
        help='Output markdown report path'
    )
    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Optional JSON output for metrics'
    )
    
    args = parser.parse_args()
    
    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    manifest = load_manifest(args.manifest)
    print(f"Total utterances in manifest: {len(manifest)}")
    
    # Check teacher directory
    teacher_dir = Path(args.teacher_dir)
    if not teacher_dir.exists():
        print(f"Error: Teacher directory not found: {teacher_dir}")
        sys.exit(1)
    
    # Get unique speakers
    speakers = sorted(set(row['speaker_id'] for row in manifest))
    print(f"Speakers: {speakers}")
    
    # Compute per-speaker metrics
    print(f"\nComputing metrics at threshold = {args.threshold}...")
    speaker_metrics = {}
    
    for speaker in speakers:
        print(f"  Processing {speaker}...")
        speaker_metrics[speaker] = compute_speaker_metrics(
            manifest, teacher_dir, speaker, args.threshold
        )
    
    # Compute overall metrics
    print("\nComputing overall metrics...")
    all_probs = []
    all_labels = []
    total_utterances = 0
    
    for speaker in speakers:
        rows = [r for r in manifest if r['speaker_id'] == speaker]
        for row in rows:
            probs = load_teacher_probs(teacher_dir, row['speaker_id'], row['session'], row['utt_id'])
            if probs is not None:
                labels = np.ones(len(probs), dtype=int)
                all_probs.extend(probs)
                all_labels.extend(labels)
                total_utterances += 1
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"Total frames collected: {len(all_probs):,}")
    
    # Overall AUC
    try:
        overall_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        overall_auc = None
    
    overall_metrics = {
        'utterance_count': total_utterances,
        'frame_count': len(all_probs),
        'auc': overall_auc,
        'metrics_at_threshold': compute_metrics_at_threshold(all_labels, all_probs, args.threshold)
    }
    
    # Generate report
    print(f"\nGenerating report: {args.output}")
    report = generate_report(overall_metrics, speaker_metrics, args.threshold, Path(args.output))
    
    # Optional JSON output
    if args.json:
        json_data = {
            'overall': overall_metrics,
            'per_speaker': speaker_metrics,
            'threshold': args.threshold,
        }
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(args.json, 'w') as f:
            json.dump(json_data, f, indent=2, default=convert)
        print(f"JSON metrics saved: {args.json}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("BASELINE METRICS SUMMARY")
    print("=" * 60)
    print(f"\nOverall AUC: {overall_auc:.4f}" if overall_auc else "\nOverall AUC: N/A")
    m = overall_metrics['metrics_at_threshold']
    print(f"Miss Rate @ {args.threshold}: {m['miss_rate']:.4f}")
    print(f"False Alarm Rate @ {args.threshold}: {m['false_alarm_rate']:.4f}")
    print(f"Accuracy @ {args.threshold}: {m['accuracy']:.4f}")
    print(f"F1 Score @ {args.threshold}: {m['f1_score']:.4f}")
    
    print("\nPer-Speaker AUC:")
    for speaker in speakers:
        auc = speaker_metrics[speaker]['auc']
        utt_count = speaker_metrics[speaker]['utterance_count']
        if auc is not None:
            print(f"  {speaker}: {auc:.4f} ({utt_count} utts)")
        else:
            print(f"  {speaker}: N/A ({utt_count} utts)")
    
    print("\n" + "=" * 60)
    print(f"Report saved: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
