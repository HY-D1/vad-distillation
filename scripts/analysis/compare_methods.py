#!/usr/bin/env python3
"""
Unified comparison script for VAD methods.

Compares multiple VAD methods against ground truth or proxy labels, generating
comparison tables and plots.

Usage:
    # Compare methods against teacher (proxy labels)
    python scripts/compare_methods.py \
        --manifest manifests/torgo_pilot.csv \
        --methods outputs/exp1/frame_probs,outputs/exp2/frame_probs \
        --method-names "Baseline,Our Method" \
        --output-dir outputs/comparison \
        --proxy-labels teacher

    # Compare with all speech proxy (sanity check)
    python scripts/compare_methods.py \
        --manifest manifests/torgo_pilot.csv \
        --methods outputs/exp1/frame_probs,baselines/energy/frame_probs \
        --method-names "Our Model,Energy" \
        --output-dir outputs/comparison \
        --proxy-labels all_speech

    # Compare with true labels (rarely available)
    python scripts/compare_methods.py \
        --manifest manifests/torgo_pilot.csv \
        --methods outputs/exp1/frame_probs,outputs/exp2/frame_probs \
        --method-names "Model A,Model B" \
        --output-dir outputs/comparison \
        --proxy-labels none
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare multiple VAD methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare against teacher labels
  python scripts/compare_methods.py \\
      --manifest manifests/torgo_pilot.csv \\
      --methods outputs/exp1/frame_probs,outputs/exp2/frame_probs \\
      --method-names "Baseline,Ours" \\
      --output-dir outputs/comparison \\
      --proxy-labels teacher

  # All speech proxy (sanity check)
  python scripts/compare_methods.py \\
      --manifest manifests/torgo_pilot.csv \\
      --methods outputs/exp1/frame_probs,baselines/energy/frame_probs \\
      --method-names "Our Model,Energy" \\
      --output-dir outputs/comparison \\
      --proxy-labels all_speech
        """
    )
    
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to manifest CSV (ground truth/proxy labels)'
    )
    parser.add_argument(
        '--methods',
        type=str,
        required=True,
        help='Comma-separated list of method output directories'
    )
    parser.add_argument(
        '--method-names',
        type=str,
        default=None,
        help='Comma-separated names for legend/table (default: use directory names)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Where to save comparison results'
    )
    parser.add_argument(
        '--proxy-labels',
        type=str,
        choices=['teacher', 'hard', 'all_speech', 'none'],
        default='teacher',
        help='Label source: teacher (thresholded teacher_probs), hard (frame-level hard labels), '
             'all_speech (all 1s), none (use manifest labels when available)'
    )
    parser.add_argument(
        '--teacher-dir',
        type=str,
        default='teacher_probs',
        help='Directory with teacher probabilities (for teacher proxy)'
    )
    parser.add_argument(
        '--hard-label-dir',
        type=str,
        default='teacher_hard_labels/thresh_0.5',
        help='Directory with frame-level hard labels (.npy) when --proxy-labels hard'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for binary metrics'
    )
    parser.add_argument(
        '--timing-file',
        type=str,
        default=None,
        help='Optional JSON file with timing data per method (method_name -> ms_per_frame)'
    )
    parser.add_argument(
        '--model-size-file',
        type=str,
        default=None,
        help='Optional JSON file with model sizes (method_name -> size_in_kb)'
    )
    
    return parser.parse_args()


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load manifest CSV file."""
    rows = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_file_id(row: Dict) -> str:
    """
    Generate file ID from manifest row.
    
    Tries multiple naming patterns:
    - {speaker}_{session}_{utt_id:04d}
    - {speaker}_{utt_id}
    """
    speaker = row['speaker_id']
    session = row.get('session', '')
    utt_id = row['utt_id']
    
    # Try numbered format first
    try:
        utt_num = int(utt_id)
        if session:
            return f"{speaker}_{session}_{utt_num:04d}"
        return f"{speaker}_{utt_num}"
    except ValueError:
        pass
    
    # Fallback to string ID
    if session:
        return f"{speaker}_{session}_{utt_id}"
    return f"{speaker}_{utt_id}"


def load_frame_probs(frame_probs_dir: Path, file_id: str) -> Optional[np.ndarray]:
    """
    Load frame probabilities for a specific utterance.
    
    Args:
        frame_probs_dir: Directory containing .npy files
        file_id: File identifier from manifest
    
    Returns:
        Array of frame probabilities or None if not found
    """
    # Try exact match first
    prob_file = frame_probs_dir / f"{file_id}.npy"
    if prob_file.exists():
        return np.load(prob_file)
    
    # Try without padding
    parts = file_id.split('_')
    if len(parts) >= 3:
        try:
            utt_num = int(parts[-1])
            short_id = '_'.join(parts[:-1] + [str(utt_num)])
            prob_file = frame_probs_dir / f"{short_id}.npy"
            if prob_file.exists():
                return np.load(prob_file)
        except ValueError:
            pass
    
    return None


def load_ground_truth_labels(
    manifest_rows: List[Dict],
    proxy_type: str,
    teacher_dir: Path
) -> Dict[str, np.ndarray]:
    """
    Load ground truth labels for all utterances.
    
    Args:
        manifest_rows: List of manifest rows
        proxy_type: 'teacher', 'all_speech', or 'none'
        teacher_dir: Directory with teacher probabilities
    
    Returns:
        Dictionary mapping file_id to label array
    """
    labels_dict = {}
    
    for row in manifest_rows:
        file_id = get_file_id(row)
        
        if proxy_type == 'teacher':
            # Load teacher probabilities and threshold at 0.5
            teacher_probs = load_frame_probs(teacher_dir, file_id)
            if teacher_probs is not None:
                labels_dict[file_id] = (teacher_probs >= 0.5).astype(int)
        elif proxy_type == 'all_speech':
            # Assume all frames are speech (will be set when we know the length)
            labels_dict[file_id] = None  # Placeholder
        elif proxy_type == 'none':
            # Expect true labels in manifest (rarely available)
            # Look for label file or annotation
            labels_dict[file_id] = None  # Not implemented
    
    return labels_dict


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Binary predictions
        y_probs: Predicted probabilities
    
    Returns:
        Dictionary with metrics
    """
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_probs = y_probs.flatten()
    
    # Filter out invalid labels (-1)
    valid_mask = y_true >= 0
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    y_probs = y_probs[valid_mask]
    
    if len(y_true) == 0:
        return {
            'auc': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'miss_rate': 0.0,
            'false_alarm_rate': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5  # All labels are the same
    
    # F1 Score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        # Handle case where only one class is present
        tn = fp = fn = tp = 0
        if len(y_true) > 0:
            if y_true[0] == 0:
                tn = np.sum(y_pred == 0)
                fp = np.sum(y_pred == 1)
            else:
                fn = np.sum(y_pred == 0)
                tp = np.sum(y_pred == 1)
    
    # Miss Rate (False Negative Rate) = FN / (FN + TP)
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # False Alarm Rate (False Positive Rate) = FP / (FP + TN)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'auc': auc,
        'f1': f1,
        'accuracy': accuracy,
        'miss_rate': miss_rate,
        'false_alarm_rate': false_alarm_rate,
        'precision': precision,
        'recall': recall,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def evaluate_method(
    method_dir: Path,
    manifest_rows: List[Dict],
    proxy_type: str,
    teacher_dir: Path,
    hard_label_dir: Optional[Path],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a single method against ground truth/proxy labels.
    
    Args:
        method_dir: Directory with frame_probs subdirectory
        manifest_rows: List of manifest rows
        proxy_type: Type of proxy labels
        teacher_dir: Directory with teacher probabilities
        hard_label_dir: Directory with frame-level hard labels
        threshold: Classification threshold
    
    Returns:
        Dictionary with metrics and statistics
    """
    frame_probs_dir = method_dir / 'frame_probs'
    if not frame_probs_dir.exists():
        frame_probs_dir = method_dir  # Try the directory itself
    
    all_probs = []
    all_labels = []
    all_predictions = []
    utterance_count = 0
    missing_count = 0
    
    for row in manifest_rows:
        file_id = get_file_id(row)
        
        # Load method's frame probabilities
        probs = load_frame_probs(frame_probs_dir, file_id)
        if probs is None:
            missing_count += 1
            continue
        
        # Load ground truth based on proxy type
        if proxy_type == 'teacher':
            teacher_probs = load_frame_probs(teacher_dir, file_id)
            if teacher_probs is None:
                missing_count += 1
                continue
            labels = (teacher_probs >= 0.5).astype(int)
        elif proxy_type == 'hard':
            if hard_label_dir is None:
                missing_count += 1
                continue
            hard_labels = load_frame_probs(hard_label_dir, file_id)
            if hard_labels is None:
                missing_count += 1
                continue
            labels = (hard_labels >= 0.5).astype(int)
        elif proxy_type == 'all_speech':
            # All frames are speech
            labels = np.ones(len(probs), dtype=int)
        elif proxy_type == 'none':
            # Would need true labels from manifest (not implemented)
            labels = np.ones(len(probs), dtype=int)  # Fallback
        
        # Ensure same length
        min_len = min(len(probs), len(labels))
        probs = probs[:min_len]
        labels = labels[:min_len]
        
        # Make predictions
        predictions = (probs >= threshold).astype(int)
        
        all_probs.extend(probs)
        all_labels.extend(labels)
        all_predictions.extend(predictions)
        utterance_count += 1
    
    if len(all_probs) == 0:
        return {
            'utterance_count': 0,
            'frame_count': 0,
            'missing_count': missing_count,
            'metrics': None
        }
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions, all_probs)
    
    return {
        'utterance_count': utterance_count,
        'frame_count': len(all_probs),
        'missing_count': missing_count,
        'metrics': metrics
    }


def format_size(size_kb) -> str:
    """Format size in KB to human-readable string.
    
    Handles special string values like "N/A" or "~1400".
    """
    # Handle string values (N/A, ~1400, etc.)
    if isinstance(size_kb, str):
        return size_kb
    
    # Handle numeric values
    if size_kb < 1024:
        return f"{size_kb:.1f} KB"
    elif size_kb < 1024 * 1024:
        return f"{size_kb / 1024:.2f} MB"
    else:
        return f"{size_kb / (1024 * 1024):.2f} GB"


def format_latency(ms_per_frame) -> str:
    """Format latency in ms/frame to human-readable string.
    
    Always shows in ms for consistency in comparison tables.
    Handles special string values like "N/A".
    """
    # Handle string values (N/A, etc.)
    if isinstance(ms_per_frame, str):
        return ms_per_frame
    
    # Always show in ms for consistency
    return f"{ms_per_frame:.3f} ms"


def load_timing_data(timing_file: Optional[str]) -> Dict[str, float]:
    """Load timing data from JSON file.
    
    Supports both flat format: {"Method": 0.5}
    And nested format: {"Method": {"latency_ms": 0.5, ...}}
    """
    if timing_file is None or not Path(timing_file).exists():
        return {}
    
    with open(timing_file, 'r') as f:
        data = json.load(f)
    
    # Convert nested format to flat format
    result = {}
    for method, value in data.items():
        if isinstance(value, dict) and 'latency_ms' in value:
            result[method] = value['latency_ms']
        elif isinstance(value, (int, float)):
            result[method] = value
    return result


def load_model_sizes(size_file: Optional[str]) -> Dict[str, Any]:
    """Load model size data from JSON file.
    
    Supports both flat format: {"Method": 473}
    And nested format: {"Method": {"size_kb": 473, ...}}
    Returns dict with method -> size_kb (or special string like "N/A")
    """
    if size_file is None or not Path(size_file).exists():
        return {}
    
    with open(size_file, 'r') as f:
        data = json.load(f)
    
    # Convert nested format to flat format, handling special values
    result = {}
    for method, value in data.items():
        if isinstance(value, dict) and 'size_kb' in value:
            # Keep the value as-is (could be number, string like "N/A", "~1400")
            result[method] = value['size_kb']
        elif isinstance(value, (int, float)):
            result[method] = value
        elif isinstance(value, str):
            result[method] = value
    return result


def save_comparison_csv(
    results: List[Dict],
    output_path: Path
):
    """Save comparison results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'Method', 'Model_Size_KB', 'Latency_ms_per_frame',
        'AUC', 'F1', 'Accuracy', 'Miss_Rate', 'FAR',
        'Precision', 'Recall', 'Utterances', 'Frames'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            metrics = result['metrics'] or {}
            row = {
                'Method': result['name'],
                'Model_Size_KB': result.get('model_size_kb', ''),
                'Latency_ms_per_frame': result.get('latency_ms', ''),
                'AUC': f"{metrics.get('auc', 0):.4f}" if metrics else '',
                'F1': f"{metrics.get('f1', 0):.4f}" if metrics else '',
                'Accuracy': f"{metrics.get('accuracy', 0):.4f}" if metrics else '',
                'Miss_Rate': f"{metrics.get('miss_rate', 0):.4f}" if metrics else '',
                'FAR': f"{metrics.get('false_alarm_rate', 0):.4f}" if metrics else '',
                'Precision': f"{metrics.get('precision', 0):.4f}" if metrics else '',
                'Recall': f"{metrics.get('recall', 0):.4f}" if metrics else '',
                'Utterances': result.get('utterance_count', 0),
                'Frames': result.get('frame_count', 0)
            }
            writer.writerow(row)


def generate_markdown_table(results: List[Dict]) -> str:
    """Generate markdown comparison table."""
    lines = []
    lines.append("| Method | Size (KB) | Latency (ms/frame) | AUC | F1 | Miss Rate | FAR |")
    lines.append("|--------|-----------|--------------------|-----|-----|-----------|-----|")
    
    for result in results:
        name = result['name']
        size = format_size(result.get('model_size_kb', 0)) if result.get('model_size_kb') else 'N/A'
        latency = format_latency(result.get('latency_ms', 0)) if result.get('latency_ms') else 'N/A'
        
        metrics = result['metrics']
        if metrics:
            auc = f"{metrics['auc']:.4f}"
            f1 = f"{metrics['f1']:.4f}"
            miss_rate = f"{metrics['miss_rate']:.4f}"
            far = f"{metrics['false_alarm_rate']:.4f}"
        else:
            auc = f1 = miss_rate = far = 'N/A'
        
        lines.append(f"| {name} | {size} | {latency} | {auc} | {f1} | {miss_rate} | {far} |")
    
    return '\n'.join(lines)


def save_comparison_markdown(
    results: List[Dict],
    output_path: Path,
    args: argparse.Namespace
):
    """Save comparison results to markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("# VAD Methods Comparison")
    lines.append("")
    lines.append(f"**Manifest:** {args.manifest}")
    lines.append(f"**Proxy Labels:** {args.proxy_labels}")
    lines.append(f"**Threshold:** {args.threshold}")
    lines.append("")
    
    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(generate_markdown_table(results))
    lines.append("")
    
    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")
    
    for result in results:
        lines.append(f"### {result['name']}")
        lines.append("")
        lines.append(f"- **Utterances:** {result['utterance_count']}")
        lines.append(f"- **Frames:** {result['frame_count']:,}")
        if result.get('missing_count', 0) > 0:
            lines.append(f"- **Missing:** {result['missing_count']}")
        
        if result.get('model_size_kb'):
            lines.append(f"- **Model Size:** {format_size(result['model_size_kb'])}")
        if result.get('latency_ms'):
            lines.append(f"- **Latency:** {result['latency_ms']:.3f} ms/frame")
        
        metrics = result['metrics']
        if metrics:
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| AUC | {metrics['auc']:.4f} |")
            lines.append(f"| F1 Score | {metrics['f1']:.4f} |")
            lines.append(f"| Accuracy | {metrics['accuracy']:.4f} |")
            lines.append(f"| Miss Rate | {metrics['miss_rate']:.4f} |")
            lines.append(f"| False Alarm Rate | {metrics['false_alarm_rate']:.4f} |")
            lines.append(f"| Precision | {metrics['precision']:.4f} |")
            lines.append(f"| Recall | {metrics['recall']:.4f} |")
            lines.append("")
            lines.append("**Confusion Matrix:**")
            lines.append("")
            lines.append("| | Pred Speech | Pred Non-Speech |")
            lines.append("|---|:---:|:---:|")
            lines.append(f"| **Actual Speech** | TP: {metrics['tp']:,} | FN: {metrics['fn']:,} |")
            lines.append(f"| **Actual Non-Speech** | FP: {metrics['fp']:,} | TN: {metrics['tn']:,} |")
        
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def plot_auc_comparison(
    results: List[Dict],
    output_path: Path
):
    """Create bar chart comparing AUC scores."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping AUC plot")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    names = [r['name'] for r in results if r['metrics']]
    aucs = [r['metrics']['auc'] for r in results if r['metrics']]
    
    if not names:
        print("Warning: No valid results for AUC plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, aucs, color='steelblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('AUC Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_miss_rate_far(
    results: List[Dict],
    output_path: Path
):
    """Create scatter plot of Miss Rate vs FAR."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping Miss Rate vs FAR plot")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_results = [r for r in results if r['metrics']]
    
    if not valid_results:
        print("Warning: No valid results for Miss Rate vs FAR plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for result in valid_results:
        metrics = result['metrics']
        ax.scatter(
            metrics['false_alarm_rate'],
            metrics['miss_rate'],
            s=200,
            alpha=0.6,
            edgecolors='black',
            linewidth=1.5,
            label=result['name']
        )
        
        # Add method name as annotation
        ax.annotate(
            result['name'],
            (metrics['false_alarm_rate'], metrics['miss_rate']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )
    
    ax.set_xlabel('False Alarm Rate', fontsize=12)
    ax.set_ylabel('Miss Rate', fontsize=12)
    ax.set_title('Miss Rate vs False Alarm Rate', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("VAD Methods Comparison")
    print("=" * 60)
    print()
    
    # Parse method directories and names
    method_dirs = [Path(d.strip()) for d in args.methods.split(',')]
    if args.method_names:
        method_names = [n.strip() for n in args.method_names.split(',')]
        if len(method_names) != len(method_dirs):
            print(f"Error: Number of method names ({len(method_names)}) doesn't match "
                  f"number of methods ({len(method_dirs)})")
            sys.exit(1)
    else:
        method_names = [d.name for d in method_dirs]
    
    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    manifest_rows = load_manifest(args.manifest)
    print(f"  Total utterances: {len(manifest_rows)}")
    print()
    
    # Load optional timing and size data
    timing_data = load_timing_data(args.timing_file)
    size_data = load_model_sizes(args.model_size_file)
    
    # Evaluate each method
    results = []
    teacher_dir = Path(args.teacher_dir)
    hard_label_dir = Path(args.hard_label_dir)
    
    print(f"Evaluating {len(method_dirs)} methods:")
    print("-" * 60)
    
    for method_name, method_dir in zip(method_names, method_dirs):
        print(f"\n{method_name}:")
        print(f"  Directory: {method_dir}")
        
        if not method_dir.exists():
            print(f"  Warning: Directory not found, skipping")
            continue
        
        start_time = time.time()
        result = evaluate_method(
            method_dir,
            manifest_rows,
            args.proxy_labels,
            teacher_dir,
            hard_label_dir if args.proxy_labels == 'hard' else None,
            args.threshold
        )
        eval_time = time.time() - start_time
        
        result['name'] = method_name
        result['dir'] = str(method_dir)
        
        # Add timing data if available
        if method_name in timing_data:
            result['latency_ms'] = timing_data[method_name]
        
        # Add size data if available
        if method_name in size_data:
            result['model_size_kb'] = size_data[method_name]
        
        results.append(result)
        
        print(f"  Utterances: {result['utterance_count']}")
        print(f"  Frames: {result['frame_count']:,}")
        if result['missing_count'] > 0:
            print(f"  Missing: {result['missing_count']}")
        
        if result['metrics']:
            m = result['metrics']
            print(f"  AUC: {m['auc']:.4f}")
            print(f"  F1: {m['f1']:.4f}")
            print(f"  Miss Rate: {m['miss_rate']:.4f}")
            print(f"  FAR: {m['false_alarm_rate']:.4f}")
        
        print(f"  Evaluation time: {eval_time:.2f}s")
    
    print()
    print("=" * 60)
    print("Generating Outputs")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / 'comparison_table.csv'
    save_comparison_csv(results, csv_path)
    print(f"  Saved: {csv_path}")
    
    # Save Markdown
    md_path = output_dir / 'comparison_table.md'
    save_comparison_markdown(results, md_path, args)
    print(f"  Saved: {md_path}")
    
    # Create plots
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_auc_comparison(results, plots_dir / 'auc_comparison.png')
    plot_miss_rate_far(results, plots_dir / 'miss_rate_far.png')
    
    print()
    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print()
    print(generate_markdown_table(results))
    print()
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
