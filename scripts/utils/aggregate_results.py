#!/usr/bin/env python3
"""
Aggregate results from all folds for status and verify modes.

Usage:
    python scripts/utils/aggregate_results.py --output-dir outputs/production_cuda/
    python scripts/utils/aggregate_results.py --output-dir outputs/production_cuda/ --format json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# All 15 folds
ALL_FOLDS = [
    "F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05",
    "FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"
]


def load_fold_summary(output_dir: Path, fold_id: str) -> Optional[Dict]:
    """Load summary.json for a single fold."""
    summary_path = output_dir / "logs" / f"fold_{fold_id}_summary.json"
    
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_fold_predictions(output_dir: Path, fold_id: str) -> Optional[Dict]:
    """Load predictions.npz for a single fold."""
    predictions_path = output_dir / "logs" / f"fold_{fold_id}_predictions.npz"
    
    if not predictions_path.exists():
        return None
    
    try:
        data = np.load(predictions_path)
        return {
            'predictions': data['predictions'],
            'labels': data['labels'],
            'probs': data['probs']
        }
    except Exception:
        return None


def aggregate_metrics(output_dir: Path, folds: List[str] = None) -> Dict:
    """Aggregate metrics across all folds."""
    if folds is None:
        folds = ALL_FOLDS
    
    results = []
    for fold_id in folds:
        summary = load_fold_summary(output_dir, fold_id)
        if summary and 'test_metrics' in summary:
            results.append({
                'fold': fold_id,
                'metrics': summary['test_metrics'],
                'epochs_trained': summary.get('epochs_trained', 0),
                'training_time': summary.get('training_time_seconds', 0)
            })
    
    if not results:
        return {
            'folds_completed': 0,
            'total_folds': len(folds),
            'message': 'No results found'
        }
    
    # Extract metrics
    metrics_keys = ['auc', 'f1', 'accuracy', 'miss_rate', 'false_alarm_rate', 
                    'precision', 'recall']
    
    aggregated = {
        'folds_completed': len(results),
        'total_folds': len(folds),
        'folds': [r['fold'] for r in results],
        'metrics': {}
    }
    
    for key in metrics_keys:
        values = [r['metrics'].get(key, 0) for r in results if key in r['metrics']]
        if values:
            aggregated['metrics'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': {r['fold']: r['metrics'].get(key, 0) for r in results}
            }
    
    # Training stats
    training_times = [r['training_time'] for r in results]
    if training_times:
        aggregated['training_time'] = {
            'total_seconds': sum(training_times),
            'mean_seconds': float(np.mean(training_times)),
            'total_formatted': format_duration(sum(training_times))
        }
    
    epochs = [r['epochs_trained'] for r in results]
    if epochs:
        aggregated['epochs'] = {
            'mean': float(np.mean(epochs)),
            'min': int(np.min(epochs)),
            'max': int(np.max(epochs))
        }
    
    return aggregated


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_fold_status(output_dir: Path, fold_id: str) -> Dict:
    """Get detailed status for a single fold."""
    status = {
        'fold': fold_id,
        'status': 'pending',
        'summary_exists': False,
        'checkpoint_exists': False,
        'metrics': {}
    }
    
    # Check summary
    summary = load_fold_summary(output_dir, fold_id)
    if summary:
        status['summary_exists'] = True
        status['status'] = 'complete'
        status['metrics'] = summary.get('test_metrics', {})
        status['epochs_trained'] = summary.get('epochs_trained', 0)
        status['best_epoch'] = summary.get('best_epoch', 0)
        status['training_time'] = format_duration(
            summary.get('training_time_seconds', 0)
        )
    
    # Check checkpoint
    checkpoint_path = output_dir / "checkpoints" / f"fold_{fold_id}_best.pt"
    if checkpoint_path.exists():
        status['checkpoint_exists'] = True
        status['checkpoint_size_kb'] = checkpoint_path.stat().st_size // 1024
    
    return status


def generate_table(output_dir: Path, folds: List[str] = None) -> str:
    """Generate a text table of fold statuses."""
    if folds is None:
        folds = ALL_FOLDS
    
    lines = []
    lines.append("┌─────────┬──────────┬──────────┬──────────┬──────────┬─────────┐")
    lines.append("│ {:<7} │ {:<8} │ {:<8} │ {:<8} │ {:<8} │ {:<7} │".format(
        "Fold", "Status", "Test AUC", "Test F1", "MissRate", "Epochs"
    ))
    lines.append("├─────────┼──────────┼──────────┼──────────┼──────────┼─────────┤")
    
    for fold_id in folds:
        status = get_fold_status(output_dir, fold_id)
        
        if status['status'] == 'complete':
            metrics = status['metrics']
            lines.append("│ {:<7} │ {:<8} │ {:<8.4f} │ {:<8.4f} │ {:<8.4f} │ {:<7} │".format(
                fold_id,
                status['status'],
                metrics.get('auc', 0),
                metrics.get('f1', 0),
                metrics.get('miss_rate', 0),
                status.get('epochs_trained', '-')
            ))
        else:
            lines.append("│ {:<7} │ {:<8} │ {:<8} │ {:<8} │ {:<8} │ {:<7} │".format(
                fold_id, status['status'], '-', '-', '-', '-'
            ))
    
    lines.append("└─────────┴──────────┴──────────┴──────────┴──────────┴─────────┘")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from all folds"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['json', 'table', 'summary'],
        default='summary',
        help="Output format"
    )
    parser.add_argument(
        "--folds",
        type=str,
        nargs='+',
        default=None,
        help="Specific folds to aggregate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        return 1
    
    folds = args.folds if args.folds else ALL_FOLDS
    
    if args.format == 'json':
        result = aggregate_metrics(output_dir, folds)
        output = json.dumps(result, indent=2)
    elif args.format == 'table':
        output = generate_table(output_dir, folds)
    else:  # summary
        result = aggregate_metrics(output_dir, folds)
        lines = []
        lines.append("=" * 60)
        lines.append("Aggregated Results Summary")
        lines.append("=" * 60)
        lines.append(f"\nFolds completed: {result['folds_completed']}/{result['total_folds']}")
        
        if 'metrics' in result and result['metrics']:
            lines.append("\nMetrics:")
            for metric_name, stats in result['metrics'].items():
                lines.append(f"  {metric_name:15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                           f"[{stats['min']:.4f}, {stats['max']:.4f}]")
        
        if 'training_time' in result:
            lines.append(f"\nTotal training time: {result['training_time']['total_formatted']}")
        
        output = "\n".join(lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to: {args.output}")
    else:
        print(output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
