"""
Status command for VAD distillation.

Shows training status overview for all folds.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from cli.config import get_all_folds
from cli.utils import (
    format_duration,
    print_error,
    print_info,
    print_success,
    print_warning,
    with_project_root
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add status-specific arguments."""
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/',
        help='Output directory to check'
    )
    parser.add_argument(
        '--fold',
        type=str,
        default=None,
        help='Show status for specific fold only'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Watch mode (continuous updates)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )


@with_project_root
def main(args: argparse.Namespace) -> int:
    """
    Execute status command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    if args.watch:
        return watch_mode(args.output_dir, args.fold)
    
    status = collect_status(args.output_dir, args.fold)
    
    if args.json:
        import json as json_module
        print(json_module.dumps(status, indent=2))
    else:
        print_status_table(status)
    
    return 0


def collect_status(output_dir: str, fold_filter: Optional[str] = None) -> Dict:
    """
    Collect status information.
    
    Args:
        output_dir: Base output directory
        fold_filter: Optional fold ID to filter
        
    Returns:
        Status dictionary
    """
    output_path = Path(output_dir)
    
    # Get all folds
    if fold_filter:
        folds = [fold_filter]
    else:
        folds = get_all_folds()
    
    status = {
        'output_dir': str(output_path),
        'timestamp': time.time(),
        'folds': {}
    }
    
    for fold in folds:
        fold_status = get_fold_status(output_path, fold)
        status['folds'][fold] = fold_status
    
    # Calculate summary
    completed = sum(1 for f in status['folds'].values() if f['status'] == 'completed')
    in_progress = sum(1 for f in status['folds'].values() if f['status'] == 'in_progress')
    failed = sum(1 for f in status['folds'].values() if f['status'] == 'failed')
    not_started = sum(1 for f in status['folds'].values() if f['status'] == 'not_started')
    
    status['summary'] = {
        'total': len(folds),
        'completed': completed,
        'in_progress': in_progress,
        'failed': failed,
        'not_started': not_started
    }
    
    return status


def get_fold_status(output_path: Path, fold: str) -> Dict:
    """Get status for a single fold."""
    status = {
        'fold': fold,
        'status': 'not_started',
        'metrics': {},
        'last_update': None
    }
    
    # Check for summary file
    summary_path = output_path / 'logs' / f'fold_{fold}_summary.json'
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            status['status'] = 'completed'
            status['metrics'] = summary.get('test_metrics', {})
            status['best_val_auc'] = summary.get('best_val_auc', 0.0)
            status['model_size_kb'] = summary.get('model_size_mb', 0) * 1024
            
            # Get file modification time
            mtime = summary_path.stat().st_mtime
            status['last_update'] = mtime
            
        except Exception:
            status['status'] = 'failed'
    
    # Check for in-progress indicators
    checkpoint_path = output_path / 'checkpoints' / f'fold_{fold}_latest.pt'
    if checkpoint_path.exists() and status['status'] == 'not_started':
        status['status'] = 'in_progress'
        status['last_update'] = checkpoint_path.stat().st_mtime
    
    return status


def print_status_table(status: Dict) -> None:
    """Print status as formatted table."""
    print("="*80)
    print("TRAINING STATUS")
    print("="*80)
    print(f"Output directory: {status['output_dir']}")
    print()
    
    # Summary
    summary = status['summary']
    print(f"Total: {summary['total']} | ", end='')
    print_success(f"Completed: {summary['completed']}")
    print(f" | In Progress: {summary['in_progress']} | ", end='')
    if summary['failed'] > 0:
        print_error(f"Failed: {summary['failed']}")
    else:
        print(f"Failed: {summary['failed']}", end='')
    print(f" | Not Started: {summary['not_started']}")
    print()
    
    # Per-fold status
    print(f"{'Fold':<8} {'Status':<12} {'Val AUC':<10} {'Test AUC':<10} {'Last Update':<20}")
    print("-"*80)
    
    for fold, fold_status in sorted(status['folds'].items()):
        status_str = fold_status['status']
        
        # Format status with emoji
        if status_str == 'completed':
            status_display = '✓ Complete'
        elif status_str == 'in_progress':
            status_display = '⋯ Running'
        elif status_str == 'failed':
            status_display = '✗ Failed'
        else:
            status_display = '○ Not Started'
        
        # Format metrics
        val_auc = fold_status.get('best_val_auc', 0.0)
        test_metrics = fold_status.get('metrics', {})
        test_auc = test_metrics.get('auc', 0.0)
        
        val_str = f"{val_auc:.4f}" if val_auc > 0 else '-'
        test_str = f"{test_auc:.4f}" if test_auc > 0 else '-'
        
        # Format last update
        last_update = fold_status.get('last_update')
        if last_update:
            time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(last_update))
        else:
            time_str = '-'
        
        print(f"{fold:<8} {status_display:<12} {val_str:<10} {test_str:<10} {time_str:<20}")
    
    print("="*80)


def watch_mode(output_dir: str, fold: Optional[str]) -> int:
    """
    Watch mode with continuous updates.
    
    Args:
        output_dir: Output directory to watch
        fold: Optional specific fold to watch
        
    Returns:
        Exit code
    """
    import time as time_module
    
    try:
        while True:
            # Clear screen (cross-platform)
            print('\033[2J\033[H', end='')
            
            status = collect_status(output_dir, fold)
            print_status_table(status)
            
            print(f"\nLast updated: {time_module.strftime('%Y-%m-%d %H:%M:%S')}")
            print("Press Ctrl+C to exit")
            
            time_module.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nExiting watch mode.")
        return 0
