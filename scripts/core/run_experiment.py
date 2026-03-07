#!/usr/bin/env python3
"""
Unified experiment runner for VAD distillation.

Supports single experiments, batch experiments from JSON matrix, and resume functionality.

Usage:
    # Single experiment
    python scripts/run_experiment.py \
        --config configs/pilot.yaml \
        --fold F01 \
        --output_dir outputs/exp_001/

    # Batch experiments from matrix
    python scripts/run_experiment.py \
        --matrix configs/week2_matrix.json \
        --parallel 2 \
        --output_dir outputs/week2/

    # Resume interrupted experiments
    python scripts/run_experiment.py \
        --matrix configs/week2_matrix.json \
        --resume \
        --output_dir outputs/week2/

    # Dry run (show what would be executed)
    python scripts/run_experiment.py \
        --matrix configs/week2_matrix.json \
        --dry-run \
        --output_dir outputs/week2/
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from utils.common import load_config, save_config


# =============================================================================
# Configuration Loading and Manipulation
# =============================================================================

def override_config(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override specific keys in config with new values.
    
    Supports nested keys using dot notation (e.g., 'training.learning_rate').
    """
    config = copy.deepcopy(config)
    
    for key, value in overrides.items():
        keys = key.split('.')
        target = config
        
        # Navigate to the nested dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        # Set the final value
        target[keys[-1]] = value
    
    return config


def load_experiment_matrix(matrix_path: str) -> Dict[str, Any]:
    """Load experiment matrix from JSON file."""
    with open(matrix_path, 'r') as f:
        return json.load(f)


# =============================================================================
# Experiment Status Tracking
# =============================================================================

def get_experiment_dir(base_output_dir: str, exp_name: str, fold: str) -> str:
    """Get the output directory for a specific experiment."""
    # Only append fold if not already in experiment name
    if exp_name.endswith(f"_{fold}"):
        return os.path.join(base_output_dir, exp_name)
    return os.path.join(base_output_dir, f"{exp_name}_{fold}")


def check_experiment_status(exp_dir: str) -> Tuple[str, Optional[Dict]]:
    """
    Check the status of an experiment.
    
    Returns:
        Tuple of (status, metrics)
        status: 'completed', 'failed', 'incomplete', 'not_started'
        metrics: Dict with metrics if completed, None otherwise
    """
    # Check for completion marker
    completed_file = os.path.join(exp_dir, '.completed')
    failed_file = os.path.join(exp_dir, '.failed')
    metrics_file = os.path.join(exp_dir, 'metrics.json')
    
    if os.path.exists(completed_file):
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                return 'completed', json.load(f)
        return 'completed', None
    
    if os.path.exists(failed_file):
        return 'failed', None
    
    # Check if partially started (has log or checkpoint dir)
    log_file = os.path.join(exp_dir, 'log.txt')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    
    if os.path.exists(log_file) or os.path.exists(checkpoint_dir):
        return 'incomplete', None
    
    return 'not_started', None


def check_all_experiments(output_dir: str, experiments: List[Dict]) -> Dict[str, List[Dict]]:
    """Check status of all experiments in the matrix."""
    status = {
        'completed': [],
        'failed': [],
        'incomplete': [],
        'not_started': []
    }
    
    for exp in experiments:
        exp_name = exp['name']
        fold = exp['fold']
        exp_dir = get_experiment_dir(output_dir, exp_name, fold)
        exp_status, _ = check_experiment_status(exp_dir)
        status[exp_status].append(exp)
    
    return status


def load_progress(output_dir: str) -> Optional[Dict]:
    """Load progress tracking file."""
    progress_file = os.path.join(output_dir, 'progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None


def save_progress(output_dir: str, progress: Dict):
    """Save progress tracking file."""
    progress_file = os.path.join(output_dir, 'progress.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


# =============================================================================
# Single Experiment Execution
# =============================================================================

def run_single_experiment(
    exp_config: Dict[str, Any],
    output_dir: str,
    dry_run: bool = False
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Run a single experiment.
    
    Args:
        exp_config: Experiment configuration dictionary
        output_dir: Base output directory
        dry_run: If True, only show what would be run
    
    Returns:
        Tuple of (success, error_message, metrics)
    """
    exp_name = exp_config['name']
    fold = exp_config['fold']
    config_path = exp_config['config']
    overrides = exp_config.get('overrides', {})
    
    # Create experiment-specific output directory
    exp_dir = get_experiment_dir(output_dir, exp_name, fold)
    
    if dry_run:
        print(f"  [DRY-RUN] Would run: {exp_name} (fold {fold})")
        print(f"    Config: {config_path}")
        print(f"    Overrides: {overrides}")
        print(f"    Output: {exp_dir}")
        return True, None, None
    
    print(f"\n  Running: {exp_name} (fold {fold})")
    print(f"  Output directory: {exp_dir}")
    
    # Create output directories
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load base config and apply overrides
    try:
        base_config = load_config(config_path)
        effective_config = override_config(base_config, overrides)
        
        # Override output directories in config
        effective_config['output'] = {
            'checkpoint_dir': checkpoint_dir,
            'log_dir': exp_dir
        }
        
        # Save effective config
        effective_config_path = os.path.join(exp_dir, 'config.yaml')
        save_config(effective_config, effective_config_path)
        
    except Exception as e:
        error_msg = f"Failed to load/override config: {str(e)}"
        print(f"    ERROR: {error_msg}")
        return False, error_msg, None
    
    # Prepare command
    # Get project root for subprocess cwd
    project_root = Path(__file__).parent.parent.resolve()
    
    cmd = [
        sys.executable,
        'train_loso.py',
        '--config', effective_config_path,
        '--fold', fold
    ]
    
    # Check for resume checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, f'fold_{fold}_latest.pt')
    if os.path.exists(latest_checkpoint):
        cmd.extend(['--resume', latest_checkpoint])
        print(f"  Resuming from checkpoint: {latest_checkpoint}")
    
    # Run experiment
    log_file = os.path.join(exp_dir, 'log.txt')
    start_time = time.time()
    
    try:
        with open(log_file, 'w') as log_f:
            log_f.write(f"Experiment: {exp_name}\n")
            log_f.write(f"Fold: {fold}\n")
            log_f.write(f"Config: {config_path}\n")
            log_f.write(f"Overrides: {json.dumps(overrides, indent=2)}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"Started: {datetime.now().isoformat()}\n")
            log_f.write("=" * 60 + "\n\n")
            log_f.flush()
            
            # Run subprocess from project root
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to log file and console
            for line in process.stdout:
                log_f.write(line)
                log_f.flush()
                print(f"    {line.rstrip()}")
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        
        # Check for summary file to extract metrics
        summary_file = os.path.join(exp_dir, f'fold_{fold}_summary.json')
        metrics = None
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                metrics = {
                    'val_auc': summary.get('best_val_auc', 0.0),
                    'test_auc': summary.get('test_metrics', {}).get('auc', 0.0),
                    'test_miss_rate': summary.get('test_metrics', {}).get('miss_rate', 0.0),
                    'test_far': summary.get('test_metrics', {}).get('false_alarm_rate', 0.0),
                    'model_size_kb': summary.get('model_size_mb', 0.0) * 1024,
                    'num_parameters': summary.get('num_parameters', 0)
                }
        
        # Mark as completed
        with open(os.path.join(exp_dir, '.completed'), 'w') as f:
            f.write(datetime.now().isoformat())
        
        # Save metrics
        training_time = time.time() - start_time
        metrics_data = {
            'experiment_name': exp_name,
            'fold': fold,
            'status': 'completed',
            'training_time_seconds': training_time,
            **(metrics or {})
        }
        
        with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"  ✓ Completed in {training_time:.1f}s")
        return True, None, metrics_data
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Training failed with return code {e.returncode}"
        print(f"    ERROR: {error_msg}")
        
        # Mark as failed
        with open(os.path.join(exp_dir, '.failed'), 'w') as f:
            f.write(f"Failed at: {datetime.now().isoformat()}\n")
            f.write(f"Error: {error_msg}\n")
        
        # Save error metrics
        metrics_data = {
            'experiment_name': exp_name,
            'fold': fold,
            'status': 'failed',
            'error': error_msg
        }
        with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        return False, error_msg, None
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"    ERROR: {error_msg}")
        
        # Write error to log
        with open(log_file, 'a') as log_f:
            log_f.write(f"\n\nERROR: {error_msg}\n")
            log_f.write(tb)
        
        # Mark as failed
        with open(os.path.join(exp_dir, '.failed'), 'w') as f:
            f.write(f"Failed at: {datetime.now().isoformat()}\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"Traceback:\n{tb}\n")
        
        return False, error_msg, None


# =============================================================================
# Batch Experiment Execution
# =============================================================================

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def run_batch_experiments(
    matrix: Dict[str, Any],
    output_dir: str,
    parallel: int = 1,
    resume: bool = False,
    dry_run: bool = False
) -> Tuple[int, int, List[Dict]]:
    """
    Run multiple experiments from a matrix.
    
    Args:
        matrix: Experiment matrix dictionary
        output_dir: Base output directory
        parallel: Number of parallel jobs
        resume: Whether to resume incomplete experiments
        dry_run: If True, only show what would be run
    
    Returns:
        Tuple of (num_succeeded, num_failed, results)
    """
    experiments = matrix.get('experiments', [])
    total = len(experiments)
    
    if total == 0:
        print("No experiments to run!")
        return 0, 0, []
    
    print(f"\n{'='*60}")
    print(f"Batch Experiment Runner")
    print(f"{'='*60}")
    print(f"Total experiments: {total}")
    print(f"Parallel jobs: {parallel}")
    print(f"Output directory: {output_dir}")
    print(f"Resume mode: {resume}")
    if dry_run:
        print("DRY RUN - No actual execution")
    print(f"{'='*60}\n")
    
    # Check status of all experiments
    if resume:
        status = check_all_experiments(output_dir, experiments)
        completed_count = len(status['completed'])
        
        # Filter to only incomplete/not started experiments
        to_run = status['incomplete'] + status['not_started']
        
        print(f"Resuming: {completed_count}/{total} already completed")
        print(f"  - {len(status['completed'])} completed")
        print(f"  - {len(status['incomplete'])} incomplete (will retry)")
        print(f"  - {len(status['not_started'])} not started")
        print(f"  - {len(status['failed'])} failed (will retry)")
        
        # Add failed experiments to retry list
        to_run.extend(status['failed'])
    else:
        to_run = experiments
    
    if dry_run:
        print(f"\n[DRY-RUN] Would run {len(to_run)} experiments:\n")
        for exp in to_run:
            run_single_experiment(exp, output_dir, dry_run=True)
        return 0, 0, []
    
    # Run experiments
    succeeded = 0
    failed = 0
    results = []
    start_time = time.time()
    
    if parallel == 1:
        # Sequential execution
        for i, exp in enumerate(to_run, 1):
            elapsed = time.time() - start_time
            avg_time = elapsed / (succeeded + failed) if (succeeded + failed) > 0 else 0
            remaining = len(to_run) - i + 1
            eta = avg_time * remaining if avg_time > 0 else 0
            
            print(f"\n[{i}/{len(to_run)}] ETA: {format_time(eta)}")
            
            success, error, metrics = run_single_experiment(exp, output_dir)
            
            if success:
                succeeded += 1
                if metrics:
                    results.append(metrics)
            else:
                failed += 1
            
            # Update progress
            progress = {
                'total_experiments': total,
                'completed': completed_count + succeeded if resume else succeeded,
                'failed': failed,
                'last_update': datetime.now().isoformat()
            }
            save_progress(output_dir, progress)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            # Submit all tasks
            future_to_exp = {
                executor.submit(run_single_experiment_wrapper, exp, output_dir): exp
                for exp in to_run
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_exp):
                completed += 1
                exp = future_to_exp[future]
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = len(to_run) - completed
                eta = avg_time * remaining
                
                print(f"\n[{completed}/{len(to_run)}] ETA: {format_time(eta)}")
                
                try:
                    success, error, metrics = future.result()
                    if success:
                        succeeded += 1
                        if metrics:
                            results.append(metrics)
                        print(f"  ✓ {exp['name']} completed")
                    else:
                        failed += 1
                        print(f"  ✗ {exp['name']} failed: {error}")
                except Exception as e:
                    failed += 1
                    print(f"  ✗ {exp['name']} crashed: {str(e)}")
                
                # Update progress
                progress = {
                    'total_experiments': total,
                    'completed': completed_count + succeeded if resume else succeeded,
                    'failed': failed,
                    'last_update': datetime.now().isoformat()
                }
                save_progress(output_dir, progress)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Batch Complete")
    print(f"{'='*60}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Total time: {format_time(total_time)}")
    print(f"{'='*60}\n")
    
    return succeeded, failed, results


def run_single_experiment_wrapper(exp: Dict[str, Any], output_dir: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Wrapper for running single experiment in subprocess (must be pickleable)."""
    return run_single_experiment(exp, output_dir, dry_run=False)


# =============================================================================
# Reporting
# =============================================================================

def generate_report(output_dir: str) -> str:
    """
    Generate summary CSV report from all experiment results.
    
    Returns:
        Path to generated CSV file
    """
    print(f"\nGenerating report from {output_dir}...")
    
    # Find all experiment directories
    exp_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            metrics_file = os.path.join(item_path, 'metrics.json')
            config_file = os.path.join(item_path, 'config.yaml')
            if os.path.exists(metrics_file):
                exp_dirs.append((item, item_path))
    
    if not exp_dirs:
        print("No experiment results found!")
        return None
    
    # Collect data
    rows = []
    for exp_name, exp_dir in sorted(exp_dirs):
        metrics_file = os.path.join(exp_dir, 'metrics.json')
        config_file = os.path.join(exp_dir, 'config.yaml')
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Load config to get hyperparameters
        alpha = None
        temperature = None
        if os.path.exists(config_file):
            try:
                config = load_config(config_file)
                alpha = config.get('alpha')
                temperature = config.get('temperature')
            except Exception:
                pass
        
        row = {
            'experiment_name': metrics.get('experiment_name', exp_name),
            'fold': metrics.get('fold', ''),
            'alpha': alpha,
            'temperature': temperature,
            'val_auc': metrics.get('val_auc', ''),
            'val_miss_rate': metrics.get('val_miss_rate', ''),
            'val_far': metrics.get('val_far', ''),
            'test_auc': metrics.get('test_auc', ''),
            'test_miss_rate': metrics.get('test_miss_rate', ''),
            'test_far': metrics.get('test_far', ''),
            'model_size_kb': metrics.get('model_size_kb', ''),
            'training_time_seconds': metrics.get('training_time_seconds', ''),
            'status': metrics.get('status', 'unknown')
        }
        rows.append(row)
    
    # Write CSV
    csv_path = os.path.join(output_dir, 'summary.csv')
    fieldnames = [
        'experiment_name', 'fold', 'alpha', 'temperature',
        'val_auc', 'val_miss_rate', 'val_far',
        'test_auc', 'test_miss_rate', 'test_far',
        'model_size_kb', 'training_time_seconds', 'status'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  Summary saved to: {csv_path}")
    print(f"  Total experiments: {len(rows)}")
    
    # Also generate simple HTML report
    html_path = os.path.join(output_dir, 'report.html')
    generate_html_report(rows, html_path)
    print(f"  HTML report saved to: {html_path}")
    
    return csv_path


def generate_html_report(rows: List[Dict], output_path: str):
    """Generate simple HTML report."""
    completed = [r for r in rows if r['status'] == 'completed']
    failed = [r for r in rows if r['status'] == 'failed']
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>VAD Distillation Experiment Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .success { color: green; }
        .failed { color: red; }
        .stats { background: #f9f9f9; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>VAD Distillation Experiment Report</h1>
    <p>Generated: {timestamp}</p>
    
    <div class="stats">
        <h2>Summary</h2>
        <p>Total experiments: {total}</p>
        <p class="success">Completed: {completed_count}</p>
        <p class="failed">Failed: {failed_count}</p>
    </div>
    
    <h2>Results</h2>
    <table>
        <tr>
            <th>Experiment</th>
            <th>Fold</th>
            <th>Alpha</th>
            <th>Temperature</th>
            <th>Val AUC</th>
            <th>Test AUC</th>
            <th>Test Miss Rate</th>
            <th>Test FAR</th>
            <th>Status</th>
        </tr>
        {rows_html}
    </table>
</body>
</html>"""
    
    rows_html = ""
    for row in rows:
        status_class = 'success' if row['status'] == 'completed' else 'failed'
        rows_html += f"""
        <tr>
            <td>{row['experiment_name']}</td>
            <td>{row['fold']}</td>
            <td>{row['alpha']}</td>
            <td>{row['temperature']}</td>
            <td>{row['val_auc']}</td>
            <td>{row['test_auc']}</td>
            <td>{row['test_miss_rate']}</td>
            <td>{row['test_far']}</td>
            <td class="{status_class}">{row['status']}</td>
        </tr>"""
    
    html = html.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        total=len(rows),
        completed_count=len(completed),
        failed_count=len(failed),
        rows_html=rows_html
    )
    
    with open(output_path, 'w') as f:
        f.write(html)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified experiment runner for VAD distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  python scripts/run_experiment.py --config configs/pilot.yaml --fold F01 --output_dir outputs/exp_001/

  # Batch from matrix
  python scripts/run_experiment.py --matrix configs/week2_matrix.json --parallel 2 --output_dir outputs/week2/

  # Resume interrupted experiments
  python scripts/run_experiment.py --matrix configs/week2_matrix.json --resume --output_dir outputs/week2/

  # Dry run
  python scripts/run_experiment.py --matrix configs/week2_matrix.json --dry-run --output_dir outputs/week2/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--config', type=str,
                             help='Single config file (for single run)')
    input_group.add_argument('--matrix', type=str,
                             help='JSON file with experiment matrix (for batch run)')
    
    # Other options
    parser.add_argument('--fold', type=str,
                        help='Which LOSO fold to train (required for single run)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume incomplete experiments from output_dir')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run without executing')
    parser.add_argument('--report-only', action='store_true',
                        help='Only generate report from existing results')
    
    args = parser.parse_args()
    
    # Report-only mode
    if args.report_only:
        generate_report(args.output_dir)
        return
    
    # Single experiment mode
    if args.config:
        if not args.fold:
            parser.error('--fold is required when using --config')
        
        exp_config = {
            'name': f"single_{Path(args.config).stem}_{args.fold}",
            'config': args.config,
            'fold': args.fold,
            'overrides': {}
        }
        
        success, error, metrics = run_single_experiment(
            exp_config, args.output_dir, dry_run=args.dry_run
        )
        
        if not args.dry_run:
            generate_report(args.output_dir)
        
        sys.exit(0 if success else 1)
    
    # Batch experiment mode
    if args.matrix:
        if not os.path.exists(args.matrix):
            print(f"ERROR: Matrix file not found: {args.matrix}")
            sys.exit(1)
        
        matrix = load_experiment_matrix(args.matrix)
        succeeded, failed, results = run_batch_experiments(
            matrix, args.output_dir,
            parallel=args.parallel,
            resume=args.resume,
            dry_run=args.dry_run
        )
        
        if not args.dry_run:
            generate_report(args.output_dir)
        
        # Exit with error code if any experiments failed
        if failed > 0:
            print(f"\nWARNING: {failed} experiment(s) failed!")
            sys.exit(1)


# =============================================================================
# Test Code
# =============================================================================

def test_experiment_runner():
    """Test the experiment runner with dummy data."""
    import tempfile
    import shutil
    
    print("="*60)
    print("Testing Experiment Runner")
    print("="*60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="vad_exp_test_")
    print(f"Test directory: {temp_dir}")
    
    try:
        # Create dummy config
        config = {
            'data': {
                'manifest': 'data/manifest.json',
                'teacher_probs_dir': 'data/teacher_probs',
                'n_mels': 40
            },
            'model': {
                'type': 'lstm',
                'params': {'input_dim': 40, 'hidden_dim': 64}
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 4,
                'learning_rate': 0.001
            },
            'alpha': 0.5,
            'temperature': 3.0
        }
        
        config_path = os.path.join(temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create dummy experiment matrix
        matrix = {
            'experiments': [
                {
                    'name': 'test_alpha_0.5_T_3.0',
                    'config': config_path,
                    'fold': 'F01',
                    'overrides': {'alpha': 0.5, 'temperature': 3.0}
                },
                {
                    'name': 'test_alpha_0.7_T_3.0',
                    'config': config_path,
                    'fold': 'F01',
                    'overrides': {'alpha': 0.7, 'temperature': 3.0}
                }
            ]
        }
        
        matrix_path = os.path.join(temp_dir, 'test_matrix.json')
        with open(matrix_path, 'w') as f:
            json.dump(matrix, f, indent=2)
        
        print("\n--- Test 1: Dry-run mode ---")
        output_dir = os.path.join(temp_dir, 'outputs')
        
        # Simulate dry-run
        for exp in matrix['experiments']:
            run_single_experiment(exp, output_dir, dry_run=True)
        
        print("\n--- Test 2: Check status functions ---")
        status = check_all_experiments(output_dir, matrix['experiments'])
        for key, exps in status.items():
            print(f"  {key}: {len(exps)}")
        
        print("\n--- Test 3: Config override ---")
        base_config = load_config(config_path)
        print(f"  Original alpha: {base_config['alpha']}")
        
        overridden = override_config(base_config, {'alpha': 0.9, 'temperature': 5.0})
        print(f"  Overridden alpha: {overridden['alpha']}")
        print(f"  Overridden temperature: {overridden['temperature']}")
        
        print("\n--- Test 4: Nested override ---")
        overridden = override_config(base_config, {'training.learning_rate': 0.01})
        print(f"  Overridden learning_rate: {overridden['training']['learning_rate']}")
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up: {temp_dir}")


if __name__ == '__main__':
    # Check if running in test mode
    if '--test' in sys.argv:
        sys.argv.remove('--test')
        test_experiment_runner()
    else:
        main()
