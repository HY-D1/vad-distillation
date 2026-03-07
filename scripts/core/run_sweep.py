#!/usr/bin/env python3
"""
Hyperparameter sweep runner for VAD distillation experiments.

Supports grid search and random search over alpha and temperature parameters.

Usage Examples:
    # Grid search over alpha and temperature
    python scripts/run_sweep.py \
        --param alpha --values 0.3 0.5 0.7 0.9 \
        --param temperature --values 1 2 3 5 \
        --folds F01 M01 FC01 \
        --base-config configs/pilot.yaml \
        --output-dir outputs/sweep_alpha_t \
        --parallel 2

    # Random search
    python scripts/run_sweep.py \
        --param alpha --range 0.0 1.0 \
        --param temperature --range 1 10 \
        --n-samples 20 \
        --folds F01 \
        --base-config configs/pilot.yaml \
        --output-dir outputs/sweep_random/

    # Single parameter sweep with fixed value
    python scripts/run_sweep.py \
        --param alpha --values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --fixed temperature=3.0 \
        --folds F01 \
        --base-config configs/pilot.yaml \
        --output-dir outputs/sweep_alpha/

    # Dry run
    python scripts/run_sweep.py \
        --param alpha --values 0.3 0.5 \
        --folds F01 \
        --base-config configs/pilot.yaml \
        --output-dir outputs/test_sweep \
        --dry-run
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Suppress warnings
warnings.filterwarnings('ignore')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter sweep for VAD distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Grid search:
    %(prog)s --param alpha --values 0.3 0.5 --param temperature --values 2 3 \\
             --folds F01 --base-config configs/pilot.yaml --output-dir outputs/sweep
  
  Random search:
    %(prog)s --param alpha --range 0.0 1.0 --param temperature --range 1 10 \\
             --n-samples 20 --folds F01 --base-config configs/pilot.yaml --output-dir outputs/sweep
  
  Single parameter sweep:
    %(prog)s --param alpha --values 0.1 0.5 0.9 --fixed temperature=3.0 \\
             --folds F01 --base-config configs/pilot.yaml --output-dir outputs/sweep
        """
    )
    
    # Parameter specification
    parser.add_argument(
        '--param', 
        action='append', 
        dest='params',
        help='Parameter name (can specify multiple times, e.g., --param alpha --param temperature)'
    )
    parser.add_argument(
        '--values', 
        action='append', 
        nargs='+', 
        type=float,
        dest='values_list',
        help='Values for grid search (provide once per --param)'
    )
    parser.add_argument(
        '--range', 
        action='append', 
        nargs=2, 
        type=float,
        dest='ranges',
        help='Min max for random search (provide once per --param)'
    )
    parser.add_argument(
        '--n-samples', 
        type=int,
        help='Number of random samples for random search'
    )
    parser.add_argument(
        '--fixed', 
        action='append',
        help='Fixed parameters as key=value pairs (e.g., --fixed temperature=3.0)'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--folds', 
        nargs='+', 
        required=True,
        help='Which folds to run (e.g., F01 M01 FC01)'
    )
    parser.add_argument(
        '--base-config', 
        required=True,
        help='Base config file path'
    )
    parser.add_argument(
        '--output-dir', 
        required=True,
        help='Output directory for sweep results'
    )
    
    # Execution control
    parser.add_argument(
        '--parallel', 
        type=int, 
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show experiments without running them'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Only run incomplete experiments'
    )
    parser.add_argument(
        '--max-experiments', 
        type=int,
        help='Limit total number of experiments to run'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=6140,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs to train (overrides config)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help='Early stopping patience (overrides config)'
    )
    
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if not args.params:
        raise ValueError("Must specify at least one --param")
    
    # Check if base config exists
    if not os.path.exists(args.base_config):
        raise FileNotFoundError(f"Base config not found: {args.base_config}")
    
    # Validate parameter specification
    if args.ranges is not None and args.n_samples is None:
        raise ValueError("Must specify --n-samples when using --range")
    
    if args.ranges is not None and args.values_list is not None:
        raise ValueError("Cannot specify both --values and --range for the same parameter")
    
    # Check that number of params matches values/ranges
    if args.values_list is not None and len(args.params) != len(args.values_list):
        raise ValueError(f"Number of --param ({len(args.params)}) must match number of --values ({len(args.values_list)})")
    
    if args.ranges is not None and len(args.params) != len(args.ranges):
        raise ValueError(f"Number of --param ({len(args.params)}) must match number of --range ({len(args.ranges)})")


def parse_fixed_params(fixed: Optional[List[str]]) -> Dict[str, float]:
    """Parse fixed parameters from key=value strings."""
    if fixed is None:
        return {}
    
    fixed_params = {}
    for item in fixed:
        if '=' not in item:
            raise ValueError(f"Fixed parameter must be in format key=value: {item}")
        key, value = item.split('=', 1)
        try:
            fixed_params[key] = float(value)
        except ValueError:
            fixed_params[key] = value
    return fixed_params


def generate_grid_search(
    params: List[str], 
    values_list: List[List[float]], 
    folds: List[str],
    fixed_params: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Generate all combinations for grid search.
    
    Args:
        params: List of parameter names
        values_list: List of value lists (one per parameter)
        folds: List of fold IDs
        fixed_params: Fixed parameters to add to each experiment
    
    Returns:
        List of experiment configurations
    """
    experiments = []
    
    # Create all parameter combinations
    param_combinations = list(itertools.product(*values_list))
    
    for fold in folds:
        for combo in param_combinations:
            exp = {'fold': fold}
            # Add swept parameters
            for param, value in zip(params, combo):
                exp[param] = value
            # Add fixed parameters
            exp.update(fixed_params)
            experiments.append(exp)
    
    return experiments


def generate_random_search(
    params: List[str],
    ranges: List[List[float]],
    n_samples: int,
    folds: List[str],
    fixed_params: Dict[str, float],
    seed: int = 6140
) -> List[Dict[str, Any]]:
    """
    Generate random samples for random search.
    
    Args:
        params: List of parameter names
        ranges: List of [min, max] ranges (one per parameter)
        n_samples: Number of random samples per fold
        folds: List of fold IDs
        fixed_params: Fixed parameters to add to each experiment
        seed: Random seed for reproducibility
    
    Returns:
        List of experiment configurations
    """
    random.seed(seed)
    np_random = random.Random(seed)
    
    experiments = []
    
    for fold in folds:
        for _ in range(n_samples):
            exp = {'fold': fold}
            # Add random parameters
            for param, (min_val, max_val) in zip(params, ranges):
                # Use uniform sampling
                value = min_val + np_random.random() * (max_val - min_val)
                exp[param] = value
            # Add fixed parameters
            exp.update(fixed_params)
            experiments.append(exp)
    
    return experiments


def experiment_to_dirname(exp: Dict[str, Any]) -> str:
    """Convert experiment config to directory name."""
    parts = []
    for key in sorted(exp.keys()):
        # Skip fold and internal keys (starting with _)
        if key != 'fold' and not key.startswith('_'):
            value = exp[key]
            # Format: param_value
            if isinstance(value, float):
                parts.append(f"{key}_{value:.2f}")
            else:
                parts.append(f"{key}_{value}")
    parts.append(f"fold_{exp['fold']}")
    return '_'.join(parts)


def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load base configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_experiment_config(
    base_config: Dict[str, Any],
    exp_params: Dict[str, Any],
    epochs: Optional[int] = None,
    patience: Optional[int] = None
) -> Dict[str, Any]:
    """Create experiment config by overriding base config with sweep parameters."""
    config = base_config.copy()
    
    # Override with sweep parameters
    for key, value in exp_params.items():
        if key != 'fold':
            config[key] = value
    
    # Override with command-line arguments if provided
    if epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['num_epochs'] = epochs
    
    if patience is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['early_stopping_patience'] = patience
    
    return config


def run_single_experiment(
    exp_params: Dict[str, Any],
    base_config: Dict[str, Any],
    output_dir: str,
    base_config_path: str,
    epochs: Optional[int] = None,
    patience: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a single experiment.
    
    Args:
        exp_params: Experiment parameters (alpha, temperature, fold)
        base_config: Base configuration dictionary
        output_dir: Root output directory for sweep
        base_config_path: Path to base config file
    
    Returns:
        Dictionary with experiment results
    """
    fold = exp_params['fold']
    exp_dirname = experiment_to_dirname(exp_params)
    exp_output_dir = os.path.join(output_dir, exp_dirname)
    
    # Create output directory
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Get epochs and patience from parent scope (passed via partial)
    sweep_epochs = exp_params.pop('_sweep_epochs', None)
    sweep_patience = exp_params.pop('_sweep_patience', None)
    
    # Create experiment config
    exp_config = create_experiment_config(base_config, exp_params, sweep_epochs, sweep_patience)
    
    # Set output_dir - use nested 'output' key if base_config has it, otherwise flat
    if 'output' in base_config:
        if 'output' not in exp_config:
            exp_config['output'] = {}
        exp_config['output']['checkpoint_dir'] = os.path.join(exp_output_dir, 'checkpoints/')
        exp_config['output']['log_dir'] = os.path.join(exp_output_dir, 'logs/')
    else:
        exp_config['output_dir'] = exp_output_dir
    
    # Save experiment config
    exp_config_path = os.path.join(exp_output_dir, 'config.yaml')
    with open(exp_config_path, 'w') as f:
        yaml.dump(exp_config, f)
    
    # Run training
    result = {
        'exp_params': exp_params,
        'output_dir': exp_output_dir,
        'success': False,
        'metrics': {},
        'error': None
    }
    
    try:
        # Use train_loso.py with our generated config
        # Change to project root for execution
        project_root = Path(__file__).parent.parent.resolve()
        cmd = [
            sys.executable, 'train_loso.py',
            '--config', exp_config_path,
            '--fold', fold
        ]
        
        # Run subprocess
        start_time = time.time()
        process = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        result['duration'] = time.time() - start_time
        
        # Check if successful
        if process.returncode == 0:
            result['success'] = True
            # Try to load metrics from summary
            summary_path = os.path.join(exp_output_dir, 'logs', f'fold_{fold}_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    result['metrics'] = summary.get('test_metrics', {})
                    result['best_val_auc'] = summary.get('best_val_auc', 0.0)
            else:
                result['warning'] = 'Summary file not found'
        else:
            result['error'] = process.stderr
            result['returncode'] = process.returncode
            
    except subprocess.TimeoutExpired:
        result['error'] = 'Experiment timed out after 1 hour'
    except Exception as e:
        result['error'] = str(e)
    
    return result


def is_experiment_complete(exp_params: Dict[str, Any], output_dir: str) -> bool:
    """Check if an experiment has already been completed."""
    fold = exp_params['fold']
    exp_dirname = experiment_to_dirname(exp_params)
    exp_output_dir = os.path.join(output_dir, exp_dirname)
    summary_path = os.path.join(exp_output_dir, 'logs', f'fold_{fold}_summary.json')
    return os.path.exists(summary_path)


def run_sweep(
    experiments: List[Dict[str, Any]],
    base_config: Dict[str, Any],
    base_config_path: str,
    output_dir: str,
    parallel: int = 1,
    resume: bool = False,
    max_experiments: Optional[int] = None,
    epochs: Optional[int] = None,
    patience: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Run all experiments in the sweep.
    
    Args:
        experiments: List of experiment configurations
        base_config: Base configuration
        base_config_path: Path to base config
        output_dir: Output directory
        parallel: Number of parallel jobs
        resume: Only run incomplete experiments
        max_experiments: Limit total number of experiments
    
    Returns:
        List of experiment results
    """
    # Filter completed experiments if resuming
    if resume:
        experiments = [exp for exp in experiments if not is_experiment_complete(exp, output_dir)]
        print(f"Resuming: {len(experiments)} experiments remaining")
    
    # Limit experiments if specified
    if max_experiments is not None:
        experiments = experiments[:max_experiments]
        print(f"Limited to {len(experiments)} experiments")
    
    print(f"\nRunning {len(experiments)} experiments with {parallel} parallel job(s)...")
    print("=" * 60)
    
    results = []
    
    # Inject epochs and patience into each experiment for parallel processing
    for exp in experiments:
        exp['_sweep_epochs'] = epochs
        exp['_sweep_patience'] = patience
    
    if parallel == 1:
        # Sequential execution
        for i, exp in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Running: alpha={exp.get('alpha', 'N/A'):.2f}, "
                  f"temperature={exp.get('temperature', 'N/A'):.2f}, fold={exp['fold']}")
            result = run_single_experiment(exp, base_config, output_dir, base_config_path, epochs, patience)
            results.append(result)
            
            if result['success']:
                metrics = result.get('metrics', {})
                print(f"  ✓ Complete - AUC: {metrics.get('auc', 0):.4f}, "
                      f"F1: {metrics.get('f1', 0):.4f}")
            else:
                print(f"  ✗ Failed - {result.get('error', 'Unknown error')[:100]}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            # Submit all jobs
            futures = {
                executor.submit(
                    run_single_experiment, 
                    exp, 
                    base_config, 
                    output_dir, 
                    base_config_path,
                    epochs,
                    patience
                ): exp for exp in experiments
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(futures), 1):
                exp = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    status = "✓" if result['success'] else "✗"
                    metrics = result.get('metrics', {})
                    print(f"[{i}/{len(experiments)}] {status} alpha={exp.get('alpha', 'N/A'):.2f}, "
                          f"T={exp.get('temperature', 'N/A'):.2f}, fold={exp['fold']} - "
                          f"AUC: {metrics.get('auc', 0):.4f}")
                except Exception as e:
                    print(f"[{i}/{len(experiments)}] ✗ alpha={exp.get('alpha', 'N/A'):.2f}, "
                          f"T={exp.get('temperature', 'N/A'):.2f}, fold={exp['fold']} - Error: {e}")
                    results.append({
                        'exp_params': exp,
                        'success': False,
                        'error': str(e)
                    })
    
    return results


def save_results_csv(results: List[Dict[str, Any]], output_dir: str) -> str:
    """Save results to CSV file."""
    import csv
    
    csv_path = os.path.join(output_dir, 'results.csv')
    
    # Collect all field names
    fieldnames = ['alpha', 'temperature', 'fold', 'success', 'duration', 'best_val_auc']
    metric_keys = set()
    for result in results:
        metric_keys.update(result.get('metrics', {}).keys())
    fieldnames.extend(sorted(metric_keys))
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {}
            exp_params = result.get('exp_params', {})
            row['alpha'] = exp_params.get('alpha', '')
            row['temperature'] = exp_params.get('temperature', '')
            row['fold'] = exp_params.get('fold', '')
            row['success'] = result.get('success', False)
            row['duration'] = result.get('duration', '')
            row['best_val_auc'] = result.get('best_val_auc', '')
            
            # Add metrics
            for key in metric_keys:
                row[key] = result.get('metrics', {}).get(key, '')
            
            writer.writerow(row)
    
    print(f"\nResults saved to: {csv_path}")
    return csv_path


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from results."""
    successful = [r for r in results if r.get('success', False)]
    
    if not successful:
        return {'error': 'No successful experiments'}
    
    # Find best configuration
    best_result = max(successful, key=lambda r: r.get('metrics', {}).get('auc', 0))
    
    summary = {
        'total_experiments': len(results),
        'successful_experiments': len(successful),
        'failed_experiments': len(results) - len(successful),
        'best_config': {
            'alpha': best_result['exp_params'].get('alpha'),
            'temperature': best_result['exp_params'].get('temperature'),
            'fold': best_result['exp_params'].get('fold'),
            'metrics': best_result.get('metrics', {})
        }
    }
    
    # Compute average metrics across folds for each config
    from collections import defaultdict
    config_metrics = defaultdict(list)
    
    for result in successful:
        params = result['exp_params']
        config_key = (params.get('alpha'), params.get('temperature'))
        config_metrics[config_key].append(result.get('metrics', {}).get('auc', 0))
    
    # Find best config averaged across folds
    best_avg_config = None
    best_avg_auc = 0
    for config_key, aucs in config_metrics.items():
        avg_auc = sum(aucs) / len(aucs)
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            best_avg_config = config_key
    
    summary['best_config_averaged'] = {
        'alpha': best_avg_config[0] if best_avg_config else None,
        'temperature': best_avg_config[1] if best_avg_config else None,
        'avg_auc': best_avg_auc
    }
    
    return summary


def save_summary(summary: Dict[str, Any], output_dir: str) -> str:
    """Save summary to JSON file."""
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    return summary_path


def create_visualizations(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Create visualization plots from results."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available, skipping visualizations")
        return
    
    successful = [r for r in results if r.get('success', False)]
    if not successful:
        print("No successful experiments to visualize")
        return
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data
    data = []
    for result in successful:
        exp = result['exp_params']
        metrics = result.get('metrics', {})
        data.append({
            'alpha': exp.get('alpha'),
            'temperature': exp.get('temperature'),
            'fold': exp.get('fold'),
            'auc': metrics.get('auc', 0),
            'f1': metrics.get('f1', 0),
            'miss_rate': metrics.get('miss_rate', 0),
            'false_alarm_rate': metrics.get('false_alarm_rate', 0)
        })
    
    # Check if we have both alpha and temperature
    has_alpha = all(d['alpha'] is not None for d in data)
    has_temperature = all(d['temperature'] is not None for d in data)
    
    if has_alpha and has_temperature:
        # Create heatmap
        create_heatmap(data, plots_dir)
    
    if has_alpha:
        create_alpha_plot(data, plots_dir)
    
    if has_temperature:
        create_temperature_plot(data, plots_dir)
    
    print(f"Plots saved to: {plots_dir}")


def create_heatmap(data: List[Dict], plots_dir: str) -> None:
    """Create alpha x temperature heatmap."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from collections import defaultdict
    except ImportError:
        return
    
    # Aggregate by alpha and temperature
    aggregated = defaultdict(list)
    for d in data:
        key = (d['alpha'], d['temperature'])
        aggregated[key].append(d['auc'])
    
    # Average across folds
    alphas = sorted(set(d['alpha'] for d in data))
    temperatures = sorted(set(d['temperature'] for d in data))
    
    heatmap_data = np.zeros((len(alphas), len(temperatures)))
    for i, alpha in enumerate(alphas):
        for j, temp in enumerate(temperatures):
            if (alpha, temp) in aggregated:
                heatmap_data[i, j] = np.mean(aggregated[(alpha, temp)])
            else:
                heatmap_data[i, j] = np.nan
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=[f'{t:.1f}' for t in temperatures],
        yticklabels=[f'{a:.2f}' for a in alphas],
        annot=True,
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'AUC'}
    )
    plt.xlabel('Temperature')
    plt.ylabel('Alpha')
    plt.title('AUC by Alpha and Temperature')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'heatmap_auc.png'), dpi=150)
    plt.close()


def create_alpha_plot(data: List[Dict], plots_dir: str) -> None:
    """Create AUC vs alpha plot."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
    except ImportError:
        return
    
    # Group by temperature if available
    temps = sorted(set(d['temperature'] for d in data if d['temperature'] is not None))
    
    plt.figure(figsize=(10, 6))
    
    if len(temps) > 1:
        # Plot lines for each temperature
        for temp in temps:
            temp_data = [d for d in data if d['temperature'] == temp]
            alphas = sorted(set(d['alpha'] for d in temp_data))
            aucs = []
            auc_stds = []
            for alpha in alphas:
                alpha_aucs = [d['auc'] for d in temp_data if d['alpha'] == alpha]
                aucs.append(np.mean(alpha_aucs))
                auc_stds.append(np.std(alpha_aucs) if len(alpha_aucs) > 1 else 0)
            plt.errorbar(alphas, aucs, yerr=auc_stds, marker='o', label=f'T={temp:.1f}', capsize=3)
        plt.legend(title='Temperature')
    else:
        # Just plot alpha vs AUC
        alphas = sorted(set(d['alpha'] for d in data))
        aucs = []
        auc_stds = []
        for alpha in alphas:
            alpha_aucs = [d['auc'] for d in data if d['alpha'] == alpha]
            aucs.append(np.mean(alpha_aucs))
            auc_stds.append(np.std(alpha_aucs) if len(alpha_aucs) > 1 else 0)
        plt.errorbar(alphas, aucs, yerr=auc_stds, marker='o', capsize=3, color='steelblue')
    
    plt.xlabel('Alpha')
    plt.ylabel('AUC')
    plt.title('AUC vs Alpha')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'alpha_vs_auc.png'), dpi=150)
    plt.close()


def create_temperature_plot(data: List[Dict], plots_dir: str) -> None:
    """Create AUC vs temperature plot."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    # Group by alpha if available
    alphas = sorted(set(d['alpha'] for d in data if d['alpha'] is not None))
    
    plt.figure(figsize=(10, 6))
    
    if len(alphas) > 1:
        # Plot lines for each alpha
        for alpha in alphas:
            alpha_data = [d for d in data if d['alpha'] == alpha]
            temps = sorted(set(d['temperature'] for d in alpha_data))
            aucs = []
            auc_stds = []
            for temp in temps:
                temp_aucs = [d['auc'] for d in alpha_data if d['temperature'] == temp]
                aucs.append(np.mean(temp_aucs))
                auc_stds.append(np.std(temp_aucs) if len(temp_aucs) > 1 else 0)
            plt.errorbar(temps, aucs, yerr=auc_stds, marker='o', label=f'α={alpha:.2f}', capsize=3)
        plt.legend(title='Alpha')
    else:
        # Just plot temperature vs AUC
        temps = sorted(set(d['temperature'] for d in data))
        aucs = []
        auc_stds = []
        for temp in temps:
            temp_aucs = [d['auc'] for d in data if d['temperature'] == temp]
            aucs.append(np.mean(temp_aucs))
            auc_stds.append(np.std(temp_aucs) if len(temp_aucs) > 1 else 0)
        plt.errorbar(temps, aucs, yerr=auc_stds, marker='o', capsize=3, color='coral')
    
    plt.xlabel('Temperature')
    plt.ylabel('AUC')
    plt.title('AUC vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'temperature_vs_auc.png'), dpi=150)
    plt.close()


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print summary table of results."""
    successful = [r for r in results if r.get('success', False)]
    
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    
    if successful:
        # Find best result
        best = max(successful, key=lambda r: r.get('metrics', {}).get('auc', 0))
        print("\nBest configuration:")
        print(f"  Alpha: {best['exp_params'].get('alpha', 'N/A')}")
        print(f"  Temperature: {best['exp_params'].get('temperature', 'N/A')}")
        print(f"  Fold: {best['exp_params'].get('fold', 'N/A')}")
        metrics = best.get('metrics', {})
        print(f"  AUC: {metrics.get('auc', 0):.4f}")
        print(f"  F1: {metrics.get('f1', 0):.4f}")
        print(f"  Miss Rate: {metrics.get('miss_rate', 0):.4f}")
        print(f"  False Alarm Rate: {metrics.get('false_alarm_rate', 0):.4f}")
    
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Validate arguments
    try:
        validate_args(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Parse fixed parameters
    fixed_params = parse_fixed_params(args.fixed)
    
    # Load base config
    print(f"Loading base config from: {args.base_config}")
    base_config = load_base_config(args.base_config)
    
    # Generate experiments
    if args.ranges is not None:
        # Random search
        print(f"\nGenerating random search: {args.n_samples} samples per fold")
        experiments = generate_random_search(
            args.params, 
            args.ranges, 
            args.n_samples, 
            args.folds,
            fixed_params,
            seed=args.seed
        )
    else:
        # Grid search
        print(f"\nGenerating grid search")
        experiments = generate_grid_search(
            args.params, 
            args.values_list, 
            args.folds,
            fixed_params
        )
    
    print(f"Total experiments: {len(experiments)}")
    
    # Dry run
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Experiments that would be executed:")
        print("=" * 60)
        for i, exp in enumerate(experiments[:20], 1):
            exp_name = experiment_to_dirname(exp)
            print(f"  {i}. {exp_name}")
            if i == 20 and len(experiments) > 20:
                print(f"  ... and {len(experiments) - 20} more")
                break
        print("=" * 60)
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run sweep
    start_time = time.time()
    results = run_sweep(
        experiments,
        base_config,
        args.base_config,
        args.output_dir,
        parallel=args.parallel,
        resume=args.resume,
        max_experiments=args.max_experiments,
        epochs=args.epochs,
        patience=args.patience
    )
    sweep_duration = time.time() - start_time
    
    # Save results
    csv_path = save_results_csv(results, args.output_dir)
    
    # Compute and save summary
    summary = compute_summary_stats(results)
    summary['sweep_duration_seconds'] = sweep_duration
    summary['sweep_duration_hours'] = sweep_duration / 3600
    save_summary(summary, args.output_dir)
    
    # Create visualizations
    create_visualizations(results, args.output_dir)
    
    # Print summary
    print_summary_table(results)
    
    print(f"\nSweep completed in {sweep_duration/3600:.2f} hours")
    print(f"Results directory: {args.output_dir}")


if __name__ == '__main__':
    main()
