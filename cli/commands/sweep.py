"""
Sweep command for VAD distillation.

Hyperparameter sweep wrapper.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from cli.utils import (
    print_error,
    print_info,
    print_success,
    print_warning,
    with_project_root
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add sweep-specific arguments."""
    parser.add_argument(
        '--param',
        action='append',
        dest='params',
        help='Parameter name (repeatable)'
    )
    parser.add_argument(
        '--values',
        action='append',
        nargs='+',
        type=float,
        dest='values_list',
        help='Values for grid search'
    )
    parser.add_argument(
        '--range',
        action='append',
        nargs=2,
        type=float,
        dest='ranges',
        help='Min max for random search'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='Number of random samples'
    )
    parser.add_argument(
        '--fixed',
        action='append',
        help='Fixed parameters (key=value)'
    )
    parser.add_argument(
        '--folds',
        nargs='+',
        required=True,
        help='Folds to sweep'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pilot.yaml',
        help='Base config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Parallel jobs'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from output directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show without executing'
    )


@with_project_root
def main(args: argparse.Namespace) -> int:
    """
    Execute sweep command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    # Build command for run_sweep.py
    cmd = [
        sys.executable,
        'scripts/core/run_sweep.py',
        '--folds'
    ] + args.folds + [
        '--base-config', args.config
    ]
    
    # Add output directory
    if args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])
    
    # Add parallel
    if args.parallel > 1:
        cmd.extend(['--parallel', str(args.parallel)])
    
    # Add resume
    if args.resume:
        cmd.extend(['--resume'])
    
    # Add dry-run
    if args.dry_run:
        cmd.append('--dry-run')
    
    # Add parameters
    if args.params:
        for param in args.params:
            cmd.extend(['--param', param])
    
    if args.values_list:
        for values in args.values_list:
            cmd.extend(['--values'] + [str(v) for v in values])
    
    if args.ranges:
        for range_vals in args.ranges:
            cmd.extend(['--range'] + [str(v) for v in range_vals])
    
    if args.n_samples:
        cmd.extend(['--n-samples', str(args.n_samples)])
    
    if args.fixed:
        for fixed in args.fixed:
            cmd.extend(['--fixed', fixed])
    
    print("="*60)
    print("HYPERPARAMETER SWEEP")
    print("="*60)
    print(f"Folds: {', '.join(args.folds)}")
    print(f"Config: {args.config}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print()
    
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print_warning("\nSweep interrupted by user")
        return 130
