"""
Analyze command for VAD distillation.

Analysis and visualization operations.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from cli.utils import (
    print_error,
    print_info,
    print_success,
    with_project_root
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add analyze-specific arguments."""
    parser.add_argument(
        'subcommand',
        nargs='?',
        choices=['compare', 'report', 'full'],
        default='full',
        help='Analysis subcommand'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/',
        help='Output directory to analyze'
    )
    parser.add_argument(
        '--fold',
        type=str,
        default=None,
        help='Analyze specific fold'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default=None,
        help='Comma-separated method directories'
    )
    parser.add_argument(
        '--method-names',
        type=str,
        default=None,
        help='Comma-separated method names'
    )
    parser.add_argument(
        '--with-baselines',
        action='store_true',
        help='Include baseline comparison'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='manifests/torgo_sentences.csv',
        help='Manifest path'
    )


@with_project_root
def main(args: argparse.Namespace) -> int:
    """
    Execute analyze command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    if args.subcommand == 'compare' or (args.methods and args.subcommand != 'report'):
        return run_comparison(args)
    elif args.subcommand == 'report':
        return generate_report(args)
    else:
        # Full analysis
        print("="*60)
        print("FULL ANALYSIS")
        print("="*60)
        
        ret = generate_report(args)
        if ret != 0:
            return ret
        
        if args.with_baselines:
            args.methods = f"outputs/baselines/silero,{args.output_dir}"
            args.method_names = "Silero,Student"
            return run_comparison(args)
        
        return 0


def run_comparison(args: argparse.Namespace) -> int:
    """Run comparison analysis."""
    if not args.methods:
        print_error("--methods required for comparison")
        return 2
    
    output_dir = Path(args.output_dir) / 'comparison'
    
    cmd = [
        sys.executable,
        'scripts/analysis/compare_methods.py',
        '--manifest', args.manifest,
        '--methods', args.methods,
        '--output-dir', str(output_dir),
        '--proxy-labels', 'teacher'
    ]
    
    if args.method_names:
        cmd.extend(['--method-names', args.method_names])
    
    print_info("Running comparison analysis...")
    
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print_success(f"Comparison saved to {output_dir}")
        return result.returncode
    except Exception as e:
        print_error(f"Comparison failed: {e}")
        return 1


def generate_report(args: argparse.Namespace) -> int:
    """Generate analysis report."""
    output_path = Path(args.output_dir)
    
    print(f"Analyzing: {output_path}")
    
    # Check for summary files
    log_dir = output_path / 'logs'
    if not log_dir.exists():
        print_error(f"No logs directory found in {output_path}")
        return 4
    
    summary_files = list(log_dir.glob('*_summary.json'))
    if not summary_files:
        print_error("No summary files found")
        return 4
    
    print_success(f"Found {len(summary_files)} summary files")
    
    # TODO: Generate comprehensive report with plots
    print_info("Report generation not yet fully implemented")
    print_info(f"Summary files available in: {log_dir}")
    
    return 0
