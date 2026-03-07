"""
Baseline command for VAD distillation.

Runs baseline VAD methods (Silero, Energy, SpeechBrain).
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
    get_device_preference,
    with_project_root
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add baseline-specific arguments."""
    parser.add_argument(
        'method',
        nargs='?',
        choices=['silero', 'energy', 'speechbrain', 'all'],
        default='all',
        help='Baseline method to run (default: all)'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='manifests/torgo_sentences.csv',
        help='Path to manifest CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: outputs/baselines/{method}/)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device for inference (cpu/cuda/auto)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode (process only 5 files)'
    )


@with_project_root
def main(args: argparse.Namespace) -> int:
    """
    Execute baseline command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    device = get_device_preference(args.device)
    
    # Determine methods to run
    if args.method == 'all':
        methods = ['silero', 'energy', 'speechbrain']
    else:
        methods = [args.method]
    
    print("="*60)
    print("BASELINE METHOD EXECUTION")
    print("="*60)
    print(f"Methods: {', '.join(methods)}")
    print(f"Device: {device}")
    print(f"Test mode: {args.test}")
    print()
    
    success_count = 0
    
    for method in methods:
        print(f"\nRunning {method.upper()}...")
        print("-"*40)
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = f"outputs/baselines/{method}/"
        
        success = run_baseline_method(
            method=method,
            manifest=args.manifest,
            output_dir=output_dir,
            device=device,
            test=args.test
        )
        
        if success:
            success_count += 1
            print_success(f"{method} completed")
        else:
            print_error(f"{method} failed")
    
    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print('='*60)
    print(f"Completed: {success_count}/{len(methods)}")
    
    return 0 if success_count == len(methods) else 1


def run_baseline_method(
    method: str,
    manifest: str,
    output_dir: str,
    device: str,
    test: bool
) -> bool:
    """
    Run a single baseline method.
    
    Args:
        method: Method name (silero/energy/speechbrain)
        manifest: Path to manifest CSV
        output_dir: Output directory
        device: Device to use
        test: Test mode flag
        
    Returns:
        True if successful
    """
    cmd = [
        sys.executable,
        'scripts/core/run_baseline.py',
        '--method', method,
        '--manifest', manifest,
        '--output-dir', output_dir,
        '--device', device
    ]
    
    if test:
        cmd.append('--test')
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Baseline {method} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print_warning(f"Baseline {method} interrupted")
        return False
