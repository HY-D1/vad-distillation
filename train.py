#!/usr/bin/env python3
"""
Training script for VAD knowledge distillation - Smoke Test Wrapper.

This script is a simple wrapper around train_loso.py for quick smoke testing
using the pilot configuration. For full LOSO training, use train_loso.py directly.

Usage:
    # Quick smoke test with pilot config
    python train.py --config configs/pilot.yaml
    
    # With specific fold
    python train.py --config configs/pilot.yaml --fold F01
    
    # Dry-run test mode (no actual training)
    python train.py --config configs/pilot.yaml --fold F01 --test

For full LOSO training:
    python train_loso.py --config configs/pilot.yaml --fold F01
    
For hyperparameter sweeps:
    python scripts/run_sweep.py --help
    python scripts/run_experiment.py --help
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VAD Distillation Training - Smoke Test Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test with pilot config
  python train.py --config configs/pilot.yaml
  
  # Test specific fold
  python train.py --config configs/pilot.yaml --fold F01
  
  # Dry-run mode (verify setup without training)
  python train.py --config configs/pilot.yaml --fold F01 --test

For full training options, see train_loso.py --help
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pilot.yaml",
        help="Path to config file (default: configs/pilot.yaml)"
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="F01",
        help="Fold ID for LOSO training (default: F01)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test/dry-run mode (verify setup without training)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified."
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Additional arguments to pass to train_loso.py (quoted string)"
    )
    
    return parser.parse_args()


def main():
    """Run the smoke test training wrapper."""
    args = parse_args()
    
    # Check that config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("\nMake sure you have created the pilot config:")
        print("  cp configs/pilot_example.yaml configs/pilot.yaml")
        sys.exit(1)
    
    # Build command to call train_loso.py
    cmd = [
        sys.executable,
        "train_loso.py",
        "--config", str(config_path),
        "--fold", args.fold,
    ]
    
    # Add optional arguments
    if args.test:
        cmd.append("--test")
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Add any extra arguments
    if args.extra_args:
        cmd.extend(args.extra_args.split())
    
    print("=" * 70)
    print("VAD Distillation Training - Smoke Test Wrapper")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Fold: {args.fold}")
    print(f"Test mode: {args.test}")
    print("-" * 70)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 70)
    print()
    
    # Run train_loso.py as a subprocess
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
