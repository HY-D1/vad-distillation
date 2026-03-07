"""
Training command for VAD distillation.

Wraps train_loso.py with a unified interface.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from cli.config import Config, get_all_folds
from cli.utils import (
    ensure_project_root,
    print_error,
    print_info,
    print_success,
    print_warning,
    get_device_preference,
    with_project_root
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add train-specific arguments to parser."""
    parser.add_argument(
        '--fold',
        type=str,
        default=None,
        help='Fold ID to train (e.g., F01)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train all folds sequentially'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/production.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        const=True,
        default=False,
        help='Resume from checkpoint (auto-detect or specify path)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (F01, 2 epochs)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Verify setup without training'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel jobs for --all'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Override alpha (distillation weight)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Override temperature for softening'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (cpu/cuda/mps/auto)'
    )


@with_project_root
def main(args: argparse.Namespace) -> int:
    """
    Execute training command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    # Handle quick mode
    if args.quick:
        print_info("Quick mode: Using F01 with 2 epochs")
        args.fold = 'F01'
        args.config = 'configs/pilot.yaml'
        args.epochs = 2
    
    # Validate arguments
    if not args.fold and not args.all:
        print_error(
            "Must specify either --fold or --all",
            "Use 'python vad.py train --fold F01' or 'python vad.py train --all'"
        )
        return 2
    
    if args.fold and args.all:
        print_warning("Both --fold and --all specified, using --all")
        args.fold = None
    
    # Load configuration
    config = Config(args.config)
    
    # Get device
    device = get_device_preference(args.device)
    print_info(f"Using device: {device}")
    
    # Determine folds to train
    if args.all:
        folds = get_all_folds(config.get('splits_dir', 'splits'))
        if not folds:
            print_error(
                "No folds found in splits directory",
                "Run 'python scripts/data/generate_loso_splits.py' first"
            )
            return 4
        print_info(f"Training {len(folds)} folds: {', '.join(folds)}")
    else:
        folds = [args.fold]
    
    # Training results
    success_count = 0
    failed_folds = []
    
    # Train each fold
    for fold in folds:
        print(f"\n{'='*60}")
        print(f"Training fold: {fold}")
        print('='*60)
        
        success = train_single_fold(
            fold=fold,
            config_path=args.config,
            resume=args.resume,
            dry_run=args.dry_run,
            alpha=args.alpha,
            temperature=args.temperature,
            epochs=args.epochs,
            device=device
        )
        
        if success:
            success_count += 1
            print_success(f"Fold {fold} completed successfully")
        else:
            failed_folds.append(fold)
            print_error(f"Fold {fold} failed")
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print('='*60)
    print(f"Completed: {success_count}/{len(folds)}")
    
    if failed_folds:
        print(f"Failed: {', '.join(failed_folds)}")
    
    return 0 if len(failed_folds) == 0 else 6


def train_single_fold(
    fold: str,
    config_path: str,
    resume: bool = False,
    dry_run: bool = False,
    alpha: Optional[float] = None,
    temperature: Optional[float] = None,
    epochs: Optional[int] = None,
    device: str = 'cpu'
) -> bool:
    """
    Train a single fold.
    
    Args:
        fold: Fold ID
        config_path: Path to config file
        resume: Whether to resume
        dry_run: Whether to do dry run only
        alpha: Override alpha value
        temperature: Override temperature value
        epochs: Override epochs
        device: Device to use
        
    Returns:
        True if successful
    """
    # Build command
    cmd = [
        sys.executable,
        'train_loso.py',
        '--config', config_path,
        '--fold', fold,
        '--device', device
    ]
    
    # Add resume if specified
    if resume:
        if isinstance(resume, str):
            cmd.extend(['--resume', resume])
        else:
            # Auto-detect checkpoint
            checkpoint_path = f"outputs/checkpoints/fold_{fold}_latest.pt"
            if Path(checkpoint_path).exists():
                cmd.extend(['--resume', checkpoint_path])
                print_info(f"Resuming from: {checkpoint_path}")
    
    # Add overrides
    if epochs is not None:
        cmd.extend(['--epochs', str(epochs)])
    
    if dry_run:
        print_info("DRY RUN - Would execute:")
        print(' '.join(cmd))
        return True
    
    # Execute training
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print_warning("Training interrupted by user")
        return False
