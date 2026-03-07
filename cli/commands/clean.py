"""
Clean command for VAD distillation.

Cleanup operations for temporary files and outputs.
"""

import argparse
import shutil
import sys
from pathlib import Path

from cli.utils import (
    confirm_action,
    print_error,
    print_info,
    print_success,
    print_warning,
    with_project_root
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add clean-specific arguments."""
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/',
        help='Target output directory'
    )
    parser.add_argument(
        '--checkpoints',
        action='store_true',
        help='Clean checkpoints'
    )
    parser.add_argument(
        '--logs',
        action='store_true',
        help='Clean logs'
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Clean cache files'
    )
    parser.add_argument(
        '--keep-best',
        action='store_true',
        help='Keep best checkpoint when cleaning checkpoints'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Clean everything (use with caution)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without deleting'
    )


@with_project_root
def main(args: argparse.Namespace) -> int:
    """
    Execute clean command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    output_path = Path(args.output_dir)
    
    print("="*60)
    print("CLEANUP OPERATION")
    print("="*60)
    
    # Determine what to clean
    if args.all:
        if not args.force and not args.dry_run:
            if not confirm_action(
                f"This will DELETE ALL files in {output_path}. Are you sure?",
                default=False
            ):
                print_info("Cleanup cancelled")
                return 0
        
        return clean_all(output_path, args.dry_run)
    
    # Clean specific components
    cleaned_any = False
    
    if args.checkpoints or (not args.logs and not args.cache):
        cleaned = clean_checkpoints(output_path, args.keep_best, args.dry_run)
        cleaned_any = cleaned_any or cleaned
    
    if args.logs:
        cleaned = clean_logs(output_path, args.dry_run)
        cleaned_any = cleaned_any or cleaned
    
    if args.cache:
        cleaned = clean_cache(args.dry_run)
        cleaned_any = cleaned_any or cleaned
    
    if not cleaned_any:
        print_info("Nothing to clean. Use --all to clean everything.")
    
    return 0


def clean_all(output_path: Path, dry_run: bool) -> int:
    """Clean all output files."""
    if not output_path.exists():
        print_info(f"Output directory does not exist: {output_path}")
        return 0
    
    if dry_run:
        print(f"[DRY RUN] Would delete: {output_path}")
        for item in output_path.rglob('*'):
            print(f"  {item}")
        return 0
    
    try:
        shutil.rmtree(output_path)
        print_success(f"Deleted: {output_path}")
        return 0
    except Exception as e:
        print_error(f"Failed to delete {output_path}: {e}")
        return 1


def clean_checkpoints(output_path: Path, keep_best: bool, dry_run: bool) -> bool:
    """Clean checkpoint files."""
    checkpoint_dir = output_path / 'checkpoints'
    if not checkpoint_dir.exists():
        return False
    
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    if not checkpoints:
        return False
    
    to_delete = checkpoints
    if keep_best:
        # Keep files with "_best" in name
        to_delete = [c for c in checkpoints if '_best' not in c.name]
        kept = [c for c in checkpoints if '_best' in c.name]
        print_info(f"Keeping {len(kept)} best checkpoint(s)")
    
    if not to_delete:
        return False
    
    if dry_run:
        print(f"[DRY RUN] Would delete {len(to_delete)} checkpoint(s):")
        for ckpt in to_delete:
            size = ckpt.stat().st_size / (1024**2)
            print(f"  {ckpt.name} ({size:.2f} MB)")
    else:
        total_size = 0
        for ckpt in to_delete:
            size = ckpt.stat().st_size
            total_size += size
            ckpt.unlink()
        
        print_success(
            f"Deleted {len(to_delete)} checkpoint(s) "
            f"({total_size / (1024**2):.2f} MB)"
        )
    
    return True


def clean_logs(output_path: Path, dry_run: bool) -> bool:
    """Clean log files."""
    log_dir = output_path / 'logs'
    if not log_dir.exists():
        return False
    
    log_files = list(log_dir.glob('*.csv')) + list(log_dir.glob('*.txt'))
    if not log_files:
        return False
    
    if dry_run:
        print(f"[DRY RUN] Would delete {len(log_files)} log file(s):")
        for log in log_files:
            print(f"  {log.name}")
    else:
        for log in log_files:
            log.unlink()
        print_success(f"Deleted {len(log_files)} log file(s)")
    
    return True


def clean_cache(dry_run: bool) -> bool:
    """Clean cache files."""
    # Find __pycache__ directories
    cache_dirs = list(Path('.').rglob('__pycache__'))
    
    if not cache_dirs:
        return False
    
    if dry_run:
        print(f"[DRY RUN] Would delete {len(cache_dirs)} cache directory(s)")
    else:
        for cache_dir in cache_dirs:
            shutil.rmtree(cache_dir, ignore_errors=True)
        print_success(f"Deleted {len(cache_dirs)} cache directory(s)")
    
    return True
