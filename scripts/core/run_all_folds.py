#!/usr/bin/env python3
"""
Run all 15 LOSO folds for VAD distillation training.

This script loops through all TORGO speaker folds and runs training
using train_loso.py with proper error handling and progress tracking.

Usage:
    # Run all folds sequentially with default config
    python scripts/core/run_all_folds.py
    
    # Run with custom config
    python scripts/core/run_all_folds.py --config configs/pilot_cuda.yaml
    
    # Run specific folds
    python scripts/core/run_all_folds.py --folds F01 M01 FC01
    
    # Run with parallel processing (2 folds at a time)
    python scripts/core/run_all_folds.py --parallel 2
    
    # Resume training for all folds
    python scripts/core/run_all_folds.py --resume

Author: VAD Distillation Project
Date: 2026-03-07
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Default 15 TORGO speaker folds
DEFAULT_FOLDS = [
    "F01", "F03", "F04",
    "M01", "M02", "M03", "M04", "M05",
    "FC01", "FC02", "FC03",
    "MC01", "MC02", "MC03", "MC04"
]

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def get_colored_output():
    """Get colored output functions based on platform support."""
    try:
        import colorama
        colorama.init()
        GREEN = colorama.Fore.GREEN
        RED = colorama.Fore.RED
        YELLOW = colorama.Fore.YELLOW
        BLUE = colorama.Fore.BLUE
        CYAN = colorama.Fore.CYAN
        RESET = colorama.Style.RESET_ALL
        BOLD = colorama.Style.BRIGHT
    except ImportError:
        GREEN = RED = YELLOW = BLUE = CYAN = RESET = BOLD = ""
    return GREEN, RED, YELLOW, BLUE, CYAN, RESET, BOLD


def print_header(text: str):
    """Print a formatted header."""
    _, _, _, _, CYAN, RESET, BOLD = get_colored_output()
    width = 70
    print("\n" + "=" * width)
    print(f"{BOLD}{CYAN}{text.center(width)}{RESET}")
    print("=" * width + "\n")


def print_fold_status(fold: str, status: str, message: str = ""):
    """Print fold status with color coding."""
    GREEN, RED, YELLOW, _, CYAN, RESET, BOLD = get_colored_output()
    
    status_colors = {
        "PENDING": YELLOW,
        "RUNNING": BLUE,
        "SUCCESS": GREEN,
        "FAILED": RED,
        "SKIPPED": YELLOW
    }
    
    color = status_colors.get(status, "")
    status_str = f"{BOLD}[{status}]{RESET}" if status in ["SUCCESS", "FAILED"] else f"[{status}]"
    
    print(f"  {CYAN}{fold}{RESET}: {color}{status_str}{RESET} {message}")


def run_single_fold(
    fold: str,
    config: str,
    resume: bool,
    extra_args: List[str],
    verbose: bool = False
) -> Tuple[str, bool, str, float]:
    """
    Run training for a single fold.
    
    Args:
        fold: Fold ID (e.g., F01)
        config: Path to config file
        resume: Whether to resume from checkpoint
        extra_args: Additional arguments to pass to train_loso.py
        verbose: Whether to show full training output
        
    Returns:
        Tuple of (fold_id, success, message, duration_seconds)
    """
    GREEN, RED, YELLOW, _, CYAN, RESET, BOLD = get_colored_output()
    
    start_time = time.time()
    
    # Build command
    cmd = [
        sys.executable,
        "train_loso.py",
        "--config", config,
        "--fold", fold
    ]
    
    if resume:
        checkpoint_path = f"outputs/{Path(config).stem}/checkpoints/fold_{fold}_latest.pt"
        if Path(checkpoint_path).exists():
            cmd.extend(["--resume", checkpoint_path])
    
    # Add any extra arguments
    cmd.extend(extra_args)
    
    # Determine working directory
    cwd = PROJECT_ROOT
    
    try:
        # Run training
        if verbose:
            # Show full output
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                capture_output=False,
                text=True
            )
        else:
            # Capture output, only show on error
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True
            )
            
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return (fold, True, f"Completed in {duration/60:.1f} min", duration)
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return (fold, False, f"Exit code {result.returncode}: {error_msg}", duration)
            
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        error_msg = e.stderr[-500:] if e.stderr else str(e)
        return (fold, False, f"Error: {error_msg}", duration)
    except Exception as e:
        duration = time.time() - start_time
        return (fold, False, f"Exception: {str(e)}", duration)


def run_all_folds(
    folds: List[str],
    config: str,
    parallel: int,
    resume: bool,
    extra_args: List[str],
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Run training for all specified folds.
    
    Args:
        folds: List of fold IDs
        config: Path to config file
        parallel: Number of parallel jobs
        resume: Whether to resume from checkpoints
        extra_args: Additional arguments for train_loso.py
        verbose: Whether to show verbose output
        
    Returns:
        Dictionary with results for each fold
    """
    GREEN, RED, YELLOW, BLUE, CYAN, RESET, BOLD = get_colored_output()
    
    results = {}
    total_start = time.time()
    
    print_header("LOSO Training - All Folds")
    
    print(f"{BOLD}Configuration:{RESET}")
    print(f"  Config file: {CYAN}{config}{RESET}")
    print(f"  Total folds: {CYAN}{len(folds)}{RESET}")
    print(f"  Parallel jobs: {CYAN}{parallel}{RESET}")
    print(f"  Resume mode: {CYAN}{resume}{RESET}")
    print(f"  Working dir: {CYAN}{PROJECT_ROOT}{RESET}")
    print()
    
    if parallel > 1:
        # Parallel execution
        print(f"{BOLD}Running folds in parallel (max {parallel} concurrent)...{RESET}\n")
        
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            # Submit all jobs
            future_to_fold = {
                executor.submit(
                    run_single_fold, fold, config, resume, extra_args, verbose
                ): fold for fold in folds
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_fold):
                fold, success, message, duration = future.result()
                completed += 1
                
                results[fold] = {
                    "success": success,
                    "message": message,
                    "duration": duration
                }
                
                status = "SUCCESS" if success else "FAILED"
                progress = f"({completed}/{len(folds)})"
                print(f"  {CYAN}{fold}{RESET} {progress}: {GREEN if success else RED}{status}{RESET} - {message}")
    else:
        # Sequential execution
        print(f"{BOLD}Running folds sequentially...{RESET}\n")
        
        for i, fold in enumerate(folds, 1):
            progress = f"({i}/{len(folds)})"
            print(f"\n{BOLD}Starting fold {CYAN}{fold}{RESET} {progress}{RESET}")
            print("-" * 50)
            
            fold, success, message, duration = run_single_fold(
                fold, config, resume, extra_args, verbose
            )
            
            results[fold] = {
                "success": success,
                "message": message,
                "duration": duration
            }
            
            status = "SUCCESS" if success else "FAILED"
            status_color = GREEN if success else RED
            print(f"\n{BOLD}Fold {CYAN}{fold}{RESET} {status_color}{status}{RESET}: {message}{RESET}")
    
    total_duration = time.time() - total_start
    results["_meta"] = {
        "total_duration": total_duration,
        "completed_at": datetime.now().isoformat()
    }
    
    return results


def print_summary(results: Dict[str, Dict], folds: List[str]):
    """Print a summary of all fold results."""
    GREEN, RED, YELLOW, _, CYAN, RESET, BOLD = get_colored_output()
    
    print_header("Training Summary")
    
    success_count = sum(1 for f in folds if results.get(f, {}).get("success", False))
    failed_count = len(folds) - success_count
    
    # Summary stats
    print(f"{BOLD}Overall Results:{RESET}")
    print(f"  Total folds: {CYAN}{len(folds)}{RESET}")
    print(f"  Successful:  {GREEN}{success_count}{RESET}")
    print(f"  Failed:      {RED if failed_count > 0 else GREEN}{failed_count}{RESET}")
    
    if "_meta" in results:
        total_time = results["_meta"]["total_duration"]
        print(f"  Total time:  {CYAN}{timedelta(seconds=int(total_time))}{RESET}")
        print(f"  Avg time/fold: {CYAN}{total_time/len(folds)/60:.1f} min{RESET}")
    print()
    
    # Individual fold details
    print(f"{BOLD}Fold Details:{RESET}")
    
    # First show successful folds
    for fold in folds:
        if results.get(fold, {}).get("success", False):
            duration = results[fold].get("duration", 0)
            print_fold_status(fold, "SUCCESS", f"{duration/60:.1f} min")
    
    # Then show failed folds
    for fold in folds:
        if not results.get(fold, {}).get("success", False):
            message = results.get(fold, {}).get("message", "Unknown error")
            # Truncate long messages
            if len(message) > 60:
                message = message[:57] + "..."
            print_fold_status(fold, "FAILED", message)
    
    print()
    
    # Final status
    if failed_count == 0:
        print(f"{BOLD}{GREEN}✓ All {len(folds)} folds completed successfully!{RESET}\n")
    else:
        print(f"{BOLD}{YELLOW}⚠ {success_count}/{len(folds)} folds completed. {failed_count} fold(s) failed.{RESET}\n")
        print(f"{BOLD}To retry failed folds:{RESET}")
        failed_folds = [f for f in folds if not results.get(f, {}).get("success", False)]
        print(f"  python scripts/core/run_all_folds.py --folds {' '.join(failed_folds)}")
        print()


def save_results(results: Dict[str, Dict], output_path: Path):
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run all 15 LOSO folds for VAD distillation training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all folds with default config
  python scripts/core/run_all_folds.py
  
  # Run with custom config
  python scripts/core/run_all_folds.py --config configs/pilot_cuda.yaml
  
  # Run specific folds only
  python scripts/core/run_all_folds.py --folds F01 M01 FC01
  
  # Run 2 folds in parallel
  python scripts/core/run_all_folds.py --parallel 2
  
  # Resume interrupted training
  python scripts/core/run_all_folds.py --resume
  
  # Quick test mode (dry run)
  python scripts/core/run_all_folds.py --test
  
  # Verbose output
  python scripts/core/run_all_folds.py --verbose
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/production_cuda.yaml",
        help="Path to configuration YAML file (default: configs/production_cuda.yaml)"
    )
    
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help=f"List of folds to run (default: all {len(DEFAULT_FOLDS)} folds)"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of folds to run in parallel (default: 1, sequential)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoints if available"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (passes --test to train_loso.py)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full training output for each fold"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: auto-generated in outputs/)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs in config"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Determine folds to run
    folds = args.folds if args.folds else DEFAULT_FOLDS
    
    # Validate config file exists
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Build extra arguments
    extra_args = []
    if args.test:
        extra_args.append("--test")
    if args.epochs:
        extra_args.extend(["--epochs", str(args.epochs)])
    if args.device:
        extra_args.extend(["--device", args.device])
    
    # Run training
    try:
        results = run_all_folds(
            folds=folds,
            config=args.config,
            parallel=args.parallel,
            resume=args.resume,
            extra_args=extra_args,
            verbose=args.verbose
        )
        
        # Print summary
        print_summary(results, folds)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = Path(args.config).stem
            output_dir = PROJECT_ROOT / "outputs" / f"fold_runs_{config_name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"results_{timestamp}.json"
        
        save_results(results, output_path)
        
        # Exit with error code if any fold failed
        failed_count = sum(1 for f in folds if not results.get(f, {}).get("success", False))
        sys.exit(0 if failed_count == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
