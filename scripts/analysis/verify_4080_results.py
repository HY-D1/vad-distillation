#!/usr/bin/env python3
"""
Verification script for Windows RTX 4080 training results on Mac.

Validates all fold checkpoints from Windows 4080 without re-running inference.
Checks that metrics are within expected thresholds and files are complete.

Usage:
    python scripts/verify_4080_results.py --results-dir outputs/production_cuda/
    
    # With custom thresholds
    python scripts/verify_4080_results.py --results-dir outputs/production_cuda/ \
        --min-auc 0.80 --min-f1 0.65 --max-miss-rate 0.25
    
    # Save report to file
    python scripts/verify_4080_results.py --results-dir outputs/production_cuda/ \
        --output verification_report.json

Expected directory structure:
    outputs/production_cuda/
    ├── checkpoints/
    │   ├── fold_F01_best.pt
    │   ├── fold_F02_best.pt
    │   └── ...
    └── logs/
        ├── fold_F01_summary.json
        ├── fold_F02_summary.json
        └── ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Status(Enum):
    """Verification status."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    MISSING = "MISSING"


@dataclass
class FoldResult:
    """Result of verification for a single fold."""
    fold_id: str
    checkpoint_exists: bool
    summary_exists: bool
    checkpoint_loadable: bool
    best_auc: Optional[float]
    test_metrics: Dict[str, float]
    status: Status
    issues: List[str]
    
    @property
    def auc(self) -> float:
        return self.test_metrics.get('auc', 0.0)
    
    @property
    def f1(self) -> float:
        return self.test_metrics.get('f1', 0.0)
    
    @property
    def miss_rate(self) -> float:
        return self.test_metrics.get('miss_rate', 1.0)
    
    @property
    def far(self) -> float:
        return self.test_metrics.get('false_alarm_rate', 1.0)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify Windows 4080 training results on Mac",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification
  python scripts/verify_4080_results.py --results-dir outputs/production_cuda/

  # Custom thresholds
  python scripts/verify_4080_results.py --results-dir outputs/production_cuda/ \\
      --min-auc 0.80 --min-f1 0.65 --max-miss-rate 0.25

  # Save detailed report
  python scripts/verify_4080_results.py --results-dir outputs/production_cuda/ \\
      --output verification_report.json
        """
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing Windows 4080 results (with checkpoints/ and logs/)'
    )
    parser.add_argument(
        '--checkpoints-dir',
        type=str,
        default=None,
        help='Directory containing checkpoints (default: {results_dir}/checkpoints/)'
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        default=None,
        help='Directory containing logs (default: {results_dir}/logs/)'
    )
    parser.add_argument(
        '--min-auc',
        type=float,
        default=0.85,
        help='Minimum acceptable AUC (default: 0.85)'
    )
    parser.add_argument(
        '--min-f1',
        type=float,
        default=0.70,
        help='Minimum acceptable F1 score (default: 0.70)'
    )
    parser.add_argument(
        '--max-miss-rate',
        type=float,
        default=0.20,
        help='Maximum acceptable miss rate (default: 0.20)'
    )
    parser.add_argument(
        '--max-far',
        type=float,
        default=0.20,
        help='Maximum acceptable false alarm rate (default: 0.20)'
    )
    parser.add_argument(
        '--min-best-auc',
        type=float,
        default=0.50,
        help='Minimum reasonable best_auc value (default: 0.50)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save verification report to JSON file'
    )
    parser.add_argument(
        '--verify-checkpoint-loadable',
        action='store_true',
        help='Attempt to load each checkpoint with torch (requires PyTorch)'
    )
    parser.add_argument(
        '--folds',
        type=str,
        default=None,
        help='Comma-separated list of fold IDs to verify (default: auto-detect all)'
    )

    return parser.parse_args()


def find_all_folds(logs_dir: Path) -> List[str]:
    """Auto-detect all folds from summary files."""
    folds = []
    if logs_dir.exists():
        for f in logs_dir.glob("fold_*_summary.json"):
            fold_id = f.stem.replace("fold_", "").replace("_summary", "")
            folds.append(fold_id)
    return sorted(folds)


def load_summary(logs_dir: Path, fold_id: str) -> Optional[Dict]:
    """Load summary.json for a fold."""
    summary_path = logs_dir / f"fold_{fold_id}_summary.json"
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return None


def verify_checkpoint_loadable(checkpoint_path: Path) -> bool:
    """Verify that a checkpoint can be loaded with torch."""
    if not TORCH_AVAILABLE:
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Check required keys exist
        required_keys = ['model_state_dict', 'epoch']
        return all(key in checkpoint for key in required_keys)
    except Exception:
        return False


def verify_fold(
    fold_id: str,
    checkpoints_dir: Path,
    logs_dir: Path,
    args: argparse.Namespace
) -> FoldResult:
    """Verify a single fold's results."""
    issues = []
    
    # Check files exist
    checkpoint_path = checkpoints_dir / f"fold_{fold_id}_best.pt"
    checkpoint_exists = checkpoint_path.exists()
    
    summary = load_summary(logs_dir, fold_id)
    summary_exists = summary is not None
    
    # Try to load checkpoint if requested
    checkpoint_loadable = False
    if args.verify_checkpoint_loadable and checkpoint_exists:
        checkpoint_loadable = verify_checkpoint_loadable(checkpoint_path)
        if not checkpoint_loadable:
            issues.append("Checkpoint not loadable")
    
    # Extract metrics from summary
    best_auc = None
    test_metrics = {}
    
    if summary:
        best_auc = summary.get('best_val_auc')
        test_metrics = summary.get('test_metrics', {})
    else:
        issues.append("Summary file missing or invalid")
    
    # Determine overall status
    status = Status.PASS
    
    if not checkpoint_exists:
        status = Status.MISSING
        issues.append(f"Checkpoint missing: {checkpoint_path.name}")
    
    if not summary_exists:
        status = Status.MISSING
        issues.append("Summary JSON missing")
    
    # Check best_auc is reasonable
    if best_auc is not None:
        if best_auc < args.min_best_auc:
            status = Status.FAIL
            issues.append(f"best_auc ({best_auc:.4f}) < minimum ({args.min_best_auc})")
    else:
        if summary_exists:
            status = Status.FAIL
            issues.append("best_auc field missing from summary")
    
    # Check test metrics against thresholds
    if test_metrics:
        auc = test_metrics.get('auc', 0)
        f1 = test_metrics.get('f1', 0)
        miss_rate = test_metrics.get('miss_rate', 1)
        far = test_metrics.get('false_alarm_rate', 1)
        
        if auc < args.min_auc:
            if status == Status.PASS:
                status = Status.WARNING
            issues.append(f"AUC ({auc:.4f}) < threshold ({args.min_auc})")
        
        if f1 < args.min_f1:
            if status == Status.PASS:
                status = Status.WARNING
            issues.append(f"F1 ({f1:.4f}) < threshold ({args.min_f1})")
        
        if miss_rate > args.max_miss_rate:
            if status == Status.PASS:
                status = Status.WARNING
            issues.append(f"Miss Rate ({miss_rate:.4f}) > threshold ({args.max_miss_rate})")
        
        if far > args.max_far:
            if status == Status.PASS:
                status = Status.WARNING
            issues.append(f"FAR ({far:.4f}) > threshold ({args.max_far})")
    
    return FoldResult(
        fold_id=fold_id,
        checkpoint_exists=checkpoint_exists,
        summary_exists=summary_exists,
        checkpoint_loadable=checkpoint_loadable,
        best_auc=best_auc,
        test_metrics=test_metrics,
        status=status,
        issues=issues
    )


def print_summary_table(results: List[FoldResult], args: argparse.Namespace):
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("Windows RTX 4080 Results Verification Summary")
    print("=" * 100)
    
    # Header
    print(f"\n{'Fold':<8} {'Status':<10} {'AUC':>8} {'F1':>8} {'Miss%':>8} {'FAR%':>8} {'Best AUC':>10} {'Issues'}")
    print("-" * 100)
    
    # Rows
    for r in results:
        status_symbol = {
            Status.PASS: "✓ PASS",
            Status.FAIL: "✗ FAIL",
            Status.WARNING: "⚠ WARN",
            Status.MISSING: "✗ MISS"
        }.get(r.status, str(r.status))
        
        auc_str = f"{r.auc:.4f}" if r.auc > 0 else "N/A"
        f1_str = f"{r.f1:.4f}" if r.f1 > 0 else "N/A"
        miss_str = f"{r.miss_rate:.4f}" if r.miss_rate < 1 else "N/A"
        far_str = f"{r.far:.4f}" if r.far < 1 else "N/A"
        best_auc_str = f"{r.best_auc:.4f}" if r.best_auc else "N/A"
        
        issues_str = "; ".join(r.issues[:2]) if r.issues else ""
        if len(r.issues) > 2:
            issues_str += f" (+{len(r.issues) - 2} more)"
        
        print(f"{r.fold_id:<8} {status_symbol:<10} {auc_str:>8} {f1_str:>8} "
              f"{miss_str:>8} {far_str:>8} {best_auc_str:>10} {issues_str}")
    
    print("-" * 100)


def print_statistics(results: List[FoldResult], args: argparse.Namespace):
    """Print overall statistics."""
    total = len(results)
    passed = sum(1 for r in results if r.status == Status.PASS)
    warnings = sum(1 for r in results if r.status == Status.WARNING)
    failed = sum(1 for r in results if r.status == Status.FAIL)
    missing = sum(1 for r in results if r.status == Status.MISSING)
    
    # Calculate averages for passing folds
    valid_aucs = [r.auc for r in results if r.auc > 0]
    valid_f1s = [r.f1 for r in results if r.f1 > 0]
    valid_miss = [r.miss_rate for r in results if r.miss_rate < 1]
    valid_far = [r.far for r in results if r.far < 1]
    
    print("\n" + "=" * 100)
    print("Statistics")
    print("=" * 100)
    
    print(f"\nVerification Counts:")
    print(f"  Total folds:      {total}")
    print(f"  ✓ Passed:         {passed}")
    print(f"  ⚠ Warnings:       {warnings}")
    print(f"  ✗ Failed:         {failed}")
    print(f"  ✗ Missing:        {missing}")
    
    if valid_aucs:
        print(f"\nMetric Averages (valid folds only):")
        print(f"  AUC:              {sum(valid_aucs)/len(valid_aucs):.4f} "
              f"(min: {min(valid_aucs):.4f}, max: {max(valid_aucs):.4f})")
    if valid_f1s:
        print(f"  F1:               {sum(valid_f1s)/len(valid_f1s):.4f} "
              f"(min: {min(valid_f1s):.4f}, max: {max(valid_f1s):.4f})")
    if valid_miss:
        print(f"  Miss Rate:        {sum(valid_miss)/len(valid_miss):.4f} "
              f"(min: {min(valid_miss):.4f}, max: {max(valid_miss):.4f})")
    if valid_far:
        print(f"  FAR:              {sum(valid_far)/len(valid_far):.4f} "
              f"(min: {min(valid_far):.4f}, max: {max(valid_far):.4f})")
    
    print(f"\nThresholds Used:")
    print(f"  Min AUC:          {args.min_auc}")
    print(f"  Min F1:           {args.min_f1}")
    print(f"  Max Miss Rate:    {args.max_miss_rate}")
    print(f"  Max FAR:          {args.max_far}")
    
    print("\n" + "=" * 100)
    
    # Final verdict
    if passed == total:
        print("\n✅ ALL FOLDS PASSED VERIFICATION")
    elif failed == 0 and missing == 0:
        print(f"\n⚠️  {passed}/{total} FOLDS PASSED, {warnings} WITH WARNINGS")
    else:
        print(f"\n✗ {passed}/{total} FOLDS PASSED, {failed} FAILED, {missing} MISSING")
    
    print("=" * 100)


def save_report(results: List[FoldResult], output_path: Path, args: argparse.Namespace):
    """Save detailed report to JSON."""
    report = {
        'verification_timestamp': None,  # Will be filled by caller
        'results_dir': str(args.results_dir),
        'thresholds': {
            'min_auc': args.min_auc,
            'min_f1': args.min_f1,
            'max_miss_rate': args.max_miss_rate,
            'max_far': args.max_far,
            'min_best_auc': args.min_best_auc
        },
        'summary': {
            'total_folds': len(results),
            'passed': sum(1 for r in results if r.status == Status.PASS),
            'warnings': sum(1 for r in results if r.status == Status.WARNING),
            'failed': sum(1 for r in results if r.status == Status.FAIL),
            'missing': sum(1 for r in results if r.status == Status.MISSING)
        },
        'folds': []
    }
    
    for r in results:
        report['folds'].append({
            'fold_id': r.fold_id,
            'status': r.status.value,
            'checkpoint_exists': r.checkpoint_exists,
            'summary_exists': r.summary_exists,
            'checkpoint_loadable': r.checkpoint_loadable,
            'best_auc': r.best_auc,
            'test_metrics': r.test_metrics,
            'issues': r.issues
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {output_path}")


def main():
    """Main verification function."""
    args = parse_args()
    
    # Resolve paths
    results_dir = Path(args.results_dir).resolve()
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else results_dir / 'checkpoints'
    logs_dir = Path(args.logs_dir) if args.logs_dir else results_dir / 'logs'
    
    print("=" * 100)
    print("Windows RTX 4080 Training Results Verification")
    print("=" * 100)
    print(f"\nResults directory:  {results_dir}")
    print(f"Checkpoints directory: {checkpoints_dir}")
    print(f"Logs directory:     {logs_dir}")
    
    if not results_dir.exists():
        print(f"\n❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Get list of folds
    if args.folds:
        folds = [f.strip() for f in args.folds.split(',')]
    else:
        folds = find_all_folds(logs_dir)
    
    if not folds:
        print("\n❌ Error: No folds found to verify")
        print("  Expected: logs/fold_*_summary.json files")
        sys.exit(1)
    
    print(f"\nFound {len(folds)} folds to verify: {', '.join(folds)}")
    
    # Verify each fold
    print("\nVerifying folds...")
    results = []
    for fold_id in folds:
        result = verify_fold(fold_id, checkpoints_dir, logs_dir, args)
        results.append(result)
    
    # Print results
    print_summary_table(results, args)
    print_statistics(results, args)
    
    # Save report if requested
    if args.output:
        from datetime import datetime
        output_path = Path(args.output)
        report_data = {
            'verification_timestamp': datetime.now().isoformat(),
            'results_dir': str(results_dir),
            'thresholds': {
                'min_auc': args.min_auc,
                'min_f1': args.min_f1,
                'max_miss_rate': args.max_miss_rate,
                'max_far': args.max_far,
                'min_best_auc': args.min_best_auc
            },
            'summary': {
                'total_folds': len(results),
                'passed': sum(1 for r in results if r.status == Status.PASS),
                'warnings': sum(1 for r in results if r.status == Status.WARNING),
                'failed': sum(1 for r in results if r.status == Status.FAIL),
                'missing': sum(1 for r in results if r.status == Status.MISSING)
            },
            'folds': [
                {
                    'fold_id': r.fold_id,
                    'status': r.status.value,
                    'checkpoint_exists': r.checkpoint_exists,
                    'summary_exists': r.summary_exists,
                    'checkpoint_loadable': r.checkpoint_loadable,
                    'best_auc': r.best_auc,
                    'test_metrics': r.test_metrics,
                    'issues': r.issues
                }
                for r in results
            ]
        }
        save_report(results, output_path, args)
    
    # Exit with appropriate code
    failed = sum(1 for r in results if r.status in (Status.FAIL, Status.MISSING))
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
