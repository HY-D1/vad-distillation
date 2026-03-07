#!/usr/bin/env python3
"""
Compare Mac MPS vs Windows CUDA training results.

Loads metrics from both platforms and checks they are within tolerance.
Useful for verifying that Windows RTX 4080 results are reproducible on Mac.

Usage:
    # Compare all folds
    python scripts/compare_platforms.py \
        --mac-dir outputs/production/ \
        --cuda-dir outputs/production_cuda/
    
    # Compare specific folds
    python scripts/compare_platforms.py \
        --mac-dir outputs/production/ \
        --cuda-dir outputs/production_cuda/ \
        --folds F01,F02,F03
    
    # Custom tolerance
    python scripts/compare_platforms.py \
        --mac-dir outputs/production/ \
        --cuda-dir outputs/production_cuda/ \
        --tolerance 0.002
    
    # Save comparison CSV
    python scripts/compare_platforms.py \
        --mac-dir outputs/production/ \
        --cuda-dir outputs/production_cuda/ \
        --output platform_comparison.csv

Expected directory structure:
    outputs/production/          # Mac MPS results
    ├── logs/
    │   ├── fold_F01_summary.json
    │   └── ...
    └── checkpoints/
        └── ...
    
    outputs/production_cuda/     # Windows CUDA results
    ├── logs/
    │   ├── fold_F01_summary.json
    │   └── ...
    └── checkpoints/
        └── ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class ComparisonStatus(Enum):
    """Comparison status between platforms."""
    MATCH = "MATCH"
    TOLERANCE = "TOLERANCE"  # Within tolerance but not exact
    DIFFERENT = "DIFFERENT"  # Beyond tolerance
    MISSING_MAC = "MISSING_MAC"
    MISSING_CUDA = "MISSING_CUDA"


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""
    metric_name: str
    mac_value: Optional[float]
    cuda_value: Optional[float]
    difference: Optional[float]
    status: ComparisonStatus


@dataclass
class FoldComparison:
    """Comparison result for a single fold."""
    fold_id: str
    metrics: List[MetricComparison] = field(default_factory=list)
    overall_status: ComparisonStatus = ComparisonStatus.MATCH
    
    def get_metric(self, name: str) -> Optional[MetricComparison]:
        for m in self.metrics:
            if m.metric_name == name:
                return m
        return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Mac MPS vs Windows CUDA results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all folds
  python scripts/compare_platforms.py \\
      --mac-dir outputs/production/ \\
      --cuda-dir outputs/production_cuda/

  # Compare specific folds with custom tolerance
  python scripts/compare_platforms.py \\
      --mac-dir outputs/production/ \\
      --cuda-dir outputs/production_cuda/ \\
      --folds F01,F02,F03 \\
      --tolerance 0.002

  # Save CSV report
  python scripts/compare_platforms.py \\
      --mac-dir outputs/production/ \\
      --cuda-dir outputs/production_cuda/ \\
      --output platform_comparison.csv
        """
    )

    parser.add_argument(
        '--mac-dir',
        type=str,
        required=True,
        help='Directory containing Mac MPS results'
    )
    parser.add_argument(
        '--cuda-dir',
        type=str,
        required=True,
        help='Directory containing Windows CUDA results'
    )
    parser.add_argument(
        '--mac-logs-dir',
        type=str,
        default=None,
        help='Mac logs subdirectory (default: {mac_dir}/logs/)'
    )
    parser.add_argument(
        '--cuda-logs-dir',
        type=str,
        default=None,
        help='CUDA logs subdirectory (default: {cuda_dir}/logs/)'
    )
    parser.add_argument(
        '--folds',
        type=str,
        default=None,
        help='Comma-separated list of fold IDs (default: auto-detect)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.001,
        help='Acceptable difference threshold (default: 0.001)'
    )
    parser.add_argument(
        '--warning-threshold',
        type=float,
        default=0.01,
        help='Threshold for reporting as potential issue (default: 0.01)'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        default='auc,f1,miss_rate,false_alarm_rate,accuracy',
        help='Comma-separated metrics to compare (default: auc,f1,miss_rate,false_alarm_rate,accuracy)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path for detailed comparison'
    )
    parser.add_argument(
        '--json-output',
        type=str,
        default=None,
        help='Output JSON file path for detailed comparison'
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


def load_metrics(logs_dir: Path, fold_id: str) -> Optional[Dict[str, float]]:
    """Load test metrics from summary.json."""
    summary_path = logs_dir / f"fold_{fold_id}_summary.json"
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
            return data.get('test_metrics', {})
    except (json.JSONDecodeError, IOError):
        return None


def compare_metric(
    metric_name: str,
    mac_value: Optional[float],
    cuda_value: Optional[float],
    tolerance: float
) -> MetricComparison:
    """Compare a single metric between platforms."""
    if mac_value is None:
        return MetricComparison(
            metric_name=metric_name,
            mac_value=None,
            cuda_value=cuda_value,
            difference=None,
            status=ComparisonStatus.MISSING_MAC
        )
    
    if cuda_value is None:
        return MetricComparison(
            metric_name=metric_name,
            mac_value=mac_value,
            cuda_value=None,
            difference=None,
            status=ComparisonStatus.MISSING_CUDA
        )
    
    difference = abs(mac_value - cuda_value)
    
    if difference < 1e-10:
        status = ComparisonStatus.MATCH
    elif difference < tolerance:
        status = ComparisonStatus.TOLERANCE
    else:
        status = ComparisonStatus.DIFFERENT
    
    return MetricComparison(
        metric_name=metric_name,
        mac_value=mac_value,
        cuda_value=cuda_value,
        difference=difference,
        status=status
    )


def compare_fold(
    fold_id: str,
    mac_logs_dir: Path,
    cuda_logs_dir: Path,
    metrics_to_compare: List[str],
    tolerance: float
) -> FoldComparison:
    """Compare all metrics for a single fold."""
    mac_metrics = load_metrics(mac_logs_dir, fold_id)
    cuda_metrics = load_metrics(cuda_logs_dir, fold_id)
    
    result = FoldComparison(fold_id=fold_id)
    
    for metric_name in metrics_to_compare:
        mac_val = mac_metrics.get(metric_name) if mac_metrics else None
        cuda_val = cuda_metrics.get(metric_name) if cuda_metrics else None
        
        comp = compare_metric(metric_name, mac_val, cuda_val, tolerance)
        result.metrics.append(comp)
    
    # Determine overall status
    statuses = [m.status for m in result.metrics]
    
    if ComparisonStatus.MISSING_MAC in statuses:
        result.overall_status = ComparisonStatus.MISSING_MAC
    elif ComparisonStatus.MISSING_CUDA in statuses:
        result.overall_status = ComparisonStatus.MISSING_CUDA
    elif ComparisonStatus.DIFFERENT in statuses:
        result.overall_status = ComparisonStatus.DIFFERENT
    elif ComparisonStatus.TOLERANCE in statuses:
        result.overall_status = ComparisonStatus.TOLERANCE
    else:
        result.overall_status = ComparisonStatus.MATCH
    
    return result


def print_comparison_table(
    comparisons: List[FoldComparison],
    metrics_to_compare: List[str],
    tolerance: float,
    warning_threshold: float
):
    """Print formatted comparison table."""
    print("\n" + "=" * 120)
    print("Mac MPS vs Windows CUDA Platform Comparison")
    print("=" * 120)
    print(f"\nTolerance: ±{tolerance} | Warning threshold: ±{warning_threshold}")
    
    # Print per-metric summary
    print("\n" + "-" * 120)
    print("Per-Metric Summary")
    print("-" * 120)
    
    for metric_name in metrics_to_compare:
        diffs = []
        max_diff = 0
        max_diff_fold = None
        
        for comp in comparisons:
            m = comp.get_metric(metric_name)
            if m and m.difference is not None:
                diffs.append(m.difference)
                if m.difference > max_diff:
                    max_diff = m.difference
                    max_diff_fold = comp.fold_id
        
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            out_of_tolerance = sum(1 for d in diffs if d >= tolerance)
            warning_count = sum(1 for d in diffs if d >= warning_threshold)
            
            status = "✓" if out_of_tolerance == 0 else "⚠" if warning_count == 0 else "✗"
            print(f"\n{status} {metric_name.upper()}:")
            print(f"    Average difference: {avg_diff:.6f}")
            print(f"    Maximum difference: {max_diff:.6f} (in {max_diff_fold})")
            print(f"    Out of tolerance:   {out_of_tolerance}/{len(diffs)}")
            if warning_count > 0:
                print(f"    ⚠️  Potential issues (>={warning_threshold}): {warning_count}")
    
    # Print per-fold details
    print("\n" + "-" * 120)
    print("Per-Fold Details")
    print("-" * 120)
    
    # Header
    header = f"{'Fold':<8} {'Status':<12}"
    for metric in metrics_to_compare:
        header += f" {metric[:4].upper():>8}"
    header += f" {'Max Diff':>10}"
    print(header)
    print("-" * 120)
    
    # Rows
    for comp in comparisons:
        status_str = {
            ComparisonStatus.MATCH: "✓ MATCH",
            ComparisonStatus.TOLERANCE: "≈ CLOSE",
            ComparisonStatus.DIFFERENT: "✗ DIFF",
            ComparisonStatus.MISSING_MAC: "✗ NO MAC",
            ComparisonStatus.MISSING_CUDA: "✗ NO CUDA"
        }.get(comp.overall_status, str(comp.overall_status))
        
        row = f"{comp.fold_id:<8} {status_str:<12}"
        
        max_diff = 0
        for metric in metrics_to_compare:
            m = comp.get_metric(metric)
            if m and m.difference is not None:
                row += f" {m.difference:>8.4f}"
                max_diff = max(max_diff, m.difference)
            else:
                row += f" {'N/A':>8}"
        
        row += f" {max_diff:>10.4f}"
        print(row)
    
    print("-" * 120)


def print_summary(
    comparisons: List[FoldComparison],
    tolerance: float,
    warning_threshold: float
):
    """Print overall summary statistics."""
    total = len(comparisons)
    matches = sum(1 for c in comparisons if c.overall_status == ComparisonStatus.MATCH)
    tolerance_ok = sum(1 for c in comparisons if c.overall_status == ComparisonStatus.TOLERANCE)
    different = sum(1 for c in comparisons if c.overall_status == ComparisonStatus.DIFFERENT)
    missing = sum(1 for c in comparisons if c.overall_status in 
                  (ComparisonStatus.MISSING_MAC, ComparisonStatus.MISSING_CUDA))
    
    # Count potential issues
    potential_issues = []
    for comp in comparisons:
        for m in comp.metrics:
            if m.difference is not None and m.difference >= warning_threshold:
                potential_issues.append({
                    'fold': comp.fold_id,
                    'metric': m.metric_name,
                    'difference': m.difference
                })
    
    print("\n" + "=" * 120)
    print("Summary")
    print("=" * 120)
    
    print(f"\nTotal folds compared:    {total}")
    print(f"  ✓ Exact matches:       {matches}")
    print(f"  ≈ Within tolerance:    {tolerance_ok}")
    print(f"  ✗ Different:           {different}")
    print(f"  ✗ Missing data:        {missing}")
    
    if potential_issues:
        print(f"\n⚠️  Potential Issues Found (≥{warning_threshold}):")
        for issue in sorted(potential_issues, key=lambda x: x['difference'], reverse=True)[:10]:
            print(f"    {issue['fold']} - {issue['metric']}: {issue['difference']:.6f}")
        if len(potential_issues) > 10:
            print(f"    ... and {len(potential_issues) - 10} more")
    
    print("\n" + "=" * 120)
    
    # Final verdict
    if matches + tolerance_ok == total:
        print("\n✅ ALL FOLDS MATCH WITHIN TOLERANCE")
    elif different == 0:
        print(f"\n⚠️  {matches + tolerance_ok}/{total} FOLDS MATCH, BUT SOME DATA MISSING")
    else:
        print(f"\n✗ {different}/{total} FOLDS HAVE SIGNIFICANT DIFFERENCES")
    
    print("=" * 120)


def save_csv_report(
    comparisons: List[FoldComparison],
    metrics_to_compare: List[str],
    output_path: Path,
    tolerance: float,
    warning_threshold: float
):
    """Save comparison report to CSV."""
    rows = []
    
    for comp in comparisons:
        for m in comp.metrics:
            rows.append({
                'fold': comp.fold_id,
                'metric': m.metric_name,
                'mac_value': m.mac_value,
                'cuda_value': m.cuda_value,
                'difference': m.difference,
                'status': m.status.value,
                'within_tolerance': m.status in (ComparisonStatus.MATCH, ComparisonStatus.TOLERANCE),
                'potential_issue': m.difference is not None and m.difference >= warning_threshold
            })
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
    else:
        # Manual CSV generation
        import csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"\n📄 CSV report saved to: {output_path}")
        return
    
    df.to_csv(output_path, index=False)
    print(f"\n📄 CSV report saved to: {output_path}")
    
    # Also print summary statistics
    print("\nCSV Report Summary:")
    print(f"  Total comparisons:    {len(df)}")
    print(f"  Within tolerance:     {df['within_tolerance'].sum()}")
    print(f"  Potential issues:     {df['potential_issue'].sum()}")
    
    if df['potential_issue'].sum() > 0:
        print("\n  Issues by metric:")
        for metric in df['metric'].unique():
            metric_df = df[df['metric'] == metric]
            issue_count = metric_df['potential_issue'].sum()
            if issue_count > 0:
                max_diff = metric_df[metric_df['potential_issue']]['difference'].max()
                print(f"    {metric}: {int(issue_count)} issues (max diff: {max_diff:.6f})")


def save_json_report(
    comparisons: List[FoldComparison],
    output_path: Path,
    args: argparse.Namespace
):
    """Save detailed comparison report to JSON."""
    from datetime import datetime
    
    report = {
        'comparison_timestamp': datetime.now().isoformat(),
        'mac_dir': str(args.mac_dir),
        'cuda_dir': str(args.cuda_dir),
        'tolerance': args.tolerance,
        'warning_threshold': args.warning_threshold,
        'summary': {
            'total_folds': len(comparisons),
            'matches': sum(1 for c in comparisons if c.overall_status == ComparisonStatus.MATCH),
            'within_tolerance': sum(1 for c in comparisons if c.overall_status == ComparisonStatus.TOLERANCE),
            'different': sum(1 for c in comparisons if c.overall_status == ComparisonStatus.DIFFERENT),
            'missing_data': sum(1 for c in comparisons if c.overall_status in 
                               (ComparisonStatus.MISSING_MAC, ComparisonStatus.MISSING_CUDA))
        },
        'folds': []
    }
    
    for comp in comparisons:
        fold_data = {
            'fold_id': comp.fold_id,
            'overall_status': comp.overall_status.value,
            'metrics': [
                {
                    'name': m.metric_name,
                    'mac_value': m.mac_value,
                    'cuda_value': m.cuda_value,
                    'difference': m.difference,
                    'status': m.status.value
                }
                for m in comp.metrics
            ]
        }
        report['folds'].append(fold_data)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 JSON report saved to: {output_path}")


def main():
    """Main comparison function."""
    args = parse_args()
    
    # Resolve paths
    mac_dir = Path(args.mac_dir).resolve()
    cuda_dir = Path(args.cuda_dir).resolve()
    mac_logs_dir = Path(args.mac_logs_dir) if args.mac_logs_dir else mac_dir / 'logs'
    cuda_logs_dir = Path(args.cuda_logs_dir) if args.cuda_logs_dir else cuda_dir / 'logs'
    
    # Parse metrics
    metrics_to_compare = [m.strip() for m in args.metrics.split(',')]
    
    print("=" * 120)
    print("Mac MPS vs Windows CUDA Platform Comparison")
    print("=" * 120)
    print(f"\nMac results directory:    {mac_dir}")
    print(f"CUDA results directory:   {cuda_dir}")
    print(f"Tolerance:                ±{args.tolerance}")
    print(f"Warning threshold:        ±{args.warning_threshold}")
    print(f"Metrics to compare:       {', '.join(metrics_to_compare)}")
    
    if not mac_dir.exists():
        print(f"\n❌ Error: Mac directory not found: {mac_dir}")
        sys.exit(1)
    
    if not cuda_dir.exists():
        print(f"\n❌ Error: CUDA directory not found: {cuda_dir}")
        sys.exit(1)
    
    # Get list of folds
    if args.folds:
        folds = [f.strip() for f in args.folds.split(',')]
    else:
        # Auto-detect from both directories
        mac_folds = set(find_all_folds(mac_logs_dir))
        cuda_folds = set(find_all_folds(cuda_logs_dir))
        folds = sorted(mac_folds | cuda_folds)
    
    if not folds:
        print("\n❌ Error: No folds found to compare")
        print("  Expected: logs/fold_*_summary.json files in both directories")
        sys.exit(1)
    
    print(f"\nComparing {len(folds)} folds: {', '.join(folds)}")
    
    # Compare each fold
    print("\nComparing folds...")
    comparisons = []
    for fold_id in folds:
        comp = compare_fold(fold_id, mac_logs_dir, cuda_logs_dir, metrics_to_compare, args.tolerance)
        comparisons.append(comp)
    
    # Print results
    print_comparison_table(comparisons, metrics_to_compare, args.tolerance, args.warning_threshold)
    print_summary(comparisons, args.tolerance, args.warning_threshold)
    
    # Save reports if requested
    if args.output:
        save_csv_report(comparisons, metrics_to_compare, Path(args.output), 
                       args.tolerance, args.warning_threshold)
    
    if args.json_output:
        save_json_report(comparisons, Path(args.json_output), args)
    
    # Exit with appropriate code
    different = sum(1 for c in comparisons if c.overall_status == ComparisonStatus.DIFFERENT)
    sys.exit(0 if different == 0 else 1)


if __name__ == '__main__':
    main()
