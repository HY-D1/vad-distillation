#!/usr/bin/env python3
"""
Compare Windows 4080 and Mac verification results.

Generates a comparison report showing metrics from both platforms
and highlighting any differences.

Usage:
    python scripts/compare_verification.py \
        --windows-dir outputs/production_4080/logs/ \
        --mac-dir outputs/production_4080/verification/ \
        --output outputs/production_4080/verification/comparison_report.md

    # With specific folds
    python scripts/compare_verification.py \
        --windows-dir outputs/production_4080/logs/ \
        --mac-dir outputs/production_4080/verification/ \
        --folds F01,F02,F03 \
        --output comparison_report.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Windows 4080 and Mac verification results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all folds
  python scripts/compare_verification.py \\
      --windows-dir outputs/production_4080/logs/ \\
      --mac-dir outputs/production_4080/verification/ \\
      --output comparison_report.md

  # Compare specific folds
  python scripts/compare_verification.py \\
      --windows-dir outputs/production_4080/logs/ \\
      --mac-dir outputs/production_4080/verification/ \\
      --folds F01,F02,F03 \\
      --output comparison_report.md
        """
    )

    parser.add_argument(
        '--windows-dir',
        type=str,
        required=True,
        help='Directory with Windows 4080 logs/outputs'
    )
    parser.add_argument(
        '--mac-dir',
        type=str,
        required=True,
        help='Directory with Mac verification results'
    )
    parser.add_argument(
        '--folds',
        type=str,
        default=None,
        help='Comma-separated list of folds (default: auto-detect)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output markdown report path'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.001,
        help='Acceptable difference threshold (default: 0.001)'
    )

    return parser.parse_args()


def load_windows_metrics(windows_dir: Path, fold_id: str) -> Optional[Dict]:
    """Load Windows metrics from summary.json."""
    summary_path = windows_dir / f"fold_{fold_id}_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            data = json.load(f)
            return data.get('test_metrics', {})
    return None


def load_mac_metrics(mac_dir: Path, fold_id: str) -> Optional[Dict]:
    """Load Mac metrics from verification output."""
    metrics_path = mac_dir / f"fold_{fold_id}_mac_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            return data.get('metrics', {})
    return None


def compute_differences(windows_metrics: Dict, mac_metrics: Dict) -> Dict[str, float]:
    """Compute differences between Windows and Mac metrics."""
    differences = {}

    for key in ['auc', 'f1', 'accuracy', 'miss_rate', 'false_alarm_rate', 'precision', 'recall']:
        if key in windows_metrics and key in mac_metrics:
            diff = abs(mac_metrics[key] - windows_metrics[key])
            differences[key] = diff

    return differences


def get_all_folds(windows_dir: Path, mac_dir: Path) -> List[str]:
    """Auto-detect available folds."""
    folds = set()

    # Look for summary files in Windows dir
    for f in windows_dir.glob("fold_*_summary.json"):
        fold_id = f.stem.replace("fold_", "").replace("_summary", "")
        folds.add(fold_id)

    # Look for Mac metrics
    for f in mac_dir.glob("fold_*_mac_metrics.json"):
        fold_id = f.stem.replace("fold_", "").replace("_mac_metrics", "")
        folds.add(fold_id)

    return sorted(list(folds))


def generate_comparison_table(comparisons: Dict[str, Dict]) -> str:
    """Generate markdown comparison table."""
    lines = []

    # Header
    lines.append("| Fold | Metric | Windows | Mac | Diff | Status |")
    lines.append("|------|--------|---------|-----|------|--------|")

    # Rows
    for fold_id in sorted(comparisons.keys()):
        comp = comparisons[fold_id]
        windows_metrics = comp['windows']
        mac_metrics = comp['mac']
        differences = comp['differences']

        first_row = True
        for metric in ['auc', 'f1', 'accuracy', 'miss_rate', 'false_alarm_rate']:
            if metric in differences:
                win_val = windows_metrics.get(metric, 0)
                mac_val = mac_metrics.get(metric, 0)
                diff = differences[metric]
                status = '✓' if diff < 0.001 else '⚠'

                fold_cell = fold_id if first_row else ""
                lines.append(f"| {fold_cell} | {metric.upper()} | {win_val:.4f} | {mac_val:.4f} | {diff:.6f} | {status} |")
                first_row = False

    return '\n'.join(lines)


def generate_summary_stats(comparisons: Dict[str, Dict], tolerance: float) -> str:
    """Generate summary statistics."""
    lines = []

    all_diffs = []
    failed_checks = []

    for fold_id, comp in comparisons.items():
        for metric, diff in comp['differences'].items():
            all_diffs.append(diff)
            if diff >= tolerance:
                failed_checks.append((fold_id, metric, diff))

    if all_diffs:
        avg_diff = sum(all_diffs) / len(all_diffs)
        max_diff = max(all_diffs)
        min_diff = min(all_diffs)

        lines.append(f"- **Total Comparisons:** {len(all_diffs)}")
        lines.append(f"- **Average Difference:** {avg_diff:.6f}")
        lines.append(f"- **Maximum Difference:** {max_diff:.6f}")
        lines.append(f"- **Minimum Difference:** {min_diff:.6f}")
        lines.append(f"- **Failed Checks (>={tolerance}):** {len(failed_checks)}")

        if failed_checks:
            lines.append("\n**Failed Checks:**")
            for fold_id, metric, diff in failed_checks:
                lines.append(f"- {fold_id} {metric}: {diff:.6f}")

    return '\n'.join(lines)


def generate_report(args: argparse.Namespace, comparisons: Dict[str, Dict]) -> str:
    """Generate full markdown report."""
    lines = []

    # Title
    lines.append("# Windows 4080 → Mac Verification Report")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Tolerance:** ±{args.tolerance}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(generate_summary_stats(comparisons, args.tolerance))
    lines.append("")

    # Detailed Comparison Table
    lines.append("## Detailed Comparison")
    lines.append("")
    lines.append(generate_comparison_table(comparisons))
    lines.append("")

    # Per-Fold Details
    lines.append("## Per-Fold Details")
    lines.append("")

    for fold_id in sorted(comparisons.keys()):
        comp = comparisons[fold_id]
        lines.append(f"### Fold {fold_id}")
        lines.append("")

        if comp['windows'] and comp['mac']:
            lines.append("| Metric | Windows | Mac | Difference | Status |")
            lines.append("|--------|---------|-----|------------|--------|")

            for metric in ['auc', 'f1', 'accuracy', 'miss_rate', 'false_alarm_rate', 'precision', 'recall']:
                if metric in comp['windows'] and metric in comp['mac']:
                    win_val = comp['windows'][metric]
                    mac_val = comp['mac'][metric]
                    diff = abs(win_val - mac_val)
                    status = '✓ PASS' if diff < args.tolerance else '✗ FAIL'

                    lines.append(f"| {metric.upper()} | {win_val:.4f} | {mac_val:.4f} | {diff:.6f} | {status} |")

            lines.append("")
        else:
            if not comp['windows']:
                lines.append("⚠️ Windows metrics not found")
            if not comp['mac']:
                lines.append("⚠️ Mac metrics not found")
            lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")

    total_folds = len(comparisons)
    passed_folds = sum(
        1 for comp in comparisons.values()
        if all(diff < args.tolerance for diff in comp['differences'].values())
    )

    if passed_folds == total_folds:
        lines.append(f"✅ **All {total_folds} folds passed verification!**")
        lines.append("")
        lines.append("All metrics are within the acceptable tolerance (±{:.3f}), ".format(args.tolerance))
        lines.append("confirming that the Windows 4080 training outputs are reproducible on Mac.")
    else:
        lines.append(f"⚠️ **{passed_folds}/{total_folds} folds passed verification**")
        lines.append("")
        lines.append("Some metrics differ by more than the acceptable tolerance.")
        lines.append("Please review the detailed comparison above.")

    lines.append("")

    return '\n'.join(lines)


def generate_csv_report(args: argparse.Namespace, comparisons: Dict[str, Dict]) -> str:
    """Generate CSV format report."""
    rows = []

    for fold_id in sorted(comparisons.keys()):
        comp = comparisons[fold_id]
        windows_metrics = comp['windows']
        mac_metrics = comp['mac']
        differences = comp['differences']

        for metric in ['auc', 'f1', 'accuracy', 'miss_rate', 'false_alarm_rate']:
            if metric in differences:
                rows.append({
                    'fold': fold_id,
                    'metric': metric,
                    'windows': windows_metrics.get(metric, 0),
                    'mac': mac_metrics.get(metric, 0),
                    'difference': differences[metric],
                    'status': 'PASS' if differences[metric] < args.tolerance else 'FAIL'
                })

    df = pd.DataFrame(rows)
    return df


def main():
    """Main comparison function."""
    args = parse_args()

    print("=" * 60)
    print("Windows 4080 vs Mac Verification Comparison")
    print("=" * 60)

    windows_dir = Path(args.windows_dir)
    mac_dir = Path(args.mac_dir)

    if not windows_dir.exists():
        print(f"Error: Windows directory not found: {windows_dir}")
        sys.exit(1)

    if not mac_dir.exists():
        print(f"Error: Mac directory not found: {mac_dir}")
        sys.exit(1)

    # Get folds
    if args.folds:
        folds = args.folds.split(',')
    else:
        folds = get_all_folds(windows_dir, mac_dir)

    print(f"\nComparing {len(folds)} folds: {', '.join(folds)}")

    # Load and compare metrics
    comparisons = {}
    for fold_id in folds:
        print(f"\nLoading fold {fold_id}...")

        windows_metrics = load_windows_metrics(windows_dir, fold_id)
        mac_metrics = load_mac_metrics(mac_dir, fold_id)

        if windows_metrics and mac_metrics:
            differences = compute_differences(windows_metrics, mac_metrics)
            comparisons[fold_id] = {
                'windows': windows_metrics,
                'mac': mac_metrics,
                'differences': differences
            }
            print(f"  Windows AUC: {windows_metrics.get('auc', 0):.4f}")
            print(f"  Mac AUC: {mac_metrics.get('auc', 0):.4f}")
            print(f"  Difference: {differences.get('auc', 0):.6f}")
        else:
            if not windows_metrics:
                print(f"  ⚠️ Windows metrics not found")
            if not mac_metrics:
                print(f"  ⚠️ Mac metrics not found")
            comparisons[fold_id] = {
                'windows': windows_metrics or {},
                'mac': mac_metrics or {},
                'differences': {}
            }

    # Generate report
    print("\nGenerating report...")
    report = generate_report(args, comparisons)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"  Saved: {output_path}")

    # Also save CSV
    csv_path = output_path.with_suffix('.csv')
    csv_df = generate_csv_report(args, comparisons)
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_folds = len(comparisons)
    passed_folds = sum(
        1 for comp in comparisons.values()
        if comp['differences'] and all(diff < args.tolerance for diff in comp['differences'].values())
    )

    print(f"Folds compared: {total_folds}")
    print(f"Folds passed: {passed_folds}")
    print(f"Folds failed: {total_folds - passed_folds}")

    if passed_folds == total_folds:
        print("\n✅ All verifications passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some verifications failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
