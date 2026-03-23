#!/usr/bin/env python3
"""
CLI tool for visualizing VAD training results.

Usage:
    # Visualize single fold training
    python scripts/analysis/visualize_training.py --log outputs/pilot/logs/fold_F01.csv

    # Compare multiple folds
    python scripts/analysis/visualize_training.py --compare-folds --logs outputs/*/logs/fold_*.csv

    # Create full report
    python scripts/analysis/visualize_training.py --report --log outputs/pilot/logs/fold_F01.csv

    # Generate prediction visualization
    python scripts/analysis/visualize_training.py --predictions outputs/pilot/logs/fold_F01_predictions.npz
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.training_visualizer import TrainingVisualizer
except ImportError as e:
    print(f"Error importing visualization module: {e}")
    print("Make sure you're running from the project root.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize VAD training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single fold visualization
  python scripts/analysis/visualize_training.py --log outputs/pilot/logs/fold_F01.csv

  # Compare all folds
  python scripts/analysis/visualize_training.py --compare-folds \\
      --logs outputs/production_cuda/logs/fold_*.csv \\
      --output-dir analysis/comparison/

  # Full training report
  python scripts/analysis/visualize_training.py --report \\
      --log outputs/pilot/logs/fold_F01.csv \\
      --predictions outputs/pilot/logs/fold_F01_predictions.npz \\
      --summary outputs/pilot/logs/fold_F01_summary.json

  # Prediction visualization only
  python scripts/analysis/visualize_training.py \\
      --predictions outputs/pilot/logs/fold_F01_predictions.npz \\
      --num-samples 10
        """
    )

    # Input options
    parser.add_argument(
        "--log", "-l",
        type=str,
        help="Path to training log CSV file"
    )
    parser.add_argument(
        "--logs",
        type=str,
        nargs="+",
        help="Multiple log files (for comparison)"
    )
    parser.add_argument(
        "--predictions", "-p",
        type=str,
        help="Path to predictions .npz file"
    )
    parser.add_argument(
        "--summary", "-s",
        type=str,
        help="Path to summary JSON file"
    )

    # Action options
    parser.add_argument(
        "--compare-folds",
        action="store_true",
        help="Compare multiple folds"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive training report"
    )

    # Visualization options
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (default: png)"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="default",
        choices=["default", "seaborn", "ggplot"],
        help="Plot style (default: default)"
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="analysis/training_viz",
        help="Output directory (default: analysis/training_viz)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output file/directory name (default: auto-generated)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not args.compare_folds and not args.log and not args.predictions:
        print("Error: Must specify --log, --predictions, or --compare-folds")
        print("Run with --help for usage information")
        sys.exit(1)

    # Create visualizer
    visualizer = TrainingVisualizer(
        output_dir=args.output_dir,
        style=args.style
    )

    print("=" * 70)
    print("VAD Training Visualization Tool")
    print("=" * 70)

    if args.compare_folds:
        # Compare multiple folds
        if not args.logs:
            # Try to auto-discover logs if not specified
            logs_dir = Path("outputs")
            if logs_dir.exists():
                args.logs = sorted(logs_dir.rglob("fold_*.csv"))
                if not args.logs:
                    print("Error: No log files found. Specify with --logs")
                    sys.exit(1)
                print(f"Auto-discovered {len(args.logs)} log files")
            else:
                print("Error: No log files specified and outputs/ not found")
                sys.exit(1)

        print(f"\nComparing {len(args.logs)} folds...")

        # Extract fold names from paths
        fold_names = [Path(p).stem.replace("fold_", "") for p in args.logs]

        output_name = args.output_name or "fold_comparison"
        visualizer.compare_folds(
            log_paths=args.logs,
            fold_names=fold_names,
            output_name=output_name,
            save_format=args.format
        )

    elif args.report:
        # Generate comprehensive report
        if not args.log:
            print("Error: --report requires --log")
            sys.exit(1)

        print(f"\nGenerating training report...")
        print(f"  Log: {args.log}")
        if args.predictions:
            print(f"  Predictions: {args.predictions}")
        if args.summary:
            print(f"  Summary: {args.summary}")

        report_dir = visualizer.create_training_report(
            log_path=args.log,
            predictions_path=args.predictions,
            summary_path=args.summary,
            output_name=args.output_name
        )

        print(f"\nReport generated: {report_dir}")

    elif args.predictions and not args.log:
        # Just visualize predictions
        print(f"\nVisualizing predictions: {args.predictions}")
        visualizer.plot_predictions(
            predictions_path=args.predictions,
            output_name=args.output_name,
            num_samples=args.num_samples,
            save_format=args.format
        )

    else:
        # Single fold training curves
        print(f"\nVisualizing training curves: {args.log}")
        visualizer.plot_training_curves(
            log_path=args.log,
            output_name=args.output_name,
            save_format=args.format
        )

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
