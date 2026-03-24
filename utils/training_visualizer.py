"""
Training visualization utilities for VAD distillation.

Provides plotting functions for:
- Training curves (loss, metrics over epochs)
- Prediction visualizations (student vs teacher)
- Comparative analysis (multiple folds, hyperparameters)
- Export to publication-ready formats
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Matplotlib imports with backend selection for headless environments
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend by default
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

# Plotly for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TrainingVisualizer:
    """
    Visualization toolkit for VAD training analysis.

    Supports both matplotlib (static) and plotly (interactive) backends.
    """

    def __init__(self, output_dir: str = "analysis/training_viz", style: str = "default"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
            style: Matplotlib style ('default', 'seaborn', 'ggplot')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        if MATPLOTLIB_AVAILABLE:
            if style == "seaborn":
                plt.style.use('seaborn-v0_8-darkgrid')
            elif style == "ggplot":
                plt.style.use('ggplot')

    def plot_training_curves(
        self,
        log_path: Union[str, Path],
        output_name: Optional[str] = None,
        save_format: str = "png"
    ) -> Path:
        """
        Plot comprehensive training curves from a training log.

        Args:
            log_path: Path to training log CSV
            output_name: Name for output file (default: derived from log name)
            save_format: 'png', 'pdf', 'svg'

        Returns:
            Path to saved plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib is required for plotting")

        import pandas as pd

        # Load data
        df = pd.read_csv(log_path)

        if output_name is None:
            output_name = Path(log_path).stem.replace("fold_", "training_")

        output_path = self.output_dir / f"{output_name}.{save_format}"

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Training Curves: {Path(log_path).stem}", fontsize=16, fontweight='bold')

        # 1. Loss curves
        ax = axes[0, 0]
        ax.plot(df['epoch'], df['train_loss'], label='Total Loss', linewidth=2, color='#1f77b4')
        ax.plot(df['epoch'], df['train_hard_loss'], label='Hard Loss', linewidth=1.5, linestyle='--', color='#ff7f0e')
        ax.plot(df['epoch'], df['train_soft_loss'], label='Soft Loss', linewidth=1.5, linestyle='--', color='#2ca02c')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Validation AUC
        ax = axes[0, 1]
        ax.plot(df['epoch'], df['val_auc'], label='Val AUC', linewidth=2, color='#9467bd')
        if 'val_auc' in df.columns:
            best_epoch = df.loc[df['val_auc'].idxmax(), 'epoch']
            best_auc = df['val_auc'].max()
            ax.axhline(y=best_auc, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_auc:.4f}')
            ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('Validation AUC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 3. F1 Score
        ax = axes[0, 2]
        ax.plot(df['epoch'], df['val_f1'], label='F1', linewidth=2, color='#d62728')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Validation F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 4. Error rates
        ax = axes[1, 0]
        ax.plot(df['epoch'], df['val_miss_rate'], label='Miss Rate', linewidth=2, color='#e377c2')
        ax.plot(df['epoch'], df['val_false_alarm_rate'], label='False Alarm Rate', linewidth=2, color='#7f7f7f')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rate')
        ax.set_title('Error Rates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 5. Learning rate
        ax = axes[1, 1]
        ax.plot(df['epoch'], df['learning_rate'], linewidth=2, color='#bcbd22')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 6. Epoch time (if available)
        ax = axes[1, 2]
        if 'time' in df.columns:
            ax.plot(df['epoch'], df['time'], linewidth=2, color='#17becf')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Epoch Duration')
        else:
            ax.text(0.5, 0.5, 'No timing data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Epoch Duration (N/A)')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to: {output_path}")
        return output_path

    def compare_folds(
        self,
        log_paths: List[Union[str, Path]],
        fold_names: Optional[List[str]] = None,
        output_name: str = "fold_comparison",
        save_format: str = "png"
    ) -> Path:
        """
        Compare training curves across multiple folds.

        Args:
            log_paths: List of paths to training logs
            fold_names: Names for each fold (default: extracted from paths)
            output_name: Output file name
            save_format: Image format

        Returns:
            Path to saved plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib is required for plotting")

        import pandas as pd

        if fold_names is None:
            fold_names = [Path(p).stem.replace("fold_", "") for p in log_paths]

        output_path = self.output_dir / f"{output_name}.{save_format}"

        # Load all data
        dfs = []
        for path in log_paths:
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")

        if not dfs:
            raise ValueError("No valid log files found")

        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Cross-Fold Comparison", fontsize=16, fontweight='bold')

        colors = plt.cm.tab10(np.linspace(0, 1, len(dfs)))

        # 1. Val AUC comparison
        ax = axes[0, 0]
        for df, name, color in zip(dfs, fold_names, colors):
            ax.plot(df['epoch'], df['val_auc'], label=name, linewidth=2, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val AUC')
        ax.set_title('Validation AUC Across Folds')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 2. Train loss comparison
        ax = axes[0, 1]
        for df, name, color in zip(dfs, fold_names, colors):
            ax.plot(df['epoch'], df['train_loss'], label=name, linewidth=2, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train Loss')
        ax.set_title('Training Loss Across Folds')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Final metrics bar chart
        ax = axes[1, 0]
        metrics = ['val_auc', 'val_f1', 'val_accuracy']
        final_values = {m: [] for m in metrics}

        for df in dfs:
            for m in metrics:
                if m in df.columns:
                    final_values[m].append(df[m].iloc[-1] if not df.empty else 0)
                else:
                    final_values[m].append(0)

        x = np.arange(len(fold_names))
        width = 0.25

        for i, (metric, values) in enumerate(final_values.items()):
            ax.bar(x + i * width, values, width, label=metric.replace('val_', ''))

        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title('Final Metrics by Fold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(fold_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

        # 4. Summary statistics table
        ax = axes[1, 1]
        ax.axis('off')

        # Compute statistics
        summary_data = []
        for metric in ['val_auc', 'val_f1', 'val_miss_rate']:
            values = [df[metric].max() if metric in df.columns else 0 for df in dfs]
            summary_data.append([
                metric.replace('val_', '').upper(),
                f"{np.mean(values):.4f}",
                f"{np.std(values):.4f}",
                f"{np.min(values):.4f}",
                f"{np.max(values):.4f}"
            ])

        table = ax.table(
            cellText=summary_data,
            colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
            loc='center',
            cellLoc='center',
            colColours=['#4472C4'] * 5,
            colWidths=[0.25, 0.2, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Cross-Fold Statistics', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Fold comparison saved to: {output_path}")
        return output_path

    def plot_predictions(
        self,
        predictions_path: Union[str, Path],
        output_name: Optional[str] = None,
        num_samples: int = 5,
        save_format: str = "png"
    ) -> Path:
        """
        Visualize model predictions on test samples.

        Args:
            predictions_path: Path to predictions .npz file
            output_name: Output file name
            num_samples: Number of samples to plot
            save_format: Image format

        Returns:
            Path to saved plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib is required for plotting")

        # Load predictions
        data = np.load(predictions_path)
        predictions = data['predictions']
        labels = data['labels']
        probs = data['probs']
        utt_ids = data.get('utt_ids', np.arange(len(predictions)))

        if output_name is None:
            output_name = Path(predictions_path).stem + "_viz"

        output_path = self.output_dir / f"{output_name}.{save_format}"

        # Find unique utterances
        unique_utts = np.unique(utt_ids)[:num_samples]

        # Create subplots
        n_rows = (len(unique_utts) + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        fig.suptitle("Model Predictions on Test Samples", fontsize=16, fontweight='bold')

        for idx, utt_id in enumerate(unique_utts):
            ax = axes[idx]

            # Get data for this utterance
            mask = utt_ids == utt_id
            utt_labels = labels[mask]
            utt_probs = probs[mask]
            utt_preds = predictions[mask]

            # Plot
            time_axis = np.arange(len(utt_labels))

            # Ground truth
            ax.fill_between(time_axis, 0, utt_labels, alpha=0.3, color='green', label='Ground Truth')

            # Predictions
            ax.plot(time_axis, utt_probs, linewidth=2, color='blue', label='Probability')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')

            # Mark correct/incorrect predictions
            correct = (utt_preds == utt_labels)
            ax.scatter(time_axis[correct], utt_probs[correct], c='green', s=10, alpha=0.5, label='Correct')
            ax.scatter(time_axis[~correct], utt_probs[~correct], c='red', s=20, marker='x', label='Incorrect')

            ax.set_xlabel('Frame')
            ax.set_ylabel('Probability')
            ax.set_title(f'Sample: {utt_id}')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(unique_utts), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Prediction visualization saved to: {output_path}")
        return output_path

    def create_training_report(
        self,
        log_path: Union[str, Path],
        predictions_path: Optional[Union[str, Path]] = None,
        summary_path: Optional[Union[str, Path]] = None,
        output_name: Optional[str] = None
    ) -> Path:
        """
        Create a comprehensive training report with multiple visualizations.

        Args:
            log_path: Path to training log
            predictions_path: Path to predictions .npz file (optional)
            summary_path: Path to summary JSON (optional)
            output_name: Output directory name

        Returns:
            Path to report directory
        """
        if output_name is None:
            output_name = Path(log_path).stem.replace("fold_", "report_")

        report_dir = self.output_dir / output_name
        report_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating training report in: {report_dir}")

        # 1. Training curves
        print("  Creating training curves...")
        curves_path = self.plot_training_curves(
            log_path,
            output_name="training_curves",
            save_format="png"
        )
        # Also save as PDF for publication
        self.plot_training_curves(
            log_path,
            output_name="training_curves",
            save_format="pdf"
        )

        # 2. Predictions visualization
        if predictions_path and Path(predictions_path).exists():
            print("  Creating prediction visualizations...")
            self.plot_predictions(
                predictions_path,
                output_name="predictions",
                save_format="png"
            )

        # 3. Summary text file
        if summary_path and Path(summary_path).exists():
            with open(summary_path) as f:
                summary = json.load(f)

            summary_txt = report_dir / "summary.txt"
            with open(summary_txt, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("Training Summary\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Fold ID: {summary.get('fold_id', 'N/A')}\n")
                f.write(f"Train Speakers: {summary.get('train_speakers', [])}\n")
                f.write(f"Val Speaker: {summary.get('val_speaker', 'N/A')}\n")
                f.write(f"Test Speaker: {summary.get('test_speaker', 'N/A')}\n\n")
                f.write(f"Model Parameters: {summary.get('num_parameters', 0):,}\n")
                f.write(f"Model Size: {summary.get('model_size_mb', 0):.2f} MB\n\n")
                f.write(f"Best Val AUC: {summary.get('best_val_auc', 0):.4f}\n")

                test_metrics = summary.get('test_metrics', {})
                f.write("\nTest Metrics:\n")
                f.write(f"  AUC: {test_metrics.get('auc', 0):.4f}\n")
                f.write(f"  F1: {test_metrics.get('f1', 0):.4f}\n")
                f.write(f"  Miss Rate: {test_metrics.get('miss_rate', 0):.4f}\n")
                f.write(f"  False Alarm Rate: {test_metrics.get('false_alarm_rate', 0):.4f}\n")
                f.write(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}\n")

            print(f"  Summary saved to: {summary_txt}")

        print(f"\nReport complete: {report_dir}")
        return report_dir


def create_live_plotter(update_interval: int = 5):
    """
    Create a live plotting callback for use during training.

    This is a factory function that returns a callback to be called each epoch.

    Args:
        update_interval: Update plot every N epochs

    Returns:
        Callback function(epoch, metrics)
    """
    if not MATPLOTLIB_AVAILABLE:
        return lambda *args: None

    history = defaultdict(list)

    def update_plot(epoch: int, metrics: Dict[str, float]):
        """Update the live plot with new metrics."""
        for key, value in metrics.items():
            history[key].append(value)

        if (epoch + 1) % update_interval != 0:
            return

        # Clear and redraw
        plt.clf()

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        if 'train_loss' in history:
            axes[0].plot(history.get('epoch', range(len(history['train_loss']))),
                        history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history.get('epoch', range(len(history['val_loss']))),
                        history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Metric plot
        if 'val_auc' in history:
            axes[1].plot(history.get('epoch', range(len(history['val_auc']))),
                        history['val_auc'], label='Val AUC', linewidth=2, color='green')
            axes[1].set_ylim([0, 1])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Validation Performance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.1)

    return update_plot


# For backwards compatibility
TrainingPlotter = TrainingVisualizer


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example: Create visualizer and demonstrate functionality
    viz = TrainingVisualizer(output_dir="test_viz")

    # Create dummy training log
    import pandas as pd

    epochs = 20
    dummy_data = {
        'epoch': range(1, epochs + 1),
        'train_loss': [1.0 - 0.04 * i + np.random.rand() * 0.05 for i in range(epochs)],
        'train_hard_loss': [0.6 - 0.02 * i + np.random.rand() * 0.03 for i in range(epochs)],
        'train_soft_loss': [0.4 - 0.02 * i + np.random.rand() * 0.03 for i in range(epochs)],
        'val_auc': [0.5 + 0.02 * i + np.random.rand() * 0.02 for i in range(epochs)],
        'val_f1': [0.45 + 0.018 * i + np.random.rand() * 0.02 for i in range(epochs)],
        'val_miss_rate': [0.5 - 0.015 * i + np.random.rand() * 0.03 for i in range(epochs)],
        'val_false_alarm_rate': [0.4 - 0.01 * i + np.random.rand() * 0.02 for i in range(epochs)],
        'val_accuracy': [0.6 + 0.015 * i + np.random.rand() * 0.02 for i in range(epochs)],
        'learning_rate': [0.001 * (0.9 ** i) for i in range(epochs)],
        'time': [30 + np.random.rand() * 10 for i in range(epochs)]
    }

    # Clip to valid ranges
    dummy_data['val_auc'] = np.clip(dummy_data['val_auc'], 0, 1)
    dummy_data['val_f1'] = np.clip(dummy_data['val_f1'], 0, 1)

    df = pd.DataFrame(dummy_data)
    test_log = "test_viz/test_training_log.csv"
    df.to_csv(test_log, index=False)

    print("Created test training log")

    # Generate plots
    print("\nGenerating training curves...")
    viz.plot_training_curves(test_log)

    # Create dummy predictions
    n_samples = 1000
    dummy_preds = {
        'predictions': np.random.randint(0, 2, n_samples),
        'labels': np.random.randint(0, 2, n_samples),
        'probs': np.clip(np.random.rand(n_samples) + np.random.randint(0, 2, n_samples) * 0.3, 0, 1),
        'utt_ids': np.repeat(['utt_001', 'utt_002', 'utt_003', 'utt_004', 'utt_005'], 200)
    }

    test_preds = "test_viz/test_predictions.npz"
    np.savez(test_preds, **dummy_preds)

    print("\nGenerating prediction visualization...")
    viz.plot_predictions(test_preds, num_samples=3)

    # Create multiple logs for fold comparison
    log_paths = []
    for fold in ['F01', 'F02', 'M01']:
        fold_data = dummy_data.copy()
        fold_data['val_auc'] = [0.5 + 0.02 * i + np.random.rand() * 0.05 for i in range(epochs)]
        fold_data['val_auc'] = np.clip(fold_data['val_auc'], 0, 1)
        fold_df = pd.DataFrame(fold_data)
        fold_log = f"test_viz/fold_{fold}_log.csv"
        fold_df.to_csv(fold_log, index=False)
        log_paths.append(fold_log)

    print("\nGenerating fold comparison...")
    viz.compare_folds(log_paths, fold_names=['F01', 'F02', 'M01'])

    print("\nAll visualizations created in: test_viz/")
