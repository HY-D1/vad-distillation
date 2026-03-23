"""
Enhanced metrics tracking and logging for VAD training visualization.

This module provides centralized metrics management with support for:
- Detailed per-batch and per-epoch metric tracking
- TensorBoard logging
- Gradient statistics monitoring
- Learning rate scheduling tracking
- Teacher-student agreement metrics
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class MetricsTracker:
    """
    Centralized metrics tracking for training visualization.

    Tracks:
    - Loss components (total, hard, soft)
    - Performance metrics (AUC, F1, miss rate, etc.)
    - Training dynamics (gradient norms, learning rates)
    - Teacher-student agreement
    """

    def __init__(
        self,
        log_dir: str,
        fold_id: str,
        use_tensorboard: bool = False,
        log_interval: int = 10
    ):
        """
        Initialize metrics tracker.

        Args:
            log_dir: Directory to save logs
            fold_id: Fold identifier for this training run
            use_tensorboard: Whether to use TensorBoard logging
            log_interval: Log every N batches
        """
        self.log_dir = Path(log_dir)
        self.fold_id = fold_id
        self.log_interval = log_interval

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage
        self.epoch_metrics = defaultdict(list)
        self.batch_metrics = defaultdict(list)
        self.current_epoch = 0

        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard" / f"fold_{fold_id}"
                self.writer = SummaryWriter(log_dir=str(tb_dir))
                print(f"TensorBoard logging to: {tb_dir}")
                print(f"Run: tensorboard --logdir={tb_dir.parent}")
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")

        # CSV logger for detailed batch-level metrics
        self.batch_log_path = self.log_dir / f"fold_{fold_id}_batch_metrics.csv"
        self._init_batch_log()

        # Gradient statistics storage
        self.grad_stats = defaultdict(list)

    def _init_batch_log(self):
        """Initialize batch-level CSV log."""
        headers = [
            'epoch', 'batch', 'batch_time', 'total_loss', 'hard_loss', 'soft_loss',
            'learning_rate', 'grad_norm', 'data_load_time'
        ]
        with open(self.batch_log_path, 'w', newline='') as f:
            f.write(','.join(headers) + '\n')

    def log_batch(
        self,
        epoch: int,
        batch: int,
        losses: Dict[str, float],
        learning_rate: float,
        grad_norm: Optional[float] = None,
        data_load_time: float = 0.0,
        batch_time: float = 0.0
    ):
        """
        Log metrics for a single batch.

        Args:
            epoch: Current epoch number
            batch: Batch index
            losses: Dictionary with 'total_loss', 'hard_loss', 'soft_loss'
            learning_rate: Current learning rate
            grad_norm: Gradient norm (if computed)
            data_load_time: Time spent loading data
            batch_time: Total batch processing time
        """
        metrics = {
            'epoch': epoch,
            'batch': batch,
            'batch_time': batch_time,
            'total_loss': losses.get('total_loss', 0.0),
            'hard_loss': losses.get('hard_loss', 0.0),
            'soft_loss': losses.get('soft_loss', 0.0),
            'learning_rate': learning_rate,
            'grad_norm': grad_norm if grad_norm is not None else 0.0,
            'data_load_time': data_load_time
        }

        # Append to batch log
        with open(self.batch_log_path, 'a', newline='') as f:
            values = [str(metrics.get(k, '')) for k in [
                'epoch', 'batch', 'batch_time', 'total_loss', 'hard_loss', 'soft_loss',
                'learning_rate', 'grad_norm', 'data_load_time'
            ]]
            f.write(','.join(values) + '\n')

        # Store for current epoch
        for key, value in metrics.items():
            self.batch_metrics[key].append(value)

        # TensorBoard logging (every N batches)
        if self.writer is not None and batch % self.log_interval == 0:
            global_step = epoch * 10000 + batch  # Approximate
            self.writer.add_scalar('Batch/TotalLoss', metrics['total_loss'], global_step)
            self.writer.add_scalar('Batch/HardLoss', metrics['hard_loss'], global_step)
            self.writer.add_scalar('Batch/SoftLoss', metrics['soft_loss'], global_step)
            self.writer.add_scalar('Batch/GradNorm', metrics['grad_norm'], global_step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        epoch_time: float
    ):
        """
        Log end-of-epoch metrics.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics (loss, etc.)
            val_metrics: Validation metrics (AUC, F1, etc.)
            learning_rate: Current learning rate
            epoch_time: Time for this epoch
        """
        self.current_epoch = epoch

        # Combine metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics.get('train_loss', 0.0),
            'train_hard_loss': train_metrics.get('train_hard_loss', 0.0),
            'train_soft_loss': train_metrics.get('train_soft_loss', 0.0),
            'val_auc': val_metrics.get('auc', 0.0),
            'val_f1': val_metrics.get('f1', 0.0),
            'val_miss_rate': val_metrics.get('miss_rate', 0.0),
            'val_false_alarm_rate': val_metrics.get('false_alarm_rate', 0.0),
            'val_accuracy': val_metrics.get('accuracy', 0.0),
            'learning_rate': learning_rate,
            'epoch_time': epoch_time
        }

        # Store in history
        for key, value in metrics.items():
            self.epoch_metrics[key].append(value)

        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar('Epoch/TrainLoss', metrics['train_loss'], epoch)
            self.writer.add_scalar('Epoch/TrainHardLoss', metrics['train_hard_loss'], epoch)
            self.writer.add_scalar('Epoch/TrainSoftLoss', metrics['train_soft_loss'], epoch)
            self.writer.add_scalar('Epoch/ValAUC', metrics['val_auc'], epoch)
            self.writer.add_scalar('Epoch/ValF1', metrics['val_f1'], epoch)
            self.writer.add_scalar('Epoch/ValMissRate', metrics['val_miss_rate'], epoch)
            self.writer.add_scalar('Epoch/ValFalseAlarmRate', metrics['val_false_alarm_rate'], epoch)
            self.writer.add_scalar('Epoch/LearningRate', metrics['learning_rate'], epoch)

        # Clear batch metrics for next epoch
        self.batch_metrics.clear()

    def log_gradients(self, model: nn.Module, epoch: int):
        """
        Log gradient statistics for model parameters.

        Args:
            model: PyTorch model
            epoch: Current epoch
        """
        grad_norms = []
        grad_means = []
        grad_stds = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
                grad_means.append(param.grad.mean().item())
                grad_stds.append(param.grad.std().item())

        stats = {
            'epoch': epoch,
            'mean_grad_norm': np.mean(grad_norms) if grad_norms else 0.0,
            'max_grad_norm': np.max(grad_norms) if grad_norms else 0.0,
            'mean_grad_mean': np.mean(grad_means) if grad_means else 0.0,
            'mean_grad_std': np.mean(grad_stds) if grad_stds else 0.0
        }

        for key, value in stats.items():
            self.grad_stats[key].append(value)

        if self.writer is not None:
            self.writer.add_histogram('Gradients/Norm', np.array(grad_norms), epoch)
            self.writer.add_scalar('Gradients/MeanNorm', stats['mean_grad_norm'], epoch)

    def log_predictions(
        self,
        epoch: int,
        predictions: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
        teacher_probs: Optional[np.ndarray] = None,
        sample_size: int = 1000
    ):
        """
        Log prediction distributions for visualization.

        Args:
            epoch: Current epoch
            predictions: Binary predictions
            labels: Ground truth labels
            probs: Predicted probabilities
            teacher_probs: Teacher probabilities (if available)
            sample_size: Number of samples to log (for efficiency)
        """
        if self.writer is None:
            return

        # Subsample if needed
        if len(probs) > sample_size:
            indices = np.random.choice(len(probs), sample_size, replace=False)
            probs = probs[indices]
            labels = labels[indices]
            if teacher_probs is not None:
                teacher_probs = teacher_probs[indices]

        # Log histograms
        self.writer.add_histogram('Predictions/Probabilities', probs, epoch)
        self.writer.add_histogram('Predictions/PositiveProbs', probs[labels == 1], epoch)
        self.writer.add_histogram('Predictions/NegativeProbs', probs[labels == 0], epoch)

        if teacher_probs is not None:
            self.writer.add_histogram('Predictions/TeacherProbs', teacher_probs, epoch)

            # Teacher-student agreement
            agreement = np.abs(probs - teacher_probs)
            self.writer.add_scalar('Metrics/TeacherStudentMAE', np.mean(agreement), epoch)

    def get_summary(self) -> Dict:
        """Get summary of all tracked metrics."""
        return {
            'fold_id': self.fold_id,
            'num_epochs': len(self.epoch_metrics.get('epoch', [])),
            'final_train_loss': self.epoch_metrics['train_loss'][-1] if self.epoch_metrics['train_loss'] else 0.0,
            'final_val_auc': self.epoch_metrics['val_auc'][-1] if self.epoch_metrics['val_auc'] else 0.0,
            'best_val_auc': max(self.epoch_metrics['val_auc']) if self.epoch_metrics['val_auc'] else 0.0,
            'best_val_f1': max(self.epoch_metrics['val_f1']) if self.epoch_metrics['val_f1'] else 0.0,
        }

    def save(self, path: Optional[str] = None):
        """
        Save all tracked metrics to disk.

        Args:
            path: Path to save metrics (default: log_dir/fold_metrics.json)
        """
        if path is None:
            path = self.log_dir / f"fold_{self.fold_id}_detailed_metrics.json"

        data = {
            'epoch_metrics': dict(self.epoch_metrics),
            'grad_stats': dict(self.grad_stats),
            'summary': self.get_summary()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Metrics saved to: {path}")

    def close(self):
        """Clean up resources."""
        if self.writer is not None:
            self.writer.close()


class TeacherStudentComparator:
    """
    Compare student predictions with teacher predictions.

    Useful for understanding how well the student is learning from the teacher.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.student_probs = []
        self.teacher_probs = []
        self.labels = []

    def add_batch(
        self,
        student_probs: Union[torch.Tensor, np.ndarray],
        teacher_probs: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray]
    ):
        """
        Add a batch of predictions for comparison.

        Args:
            student_probs: Student model probabilities
            teacher_probs: Teacher model probabilities
            labels: Ground truth labels
        """
        # Convert to numpy
        if isinstance(student_probs, torch.Tensor):
            student_probs = student_probs.detach().cpu().numpy()
        if isinstance(teacher_probs, torch.Tensor):
            teacher_probs = teacher_probs.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        self.student_probs.append(student_probs.flatten())
        self.teacher_probs.append(teacher_probs.flatten())
        self.labels.append(labels.flatten())

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute comparison metrics.

        Returns:
            Dictionary with agreement metrics
        """
        if not self.student_probs:
            return {}

        student = np.concatenate(self.student_probs)
        teacher = np.concatenate(self.teacher_probs)
        labels = np.concatenate(self.labels)

        # Filter valid labels
        valid_mask = labels >= 0
        student = student[valid_mask]
        teacher = teacher[valid_mask]
        labels = labels[valid_mask]

        # Compute metrics
        mae = np.mean(np.abs(student - teacher))
        mse = np.mean((student - teacher) ** 2)

        # Correlation
        if len(student) > 1 and np.std(student) > 0 and np.std(teacher) > 0:
            correlation = np.corrcoef(student, teacher)[0, 1]
        else:
            correlation = 0.0

        # Agreement at threshold
        student_pred = (student > 0.5).astype(int)
        teacher_pred = (teacher > 0.5).astype(int)
        agreement = np.mean(student_pred == teacher_pred)

        # Per-class agreement
        speech_mask = labels == 1
        silence_mask = labels == 0

        speech_agreement = np.mean(student_pred[speech_mask] == teacher_pred[speech_mask]) if np.any(speech_mask) else 0.0
        silence_agreement = np.mean(student_pred[silence_mask] == teacher_pred[silence_mask]) if np.any(silence_mask) else 0.0

        return {
            'mae': mae,
            'mse': mse,
            'correlation': correlation,
            'agreement': agreement,
            'speech_agreement': speech_agreement,
            'silence_agreement': silence_agreement
        }


def compute_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute the total gradient norm for a model.

    Args:
        model: PyTorch model
        norm_type: Type of norm (default: L2)

    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example of how to use MetricsTracker
    tracker = MetricsTracker(
        log_dir="test_logs",
        fold_id="F01",
        use_tensorboard=False
    )

    # Simulate training
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")

        # Simulate batches
        for batch in range(10):
            losses = {
                'total_loss': 0.5 - epoch * 0.1 + np.random.rand() * 0.1,
                'hard_loss': 0.3 - epoch * 0.05 + np.random.rand() * 0.05,
                'soft_loss': 0.2 - epoch * 0.05 + np.random.rand() * 0.05
            }

            tracker.log_batch(
                epoch=epoch,
                batch=batch,
                losses=losses,
                learning_rate=0.001,
                grad_norm=1.0 - epoch * 0.1,
                batch_time=0.1
            )

        # Epoch end
        train_metrics = {
            'train_loss': 0.5 - epoch * 0.1,
            'train_hard_loss': 0.3 - epoch * 0.05,
            'train_soft_loss': 0.2 - epoch * 0.05
        }

        val_metrics = {
            'auc': 0.7 + epoch * 0.05,
            'f1': 0.65 + epoch * 0.05,
            'miss_rate': 0.3 - epoch * 0.05,
            'false_alarm_rate': 0.25 - epoch * 0.03,
            'accuracy': 0.75 + epoch * 0.05
        }

        tracker.log_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=0.001,
            epoch_time=60.0
        )

        print(f"  Val AUC: {val_metrics['auc']:.4f}")

    # Save and summarize
    tracker.save()
    summary = tracker.get_summary()
    print("\nTraining Summary:")
    print(f"  Best Val AUC: {summary['best_val_auc']:.4f}")
    print(f"  Final Val AUC: {summary['final_val_auc']:.4f}")

    tracker.close()
