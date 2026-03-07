#!/usr/bin/env python3
"""
Verification script for Windows 4080 checkpoints on Mac.

Loads a checkpoint trained on Windows 4080 and runs evaluation on Mac
to verify results are consistent.

Usage:
    python scripts/verify_checkpoint.py \
        --checkpoint outputs/production_4080/checkpoints/fold_F01_latest_best.pt \
        --config configs/production.yaml \
        --fold F01 \
        --output-dir outputs/production_4080/verification/

    # With specific device
    python scripts/verify_checkpoint.py \
        --checkpoint ... \
        --config ... \
        --fold F01 \
        --device cpu \
        --output-dir ...

    # With custom batch size (for memory constraints)
    python scripts/verify_checkpoint.py \
        --checkpoint ... \
        --config ... \
        --fold F01 \
        --batch-size 4 \
        --output-dir ...
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data import TORGODataset, collate_fn
from utils import load_config, ensure_dir, format_duration, get_device, compute_metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify Windows 4080 checkpoint on Mac",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/verify_checkpoint.py \\
      --checkpoint outputs/production_4080/checkpoints/fold_F01_latest_best.pt \\
      --config configs/production.yaml \\
      --fold F01 \\
      --output-dir outputs/production_4080/verification/

  # Use CPU explicitly
  python scripts/verify_checkpoint.py \\
      --checkpoint ... --config ... --fold F01 \\
      --device cpu --output-dir ...

  # Reduce batch size for memory
  python scripts/verify_checkpoint.py \\
      --checkpoint ... --config ... --fold F01 \\
      --batch-size 4 --output-dir ...
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.pt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--fold',
        type=str,
        required=True,
        help='Fold ID (e.g., F01)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save verification results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu/mps/cuda). Auto-detected if not specified.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for inference (overrides config)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--compare-original',
        action='store_true',
        help='Compare metrics with original Windows metrics from summary.json'
    )

    return parser.parse_args()


def load_fold_config(fold_id: str) -> Dict:
    """Load fold configuration from JSON file."""
    fold_path = Path("splits") / f"fold_{fold_id}.json"
    if not fold_path.exists():
        raise FileNotFoundError(f"Fold config not found: {fold_path}")

    with open(fold_path, 'r') as f:
        return json.load(f)


def create_model(config: Dict, checkpoint: Dict) -> nn.Module:
    """Create model from config and load checkpoint weights."""
    from models.tinyvad_student import create_student_model

    # Handle both nested and flat config structures
    if 'model' in config:
        model_config = config['model']
        model_type = model_config.get('type', 'tinyvad')
        model_params = model_config.get('params', model_config)
    else:
        model_type = config.get('model_type', 'tinyvad')
        model_params = {
            'n_mels': config.get('n_mels', 40),
            'cnn_channels': config.get('cnn_channels', [16, 24]),
            'gru_hidden': config.get('gru_hidden', 24),
            'gru_layers': config.get('gru_layers', 2),
            'dropout': config.get('dropout', 0.0),
        }

    # Create model
    if model_type.lower() in ('tinyvad', 'student'):
        model = create_student_model(model_params)
    elif model_type.lower() == 'tcn':
        from models.tcn import TCN
        model = TCN(**model_params)
    elif model_type.lower() == 'lstm':
        from models.lstm import LSTMVAD
        model = LSTMVAD(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def create_test_dataloader(config: Dict, fold_config: Dict, batch_size: int) -> DataLoader:
    """Create test dataloader."""
    # Handle both flat and nested config structures
    if 'data' in config:
        data_config = config['data']
        manifest = data_config.get('manifest', data_config.get('manifest_path', 'manifests/torgo_pilot.csv'))
        teacher_probs_dir = data_config.get('teacher_probs_dir', 'teacher_probs/')
        n_mels = data_config.get('n_mels', 40)
    else:
        manifest = config.get('manifest', 'manifests/torgo_pilot.csv')
        teacher_probs_dir = config.get('teacher_probs_dir', 'teacher_probs/')
        n_mels = config.get('n_mels', 40)

    test_dataset = TORGODataset(
        fold_config=fold_config,
        mode='test',
        manifest_path=manifest,
        teacher_probs_dir=teacher_probs_dir,
        n_mels=n_mels
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for Mac compatibility
        collate_fn=collate_fn,
        pin_memory=False  # Disable for Mac
    )

    return test_loader


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device,
                   threshold: float = 0.5) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on test set."""
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_labels = []
    all_probs = []
    all_utt_ids = []

    print("Running inference...")
    start_time = time.time()

    for batch_idx, batch in enumerate(test_loader):
        mels = batch['mels'].to(device)
        hard_labels = batch['hard_labels']
        utt_ids = batch['utt_ids']
        lengths = batch['lengths']

        # Forward pass
        probs = model(mels)
        predictions = (probs > threshold).long()

        # Process per utterance
        batch_size = predictions.shape[0]
        for i in range(batch_size):
            output_len = predictions.shape[1]
            input_len = lengths[i].item()
            effective_len = min(output_len, input_len)

            pred_i = predictions[i, :effective_len].cpu().numpy()
            label_i = hard_labels[i, :effective_len].cpu().numpy()
            prob_i = probs[i, :effective_len].cpu().numpy()

            all_predictions.extend(pred_i)
            all_labels.extend(label_i)
            all_probs.extend(prob_i)
            all_utt_ids.extend([utt_ids[i]] * effective_len)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

    inference_time = time.time() - start_time
    print(f"Inference completed in {format_duration(inference_time)}")

    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels, all_probs, threshold)

    return metrics, all_predictions, all_labels, all_probs


def load_original_metrics(log_dir: Path, fold_id: str) -> Optional[Dict]:
    """Load original Windows metrics from summary.json."""
    summary_path = log_dir / f"fold_{fold_id}_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            data = json.load(f)
            return data.get('test_metrics', {})
    return None


def compare_metrics(mac_metrics: Dict, windows_metrics: Dict) -> Dict[str, float]:
    """Compare Mac and Windows metrics."""
    differences = {}

    for key in ['auc', 'f1', 'accuracy', 'miss_rate', 'false_alarm_rate']:
        if key in windows_metrics:
            diff = abs(mac_metrics[key] - windows_metrics[key])
            differences[key] = diff

    return differences


def save_results(output_dir: Path,
                 fold_id: str,
                 metrics: Dict,
                 predictions: np.ndarray,
                 labels: np.ndarray,
                 probs: np.ndarray,
                 checkpoint_info: Dict,
                 comparison: Optional[Dict] = None):
    """Save verification results."""
    ensure_dir(output_dir)

    # Save metrics JSON
    results = {
        'fold_id': fold_id,
        'verification_device': checkpoint_info.get('device', 'unknown'),
        'checkpoint_epoch': checkpoint_info.get('epoch', 'unknown'),
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    if comparison:
        results['comparison_with_windows'] = comparison
        results['tolerance_check'] = {
            key: 'PASS' if diff < 0.001 else 'FAIL'
            for key, diff in comparison.items()
        }

    metrics_path = output_dir / f"fold_{fold_id}_mac_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Save predictions (optional, can be large)
    predictions_path = output_dir / f"fold_{fold_id}_mac_predictions.npz"
    np.savez(predictions_path,
             predictions=predictions,
             labels=labels,
             probs=probs)
    print(f"Saved predictions: {predictions_path}")

    return results


def print_verification_results(metrics: Dict, comparison: Optional[Dict] = None):
    """Print verification results."""
    print("\n" + "=" * 60)
    print("Verification Results (Mac)")
    print("=" * 60)
    print(f"\nMetrics:")
    print(f"  AUC:              {metrics['auc']:.4f}")
    print(f"  F1:               {metrics['f1']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Miss Rate:        {metrics['miss_rate']:.4f}")
    print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")

    if comparison:
        print("\n" + "-" * 60)
        print("Comparison with Windows 4080 Results")
        print("-" * 60)
        print(f"\n{'Metric':<20} {'Diff':<12} {'Status'}")
        print("-" * 50)

        all_pass = True
        for key, diff in comparison.items():
            status = 'PASS' if diff < 0.001 else 'FAIL'
            if status == 'FAIL':
                all_pass = False
            print(f"{key:<20} {diff:.6f}    {status}")

        print("-" * 50)
        if all_pass:
            print("\n✓ All metrics within tolerance (±0.001)")
        else:
            print("\n✗ Some metrics differ significantly")

    print("=" * 60)


def main():
    """Main verification function."""
    args = parse_args()

    print("=" * 60)
    print("Windows 4080 Checkpoint Verification on Mac")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Fold: {args.fold}")
    print("-" * 60)

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Loaded epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Parameters: {checkpoint.get('num_parameters', 'unknown'):,}")
    print(f"  Model size: {checkpoint.get('model_size_mb', 'unknown'):.2f} MB")

    # Load config
    print("\nLoading config...")
    config = load_config(args.config)

    # Get device
    device = get_device(args.device)
    checkpoint_info = {
        'device': str(device),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'num_parameters': checkpoint.get('num_parameters', 0),
        'model_size_mb': checkpoint.get('model_size_mb', 0)
    }

    # Load fold config
    print(f"\nLoading fold configuration for {args.fold}...")
    fold_config = load_fold_config(args.fold)
    train_key = 'train_speakers' if 'train_speakers' in fold_config else 'train'
    val_key = 'val_speaker' if 'val_speaker' in fold_config else 'val'
    test_key = 'test_speaker' if 'test_speaker' in fold_config else 'test'
    print(f"  Test speaker: {fold_config[test_key]}")

    # Create model
    print("\nCreating model...")
    model = create_model(config, checkpoint)
    model = model.to(device)
    print(f"  Model loaded successfully")

    # Get batch size
    batch_size = args.batch_size
    if batch_size is None:
        if 'training' in config:
            batch_size = config['training'].get('batch_size', 16)
        else:
            batch_size = config.get('batch_size', 8)
    print(f"  Batch size: {batch_size}")

    # Create dataloader
    print("\nCreating test dataloader...")
    test_loader = create_test_dataloader(config, fold_config, batch_size)
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Run evaluation
    print("\nEvaluating on test set...")
    metrics, predictions, labels, probs = evaluate_model(
        model, test_loader, device, args.threshold
    )

    # Load original metrics if requested
    original_metrics = None
    comparison = None
    if args.compare_original:
        log_dir = checkpoint_path.parent.parent / 'logs'
        original_metrics = load_original_metrics(log_dir, args.fold)
        if original_metrics:
            comparison = compare_metrics(metrics, original_metrics)
        else:
            print("\nWarning: Could not load original Windows metrics for comparison")

    # Save results
    print("\nSaving results...")
    output_dir = ensure_dir(args.output_dir)
    results = save_results(
        output_dir, args.fold, metrics,
        predictions, labels, probs,
        checkpoint_info, comparison
    )

    # Print results
    print_verification_results(metrics, comparison)

    print(f"\nVerification complete!")
    print(f"Results saved to: {output_dir}")

    # Return exit code based on comparison
    if comparison:
        if any(diff >= 0.001 for diff in comparison.values()):
            sys.exit(1)  # Fail if any metric differs significantly

    sys.exit(0)


if __name__ == '__main__':
    main()
