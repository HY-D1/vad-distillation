#!/usr/bin/env python3
"""
LOSO-aware training script for VAD distillation.

This script trains a student VAD model using knowledge distillation from a teacher model,
with Leave-One-Speaker-Out (LOSO) cross-validation.

Usage:
    python train_loso.py --config configs/pilot.yaml --fold F01
    python train_loso.py --config configs/pilot.yaml --fold F01 --resume checkpoints/fold_F01_latest.pt
    python train_loso.py --config configs/pilot.yaml --fold F01 --test  # Dry-run mode
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
import yaml

# Add project root to path for imports (works both from project root and scripts/)
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from data import TORGODataset, collate_fn
    from models.losses import DistillationLoss
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root or have proper PYTHONPATH set.")
    sys.exit(1)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(predictions: np.ndarray, 
                    labels: np.ndarray,
                    probs: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Binary predictions [N]
        labels: Ground truth labels [N]
        probs: Probability of positive class [N]
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    
    # Flatten arrays
    predictions = predictions.flatten()
    labels = labels.flatten()
    probs = probs.flatten()
    
    # Mask out invalid labels (-1)
    valid_mask = labels >= 0
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    probs = probs[valid_mask]
    
    if len(labels) == 0:
        return {
            'auc': 0.0,
            'f1': 0.0,
            'miss_rate': 0.0,
            'false_alarm_rate': 0.0,
            'accuracy': 0.0
        }
    
    # AUC
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    
    # F1 Score
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Confusion matrix for miss rate and false alarm rate
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    
    # Miss Rate (False Negative Rate) = FN / (FN + TP)
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # False Alarm Rate (False Positive Rate) = FP / (FP + TN)
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'auc': auc,
        'f1': f1,
        'miss_rate': miss_rate,
        'false_alarm_rate': false_alarm_rate,
        'accuracy': accuracy,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


# =============================================================================
# Data Loading
# =============================================================================

def load_fold_config(fold_path: str) -> Dict:
    """Load fold configuration from JSON file."""
    with open(fold_path, 'r') as f:
        return json.load(f)


def create_dataloaders(config: Dict, fold_config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from fold configuration.
    
    Args:
        config: Main configuration dictionary
        fold_config: Fold configuration with train/val/test speakers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Handle both flat and nested config structures
    if 'data' in config:
        data_config = config['data']
        training_config = config.get('training', {})
        manifest = data_config.get('manifest', data_config.get('manifest_path', 'manifests/torgo_pilot.csv'))
        teacher_probs_dir = data_config.get('teacher_probs_dir', 'teacher_probs/')
        n_mels = data_config.get('n_mels', 40)
        batch_size = training_config.get('batch_size', 16)
        num_workers = training_config.get('num_workers', 0)
    else:
        # Flat config (e.g., pilot.yaml)
        manifest = config.get('manifest', 'manifests/torgo_pilot.csv')
        teacher_probs_dir = config.get('teacher_probs_dir', 'teacher_probs/')
        n_mels = config.get('n_mels', 40)
        batch_size = config.get('batch_size', 8)
        num_workers = config.get('num_workers', 0)
    
    # Common dataset arguments
    dataset_kwargs = {
        'manifest_path': manifest,
        'teacher_probs_dir': teacher_probs_dir,
        'n_mels': n_mels,
    }
    
    # Create datasets using fold_config
    train_dataset = TORGODataset(
        fold_config=fold_config,
        mode='train',
        **dataset_kwargs
    )
    
    val_dataset = TORGODataset(
        fold_config=fold_config,
        mode='val',
        **dataset_kwargs
    )
    
    test_dataset = TORGODataset(
        fold_config=fold_config,
        mode='test',
        **dataset_kwargs
    )
    
    # Create dataloaders (batch_size and num_workers already set above)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
    print(f"Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
    print(f"Test samples: {len(test_dataset)}, batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# Model Creation
# =============================================================================

def create_model(config: Dict) -> nn.Module:
    """
    Create model from configuration.
    
    Supports both nested config (config['model']) and flat config.
    """
    from models.tinyvad_student import create_student_model
    
    # Handle both nested and flat config structures
    if 'model' in config:
        # Nested config with model section
        model_config = config['model']
        model_type = model_config.get('type', 'tinyvad')
        model_params = model_config.get('params', model_config)
    else:
        # Flat config - use all model-related params directly
        model_type = config.get('model_type', 'tinyvad')
        model_params = {
            'n_mels': config.get('n_mels', 40),
            'cnn_channels': config.get('cnn_channels', [16, 24]),
            'gru_hidden': config.get('gru_hidden', 24),
            'gru_layers': config.get('gru_layers', 2),
            'dropout': config.get('dropout', 0.0),
        }
    
    # Import model dynamically
    try:
        if model_type.lower() == 'tcn':
            from models.tcn import TCN
            model = TCN(**model_params)
        elif model_type.lower() == 'lstm':
            from models.lstm import LSTMVAD
            model = LSTMVAD(**model_params)
        elif model_type.lower() == 'mlp':
            from models.mlp import MLPAcousticVAD
            model = MLPAcousticVAD(**model_params)
        elif model_type.lower() in ('tinyvad', 'student'):
            model = create_student_model(model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except ImportError as e:
        print(f"Error importing model {model_type}: {e}")
        print("Falling back to TinyVAD student model...")
        model = create_student_model(model_params)
    
    return model


class SimpleVADModel(nn.Module):
    """Simple LSTM-based VAD model as fallback."""
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 2 classes: speech, non-speech
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch, seq_len, input_dim]
        Returns:
            logits: [batch, seq_len, 2]
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model: nn.Module, 
                train_loader: DataLoader,
                criterion: DistillationLoss,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with average loss values for the epoch.
    """
    model.train()
    
    total_loss = 0.0
    total_hard_loss = 0.0
    total_soft_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        features = batch['mels'].to(device)
        teacher_probs = batch['teacher_probs'].to(device)
        labels = batch['hard_labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)
        
        # Compute loss
        loss, loss_dict = criterion(logits, teacher_probs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss_dict['total_loss']
        total_hard_loss += loss_dict['hard_loss']
        total_soft_loss += loss_dict['soft_loss']
        num_batches += 1
        
        # Print progress every N batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss_dict['total_loss']:.4f} "
                  f"(Hard: {loss_dict['hard_loss']:.4f}, "
                  f"Soft: {loss_dict['soft_loss']:.4f})")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_hard_loss = total_hard_loss / num_batches if num_batches > 0 else 0.0
    avg_soft_loss = total_soft_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'train_loss': avg_loss,
        'train_hard_loss': avg_hard_loss,
        'train_soft_loss': avg_soft_loss
    }


@torch.no_grad()
def validate(model: nn.Module, 
             val_loader: DataLoader,
             device: torch.device,
             threshold: float = 0.5) -> Dict[str, float]:
    """
    Validate model on validation set.
    
    Supports both:
    - TinyVAD: single-class output (sigmoid probabilities)
    - 2-class models: softmax output with 2 classes
    
    Returns:
        Dictionary with validation metrics.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    for batch in val_loader:
        features = batch['mels'].to(device)
        labels = batch['hard_labels'].to(device)
        
        # Forward pass
        output = model(features)
        
        # Handle time dimension mismatch (e.g., from CNN downsampling in TinyVAD)
        # If model output is shorter than labels, pool the labels
        if output.shape[1] != labels.shape[1]:
            import torch.nn.functional as F
            target_len = output.shape[1]
            # Pool labels: [batch, seq_len] -> [batch, target_len]
            labels_pooled = F.adaptive_avg_pool1d(
                labels.unsqueeze(1).float(), target_len
            ).squeeze(1)
            labels = (labels_pooled > 0.5).long()  # Convert back to binary
        
        # Determine output type: TinyVAD outputs single probabilities (sigmoid)
        # while 2-class models output logits for 2 classes (softmax)
        if output.dim() == 2:
            # TinyVAD: output is (batch, time) single probabilities from sigmoid
            probs = output
            predictions = (probs > threshold).long()
        else:
            # 2-class model: output is (batch, time, 2) logits
            probs = torch.softmax(output, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        # Collect for metric computation
        # Handle sequence dimension
        if predictions.dim() == 2:
            predictions = predictions.cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            if output.dim() == 2:
                # TinyVAD: probs is already (batch, time) of speech probabilities
                probs_np = probs.cpu().numpy().flatten()
            else:
                # 2-class: get probability of speech (class 1)
                probs_np = probs[:, :, 1].cpu().numpy().flatten()
        else:
            predictions = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()
            if output.dim() == 1:
                probs_np = probs.cpu().numpy()
            else:
                probs_np = probs[:, 1].cpu().numpy()
        
        all_predictions.append(predictions)
        all_labels.append(labels_np)
        all_probs.append(probs_np)
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels, all_probs, threshold)
    
    return metrics


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    metrics: Dict[str, float],
                    config: Dict,
                    path: str,
                    is_best: bool = False):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'model_size_mb': get_model_size_mb(model),
        'num_parameters': count_parameters(model)
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    
    if is_best:
        best_path = path.replace('.pt', '_best.pt')
        torch.save(checkpoint, best_path)
        print(f"  Saved best model to {best_path}")
    
    print(f"  Saved checkpoint to {path}")


def load_checkpoint(path: str, 
                    model: nn.Module, 
                    optimizer: Optional[optim.Optimizer] = None) -> Tuple[int, Dict[str, float]]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    
    return epoch, metrics


# =============================================================================
# Logging
# =============================================================================

class CSVLogger:
    """Simple CSV logger for training metrics."""
    
    def __init__(self, log_path: str, fieldnames: List[str]):
        self.log_path = log_path
        self.fieldnames = fieldnames
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Write header if file doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                f.write(','.join(fieldnames) + '\n')
    
    def log(self, metrics: Dict[str, float]):
        """Log metrics to CSV."""
        with open(self.log_path, 'a', newline='') as f:
            values = [str(metrics.get(k, '')) for k in self.fieldnames]
            f.write(','.join(values) + '\n')


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics."""
    print(f"\n{prefix}Metrics:")
    print(f"  AUC: {metrics.get('auc', 0):.4f}")
    print(f"  F1: {metrics.get('f1', 0):.4f}")
    print(f"  Miss Rate: {metrics.get('miss_rate', 0):.4f}")
    print(f"  False Alarm Rate: {metrics.get('false_alarm_rate', 0):.4f}")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")


# =============================================================================
# Test/Dry-Run Mode
# =============================================================================

def test_mode(config: Dict, fold_config: Dict, device: torch.device):
    """
    Run test mode to verify setup without full training.
    """
    print("\n" + "="*60)
    print("TEST MODE - Verifying setup")
    print("="*60)
    
    # 1. Check paths exist
    print("\n1. Checking paths...")
    
    # Handle both flat and nested config structures
    if 'data' in config:
        manifest_path = config['data'].get('manifest', config['data'].get('manifest_path', ''))
        teacher_probs_path = config['data'].get('teacher_probs_dir', '')
    else:
        manifest_path = config.get('manifest', config.get('manifest_path', ''))
        teacher_probs_path = config.get('teacher_probs_dir', '')
    
    paths_to_check = [
        manifest_path,
        teacher_probs_path,
    ]
    
    all_exist = True
    for path in paths_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\nERROR: Some paths do not exist!")
        return False
    
    # 2. Load data samples
    print("\n2. Loading data samples...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config, fold_config)
        print("  ✓ DataLoaders created successfully")
    except Exception as e:
        print(f"  ✗ Error creating DataLoaders: {e}")
        return False
    
    # 3. Check model forward pass
    print("\n3. Checking model forward pass...")
    try:
        model = create_model(config).to(device)
        print(f"  ✓ Model created: {count_parameters(model):,} parameters")
        print(f"  ✓ Model size: {get_model_size_mb(model):.2f} MB")
        
        # Test forward pass with one batch
        batch = next(iter(train_loader))
        features = batch['mels'].to(device)
        teacher_probs = batch['teacher_probs'].to(device)
        labels = batch['hard_labels'].to(device)
        
        with torch.no_grad():
            logits = model(features)
        
        print(f"  ✓ Forward pass successful")
        print(f"    Input shape: {features.shape}")
        print(f"    Output shape: {logits.shape}")
        print(f"    Labels shape: {labels.shape}")
        print(f"    Teacher probs shape: {teacher_probs.shape}")
        
    except Exception as e:
        print(f"  ✗ Error in model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test loss computation
    print("\n4. Checking loss computation...")
    try:
        alpha = config.get('alpha', 0.5)
        temperature = config.get('temperature', 3.0)
        criterion = DistillationLoss(alpha=alpha, temperature=temperature)
        
        with torch.no_grad():
            loss, loss_dict = criterion(logits, teacher_probs, labels)
        
        print(f"  ✓ Loss computation successful")
        print(f"    Total loss: {loss_dict['total_loss']:.4f}")
        print(f"    Hard loss: {loss_dict['hard_loss']:.4f}")
        print(f"    Soft loss: {loss_dict['soft_loss']:.4f}")
        
    except Exception as e:
        print(f"  ✗ Error in loss computation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test one training step
    print("\n5. Testing one training step...")
    try:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass
        logits = model(features)
        loss, loss_dict = criterion(logits, teacher_probs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Training step successful")
        
    except Exception as e:
        print(f"  ✗ Error in training step: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Test validation
    print("\n6. Testing validation...")
    try:
        model.eval()
        val_metrics = validate(model, val_loader, device)
        print(f"  ✓ Validation successful")
        print_metrics(val_metrics, prefix="  ")
        
    except Exception as e:
        print(f"  ✗ Error in validation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("TEST MODE PASSED - All checks successful!")
    print("="*60)
    
    return True


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LOSO Training for VAD Distillation')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--fold', type=str, required=True,
                        help='Fold ID (e.g., F01)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                        help='Run test mode only (dry-run)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detected if not specified.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['num_epochs'] = args.epochs
        print(f"Overriding num_epochs to {args.epochs}")
    
    if args.patience is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['early_stopping_patience'] = args.patience
        print(f"Overriding early_stopping_patience to {args.patience}")
    
    # Set device
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load fold configuration
    fold_id = args.fold
    fold_path = os.path.join("splits", f"fold_{fold_id}.json")
    print(f"Loading fold configuration from {fold_path}")
    
    if not os.path.exists(fold_path):
        print(f"ERROR: Fold configuration not found: {fold_path}")
        sys.exit(1)
    
    fold_config = load_fold_config(fold_path)
    # Handle both old and new fold config formats
    train_key = 'train_speakers' if 'train_speakers' in fold_config else 'train'
    val_key = 'val_speaker' if 'val_speaker' in fold_config else 'val'
    test_key = 'test_speaker' if 'test_speaker' in fold_config else 'test'
    print(f"Train speakers: {fold_config[train_key]}")
    print(f"Val speaker: {fold_config[val_key]}")
    print(f"Test speaker: {fold_config[test_key]}")
    
    # Test mode
    if args.test:
        success = test_mode(config, fold_config, device)
        sys.exit(0 if success else 1)
    
    # Create output directories - handle both flat and nested configs
    if 'output' in config:
        checkpoint_dir = config['output'].get('checkpoint_dir', 'outputs/checkpoints/')
        log_dir = config['output'].get('log_dir', 'outputs/logs/')
    else:
        output_dir = config.get('output_dir', 'outputs/')
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, fold_config)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config).to(device)
    num_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Create optimizer - handle both flat and nested configs
    if 'training' in config:
        training_config = config['training']
    else:
        training_config = config  # Flat config
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create scheduler
    scheduler_type = training_config.get('scheduler', 'plateau')
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=training_config['num_epochs'], eta_min=1e-6
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None
    
    # Create loss function
    alpha = config.get('alpha', 0.5)
    temperature = config.get('temperature', 3.0)
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    print(f"Distillation alpha: {alpha}, temperature: {temperature}")
    
    # Setup logging
    log_path = os.path.join(log_dir, f"fold_{fold_id}.csv")
    fieldnames = ['epoch', 'train_loss', 'train_hard_loss', 'train_soft_loss',
                  'val_auc', 'val_f1', 'val_miss_rate', 'val_false_alarm_rate',
                  'val_accuracy', 'learning_rate', 'time']
    logger = CSVLogger(log_path, fieldnames)
    print(f"Logging to: {log_path}")
    
    # Setup checkpoint paths
    checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold_id}_latest.pt")
    best_checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold_id}_best.pt")
    
    # Training state
    start_epoch = 0
    best_val_auc = 0.0
    epochs_no_improve = 0
    early_stopping_patience = training_config.get('early_stopping_patience', 10)
    checkpoint_interval = training_config.get('checkpoint_interval', 10)
    num_epochs = training_config['num_epochs']
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
            start_epoch += 1  # Start from next epoch
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Warning: Resume checkpoint not found: {args.resume}")
    
    # Training loop
    print("\n" + "="*60)
    print(f"Starting training for fold {fold_id}")
    print("="*60)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_metrics['train_loss']:.4f} "
              f"(Hard: {train_metrics['train_hard_loss']:.4f}, "
              f"Soft: {train_metrics['train_soft_loss']:.4f})")
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        print_metrics(val_metrics, prefix="Val ")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['train_loss'],
            'train_hard_loss': train_metrics['train_hard_loss'],
            'train_soft_loss': train_metrics['train_soft_loss'],
            'val_auc': val_metrics['auc'],
            'val_f1': val_metrics['f1'],
            'val_miss_rate': val_metrics['miss_rate'],
            'val_false_alarm_rate': val_metrics['false_alarm_rate'],
            'val_accuracy': val_metrics['accuracy'],
            'learning_rate': current_lr,
            'time': time.time() - epoch_start_time
        }
        logger.log(log_entry)
        
        # Update scheduler
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_metrics['auc'])
            else:
                scheduler.step()
        
        # Save checkpoint
        is_best = val_metrics['auc'] > best_val_auc
        if is_best:
            best_val_auc = val_metrics['auc']
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Save latest checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or is_best or epoch == num_epochs - 1:
            save_checkpoint(
                model, optimizer, epoch, val_metrics, config,
                checkpoint_path, is_best=is_best
            )
        
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    print("\n" + "="*60)
    print("Training completed. Loading best model for test evaluation...")
    print("="*60)
    
    if os.path.exists(best_checkpoint_path):
        load_checkpoint(best_checkpoint_path, model)
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = validate(model, test_loader, device)
    print_metrics(test_metrics, prefix="Test ")
    
    # Save test predictions
    print("\nSaving test predictions...")
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    all_utt_ids = []
    
    for batch in test_loader:
        mels = batch['mels'].to(device)
        hard_labels = batch['hard_labels']
        utt_ids = batch['utt_ids']
        
        with torch.no_grad():
            probs = model(mels)  # TinyVAD outputs probabilities directly
            predictions = (probs > 0.5).long()
        
        # Handle sequence dimension
        if predictions.dim() == 2:
            predictions = predictions.cpu().numpy().flatten()
            labels_np = hard_labels.cpu().numpy().flatten()
            probs_np = probs.cpu().numpy().flatten()
        else:
            predictions = predictions.cpu().numpy()
            labels_np = hard_labels.cpu().numpy()
            probs_np = probs.cpu().numpy()
        
        all_predictions.extend(predictions)
        all_labels.extend(labels_np)
        all_probs.extend(probs_np)
        # Repeat utt_ids for each frame in sequence
        for i, utt_id in enumerate(utt_ids):
            seq_len = mels.shape[1] if predictions.ndim == 1 else 1
            all_utt_ids.extend([utt_id] * seq_len)
    
    # Save predictions to file
    predictions_path = os.path.join(log_dir, f"fold_{fold_id}_predictions.npz")
    np.savez(predictions_path,
             predictions=np.array(all_predictions),
             labels=np.array(all_labels),
             probs=np.array(all_probs),
             utt_ids=np.array(all_utt_ids))
    print(f"Predictions saved to: {predictions_path}")
    
    # Save summary - handle both old and new fold config key formats
    train_key = 'train_speakers' if 'train_speakers' in fold_config else 'train'
    val_key = 'val_speaker' if 'val_speaker' in fold_config else 'val'
    test_key = 'test_speaker' if 'test_speaker' in fold_config else 'test'
    
    summary = {
        'fold_id': fold_id,
        'train_speakers': fold_config[train_key],
        'val_speaker': fold_config[val_key],
        'test_speaker': fold_config[test_key],
        'num_parameters': num_params,
        'model_size_mb': model_size_mb,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'config': config
    }
    
    summary_path = os.path.join(log_dir, f"fold_{fold_id}_summary.json")
    with open(summary_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            return obj
        
        json.dump(convert_to_native(summary), f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    print("\n" + "="*60)
    print(f"Fold {fold_id} training completed!")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
