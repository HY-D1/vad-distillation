#!/usr/bin/env python3
"""
Training script for VAD knowledge distillation.

Usage:
    python train.py --config configs/pilot.yaml
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from models.tinyvad_student import create_student_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining hard and soft targets.
    
    Loss = (1 - alpha) * BCE(student, hard_labels) + 
           alpha * BCE(student_soft, teacher_soft)
    
    where teacher_soft is softened with temperature T.
    """
    
    def __init__(self, alpha: float = 0.5, temperature: float = 3.0):
        """
        Args:
            alpha: Weight for soft loss (0 = hard only, 1 = soft only)
            temperature: Temperature for softening teacher probabilities
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.bce = nn.BCELoss()
    
    def forward(
        self,
        student_probs: torch.Tensor,
        hard_labels: torch.Tensor,
        teacher_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_probs: Student predictions (batch, time) in [0, 1]
            hard_labels: Ground truth labels (batch, time) in {0, 1}
            teacher_probs: Teacher predictions (batch, time) in [0, 1]
        
        Returns:
            loss: Scalar loss
        """
        # Hard loss: standard BCE with ground truth
        hard_loss = self.bce(student_probs, hard_labels)
        
        # Soft loss: BCE with softened teacher probabilities
        # Temperature softening: p_soft = sigmoid(log(p) / T)
        # Or equivalently: soften in logit space
        
        # Convert to logits
        teacher_logits = torch.logit(teacher_probs.clamp(1e-7, 1 - 1e-7))
        student_logits = torch.logit(student_probs.clamp(1e-7, 1 - 1e-7))
        
        # Apply temperature
        teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
        student_soft = torch.sigmoid(student_logits / self.temperature)
        
        soft_loss = self.bce(student_soft, teacher_soft)
        
        # Combined loss
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return loss


class VADDataset(Dataset):
    """Dummy dataset for testing. Replace with actual TORGO dataset."""
    
    def __init__(self, num_samples: int = 100, seq_len: int = 100, n_mels: int = 40, cnn_stride: int = 4):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.n_mels = n_mels
        self.cnn_stride = cnn_stride
        # Labels and teacher probs must match downsampled output length
        self.label_len = seq_len // cnn_stride
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random mel spectrogram
        mels = torch.randn(self.seq_len, self.n_mels)
        
        # Random hard labels (0 or 1) - match downsampled length
        hard_labels = torch.rand(self.label_len) > 0.5
        hard_labels = hard_labels.float()
        
        # Random teacher probabilities - match downsampled length
        teacher_probs = torch.rand(self.label_len)
        
        return {
            'mels': mels,
            'hard_labels': hard_labels,
            'teacher_probs': teacher_probs,
        }


def collate_fn(batch):
    """Collate function for batching."""
    # Simple collate (assume same length for now)
    mels = torch.stack([b['mels'] for b in batch])
    hard_labels = torch.stack([b['hard_labels'] for b in batch])
    teacher_probs = torch.stack([b['teacher_probs'] for b in batch])
    
    return {
        'mels': mels,
        'hard_labels': hard_labels,
        'teacher_probs': teacher_probs,
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        mels = batch['mels'].to(device)
        hard_labels = batch['hard_labels'].to(device)
        teacher_probs = batch['teacher_probs'].to(device)
        
        # Forward
        optimizer.zero_grad()
        student_probs = model(mels)
        
        # Loss
        loss = criterion(student_probs, hard_labels, teacher_probs)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train VAD student model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pilot.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config: {config}")
    
    # Device
    device = torch.device(config.get('device', 'cpu'))
    print(f"Device: {device}")
    
    # Create model
    model = create_student_model(config.get('model', {}))
    model.to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size_kb():.2f} KB")
    
    # Create dataset (dummy for smoke test)
    # CNN stride = 2^num_cnn_layers (each pool halves the time dimension)
    num_cnn_layers = len(config.get('model', {}).get('cnn_channels', [16, 32]))
    cnn_stride = 2 ** num_cnn_layers
    dataset = VADDataset(
        num_samples=config.get('num_samples', 100),
        seq_len=config.get('seq_len', 100),
        n_mels=config.get('model', {}).get('n_mels', 40),
        cnn_stride=cnn_stride
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Loss and optimizer
    criterion = DistillationLoss(
        alpha=config.get('alpha', 0.5),
        temperature=config.get('temperature', 3.0)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3)
    )
    
    print(f"Distillation: alpha={criterion.alpha}, T={criterion.temperature}")
    
    # Training loop
    num_epochs = config.get('num_epochs', 1)
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    print("Training complete!")
    
    # Save model
    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'student_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
