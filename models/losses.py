#!/usr/bin/env python3
"""
Loss functions for VAD knowledge distillation.

This module contains various loss functions used for training
student VAD models via knowledge distillation.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined distillation loss: alpha * hard_loss + (1-alpha) * soft_loss
    
    This is the primary loss function for knowledge distillation training.
    It combines hard labels (ground truth) with soft labels (teacher probabilities).
    
    Supports both:
    - 2-class output: student_logits [..., 2], teacher_probs [..., 2]
    - 1-class output: student_logits [...], teacher_probs [...] (binary sigmoid)
    
    Args:
        alpha: Weight for hard loss (CE with ground truth). (1-alpha) for soft loss.
        temperature: Temperature for softening probability distributions.
    """
    
    def __init__(self, alpha: float = 0.5, temperature: float = 3.0):
        """
        Initialize the distillation loss.
        
        Args:
            alpha: Weight for hard loss (0 = soft only, 1 = hard only)
            temperature: Temperature for softening teacher probabilities
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def _pool_to_target_length(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Pool tensor to match target length using adaptive average pooling.
        
        Args:
            tensor: Input tensor of shape [batch, seq_len] or [batch, seq_len, features]
            target_len: Target sequence length
            
        Returns:
            Pooled tensor of shape [batch, target_len] or [batch, target_len, features]
        """
        if tensor.shape[1] == target_len:
            return tensor
        
        # Use adaptive average pooling to downsample
        # For 2D tensor [batch, seq_len], treat as [batch, 1, seq_len]
        # For 3D tensor [batch, seq_len, features], treat as [batch, features, seq_len]
        if tensor.dim() == 2:
            # [batch, seq_len] -> [batch, 1, seq_len] -> [batch, 1, target_len] -> [batch, target_len]
            tensor_pooled = F.adaptive_avg_pool1d(tensor.unsqueeze(1), target_len)
            return tensor_pooled.squeeze(1)
        else:
            # [batch, seq_len, features] -> [batch, features, seq_len] -> [batch, features, target_len] -> [batch, target_len, features]
            tensor_transposed = tensor.transpose(1, 2)  # [batch, features, seq_len]
            tensor_pooled = F.adaptive_avg_pool1d(tensor_transposed, target_len)
            return tensor_pooled.transpose(1, 2)  # [batch, target_len, features]
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: Logits from student model 
                - [batch, seq_len] for single-class (sigmoid) output
                - [batch, seq_len, 2] for 2-class (softmax) output
            teacher_probs: Soft probabilities from teacher 
                - [batch, seq_len] for single-class
                - [batch, seq_len, 2] for 2-class
            labels: Hard labels 
                - [batch, seq_len] for both cases
        
        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary with individual loss components
        """
        # Determine if this is single-class (sigmoid) or 2-class (softmax) output
        # Single-class: student_logits.dim() == 2 and last dim matches labels
        # 2-class: student_logits.dim() == 3 and last dim is 2
        is_single_class = student_logits.dim() == 2 or (
            student_logits.dim() == 3 and student_logits.shape[-1] != 2
        )
        
        # Handle time dimension mismatch (e.g., from CNN downsampling in TinyVAD)
        # If student output is shorter than labels/teacher_probs, pool the targets
        if student_logits.shape[1] != labels.shape[1]:
            target_len = student_logits.shape[1]
            labels = self._pool_to_target_length(labels, target_len)
            teacher_probs = self._pool_to_target_length(teacher_probs, target_len)
        
        # Ensure all tensors have the same dtype as student_logits
        if labels.dtype != student_logits.dtype:
            labels = labels.to(student_logits.dtype)
        if teacher_probs.dtype != student_logits.dtype:
            teacher_probs = teacher_probs.to(student_logits.dtype)
        
        if is_single_class:
            # Single-class binary classification (sigmoid output)
            # student_logits: [batch, seq_len] or [batch]
            # teacher_probs: [batch, seq_len] or [batch]
            # labels: [batch, seq_len] or [batch]
            
            # Flatten if needed
            if student_logits.dim() == 2:
                student_logits_flat = student_logits.reshape(-1)
                teacher_probs_flat = teacher_probs.reshape(-1)
                labels_flat = labels.reshape(-1).float()
            else:
                student_logits_flat = student_logits
                teacher_probs_flat = teacher_probs
                labels_flat = labels.float()
            
            # Hard loss: BCE with ground truth
            hard_loss = self.bce_loss(student_logits_flat, labels_flat)
            
            # Soft loss: BCE with softened teacher probabilities
            # Apply temperature scaling
            student_probs_soft = torch.sigmoid(student_logits_flat / self.temperature)
            teacher_probs_soft = teacher_probs_flat.pow(1 / self.temperature)
            teacher_probs_soft = teacher_probs_soft / (teacher_probs_soft + (1 - teacher_probs_flat).pow(1 / self.temperature))
            
            soft_loss = F.binary_cross_entropy(student_probs_soft, teacher_probs_soft)
            soft_loss = soft_loss * (self.temperature ** 2)  # Scale by T^2
            
        else:
            # 2-class classification (softmax output)
            # Handle both sequence and frame-level predictions
            if student_logits.dim() == 3:
                # Sequence-level: [batch, seq_len, num_classes]
                batch_size, seq_len, num_classes = student_logits.shape
                student_logits_flat = student_logits.reshape(-1, num_classes)
                teacher_probs_flat = teacher_probs.reshape(-1, num_classes)
                labels_flat = labels.reshape(-1)
            else:
                # Frame-level: [batch, num_classes]
                student_logits_flat = student_logits
                teacher_probs_flat = teacher_probs
                labels_flat = labels
            
            # Hard loss: Cross-entropy with ground truth
            hard_loss = self.ce_loss(student_logits_flat, labels_flat)
            
            # Soft loss: KL divergence with teacher predictions
            # Apply temperature scaling
            student_probs_soft = torch.log_softmax(
                student_logits_flat / self.temperature, dim=-1
            )
            teacher_probs_soft = teacher_probs_flat.pow(1 / self.temperature)
            teacher_probs_soft = teacher_probs_soft / teacher_probs_soft.sum(dim=-1, keepdim=True)
            
            soft_loss = self.kl_loss(student_probs_soft, teacher_probs_soft)
            soft_loss = soft_loss * (self.temperature ** 2)  # Scale by T^2
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        loss_dict = {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


class BCEDistillationLoss(nn.Module):
    """
    Alternative distillation loss using BCE instead of KL divergence.
    
    This version uses binary cross-entropy and is suitable when the
    student outputs probabilities directly (not logits).
    
    Loss = (1 - alpha) * BCE(student, hard_labels) + 
           alpha * BCE(student_soft, teacher_soft)
    
    where teacher_soft is softened with temperature T.
    """
    
    def __init__(self, alpha: float = 0.5, temperature: float = 3.0):
        """
        Initialize the BCE distillation loss.
        
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
        Compute combined distillation loss.
        
        Args:
            student_probs: Student predictions (batch, time) in [0, 1]
            hard_labels: Ground truth labels (batch, time) in {0, 1}
            teacher_probs: Teacher predictions (batch, time) in [0, 1]
        
        Returns:
            loss: Scalar loss tensor
        """
        # Hard loss: standard BCE with ground truth
        hard_loss = self.bce(student_probs, hard_labels)
        
        # Soft loss: BCE with softened teacher probabilities
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


class HardLabelLoss(nn.Module):
    """
    Simple hard label loss for baseline training.
    
    This is just a wrapper around CrossEntropyLoss for convenience.
    """
    
    def __init__(self):
        """Initialize the hard label loss."""
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Model predictions [batch, seq_len, num_classes] or [batch, num_classes]
            labels: Ground truth labels [batch, seq_len] or [batch]
        
        Returns:
            loss: Scalar loss tensor
        """
        if logits.dim() == 3:
            # Flatten sequence dimension
            batch_size, seq_len, num_classes = logits.shape
            logits = logits.reshape(-1, num_classes)
            labels = labels.reshape(-1)
        
        return self.ce(logits, labels)


def create_loss_function(
    loss_type: str = 'distillation',
    alpha: float = 0.5,
    temperature: float = 3.0
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('distillation', 'bce_distillation', 'hard')
        alpha: Weight for distillation loss
        temperature: Temperature for softening
    
    Returns:
        Loss function module
    
    Raises:
        ValueError: If loss_type is not recognized
    """
    if loss_type == 'distillation':
        return DistillationLoss(alpha=alpha, temperature=temperature)
    elif loss_type == 'bce_distillation':
        return BCEDistillationLoss(alpha=alpha, temperature=temperature)
    elif loss_type == 'hard':
        return HardLabelLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


__all__ = [
    'DistillationLoss',
    'BCEDistillationLoss',
    'HardLabelLoss',
    'create_loss_function',
]
