#!/usr/bin/env python3
"""
Models package for VAD distillation.

This package contains student model architectures and loss functions
for knowledge distillation from a teacher VAD model.
"""

from .losses import (
    DistillationLoss,
    BCEDistillationLoss,
    HardLabelLoss,
    create_loss_function,
)
from .tinyvad_student import (
    TinyVAD,
    create_student_micro,
    create_student_model,
    create_student_small,
    create_student_tiny,
)

__all__ = [
    # Models
    'TinyVAD',
    'create_student_model',
    'create_student_small',
    'create_student_tiny',
    'create_student_micro',
    # Losses
    'DistillationLoss',
    'BCEDistillationLoss',
    'HardLabelLoss',
    'create_loss_function',
]
