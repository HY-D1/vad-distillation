#!/usr/bin/env python3
"""
TinyVAD-style student model architecture.
CNN frontend + GRU backend for lightweight VAD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyVAD(nn.Module):
    """
    Lightweight VAD model for knowledge distillation.
    
    Architecture:
        - Conv frontend: 2-3 CNN layers with pooling
        - GRU backend: 1-2 GRU layers
        - Output: Frame-level speech probability
    
    Target size: ≤ 500 KB
    """
    
    def __init__(
        self,
        n_mels: int = 40,
        cnn_channels: list = [16, 32],
        gru_hidden: int = 32,
        gru_layers: int = 2,
        dropout: float = 0.0,
    ):
        """
        Args:
            n_mels: Number of mel frequency bins
            cnn_channels: List of CNN channel sizes
            gru_hidden: GRU hidden dimension
            gru_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_mels = n_mels
        
        # CNN frontend
        conv_layers = []
        in_channels = 1  # Single channel input (mel spectrogram)
        
        for out_channels in cnn_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),  # Pool in time only
            ])
            in_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate feature dimension after CNN
        # Assuming input is (batch, 1, time, n_mels)
        # After pooling by factor of 2 for each conv layer
        self.cnn_time_stride = 2 ** len(cnn_channels)
        self.cnn_out_channels = cnn_channels[-1] if cnn_channels else 1
        
        # Feature projection for GRU
        gru_input_dim = self.cnn_out_channels * n_mels  # After pooling
        # Actually, after CNN we have (batch, channels, time', n_mels')
        # Need to reshape: (batch, time', channels * n_mels')
        # But pooling is (2,1) so n_mels doesn't change
        
        self.gru_input_dim = self.cnn_out_channels * n_mels
        
        # GRU backend
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Output projection
        self.fc = nn.Linear(gru_hidden, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input mel spectrogram (batch, time, n_mels)
        
        Returns:
            probs: Speech probabilities (batch, time_frames)
        """
        batch_size, time_steps, n_mels = x.shape
        
        # Add channel dimension: (batch, 1, time, n_mels)
        x = x.unsqueeze(1)
        
        # CNN frontend
        x = self.conv(x)  # (batch, channels, time', n_mels)
        
        # Reshape for GRU: (batch, time', channels * n_mels)
        batch, channels, time_pooled, n_mels_out = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch, time', channels, n_mels)
        x = x.reshape(batch, time_pooled, -1)  # (batch, time', channels * n_mels)
        
        # GRU backend
        x, _ = self.gru(x)  # (batch, time', gru_hidden)
        
        # Output: (batch, time', 1)
        logits = self.fc(x).squeeze(-1)  # (batch, time')
        
        # Sigmoid for probability
        probs = torch.sigmoid(logits)
        
        return probs
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_kb(self) -> float:
        """Estimate model size in KB."""
        param_size = 0
        for param in self.parameters():
            param_size += param.numel() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        size_kb = (param_size + buffer_size) / 1024
        return size_kb


def create_student_model(config: dict = None) -> TinyVAD:
    """
    Factory function to create student model.
    
    Args:
        config: Dict with model hyperparameters
    
    Returns:
        TinyVAD model
    """
    default_config = {
        'n_mels': 40,
        'cnn_channels': [16, 32],
        'gru_hidden': 32,
        'gru_layers': 2,
        'dropout': 0.0,
    }
    
    if config:
        default_config.update(config)
    
    model = TinyVAD(**default_config)
    return model


if __name__ == "__main__":
    # Test model
    model = create_student_model()
    
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size_kb():.2f} KB")
    
    # Test forward pass
    batch_size = 4
    time_steps = 100  # 100 frames @ 32ms = 3.2 seconds
    n_mels = 40
    
    x = torch.randn(batch_size, time_steps, n_mels)
    probs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {probs.shape}")
