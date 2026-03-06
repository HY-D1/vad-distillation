#!/usr/bin/env python3
"""
TinyVAD-style student model architecture.
CNN frontend + GRU backend for lightweight VAD.

Frame Alignment Note:
    The CNN frontend uses MaxPool2d with kernel_size=(2, 1) after each conv layer,
    which downsamples the time dimension by a factor of 2 per layer.
    
    For N CNN layers, the total downsampling factor is 2^N.
    
    Example with default config (2 CNN layers):
    - Input: mel spectrogram with 32ms frames (31.25 fps)
    - After CNN: time is downsampled by 2^2 = 4
    - Output frame rate: 31.25 / 4 = 7.8125 fps (128ms frames)
    
    To match teacher at 31.25 fps output:
    - Use input with 8ms frames (125 fps), or
    - Interpolate output back to original frame rate, or
    - Adjust teacher/target labels to match student frame rate
"""

import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
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
    
    Target size: ≤ 500 KB (default config)
    
    Frame Rate Alignment:
        The CNN downsamples time by factor of 2^N where N = number of CNN layers.
        With default 2 layers: downsampling factor = 4
        
        To align with teacher (31.25 fps):
        - Input should be at higher frame rate (125 fps = 8ms frames) for 2-layer CNN
        - Or interpolate student output to match teacher frame rate
    """
    
    def __init__(
        self,
        n_mels: int = 40,
        cnn_channels: list = [16, 32],
        gru_hidden: int = 32,
        gru_layers: int = 2,
        dropout: float = 0.0,
        sample_rate: int = 16000,
        hop_length: int = 128,  # 8ms @ 16kHz for 125 fps input
    ):
        """
        Args:
            n_mels: Number of mel frequency bins
            cnn_channels: List of CNN channel sizes
            gru_hidden: GRU hidden dimension
            gru_layers: Number of GRU layers
            dropout: Dropout rate
            sample_rate: Audio sample rate for mel computation
            hop_length: Hop length for mel spectrogram (controls input frame rate)
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # CNN frontend
        conv_layers = []
        in_channels = 1  # Single channel input (mel spectrogram)
        
        for out_channels in cnn_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),  # Pool in time only, factor of 2
            ])
            in_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate feature dimension after CNN
        # After N conv layers with (2,1) pooling: time_stride = 2^N
        self.num_cnn_layers = len(cnn_channels)
        self.cnn_time_stride = 2 ** self.num_cnn_layers
        self.cnn_out_channels = cnn_channels[-1] if cnn_channels else 1
        
        # Feature projection for GRU
        # After CNN: (batch, channels, time', n_mels) -> reshape to (batch, time', channels * n_mels)
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
        
        # CNN frontend - downsamples time by factor of 2^N
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
    
    def predict(
        self,
        audio: np.ndarray,
        device: Optional[torch.device] = None,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Inference on raw audio with mel spectrogram preprocessing.
        
        Args:
            audio: Raw audio samples (numpy array), shape (samples,) or (batch, samples)
            device: Device to run inference on (default: CPU)
            return_numpy: If True, return numpy array; otherwise return torch tensor
        
        Returns:
            probs: Frame-level speech probabilities
                   - If input is 1D: shape (num_frames,)
                   - If input is 2D: shape (batch, num_frames)
        """
        if device is None:
            device = torch.device('cpu')
        
        self.eval()
        self.to(device)
        
        # Handle input dimensions
        single_input = False
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Add batch dimension
            single_input = True
        elif audio.ndim != 2:
            raise ValueError(f"Audio must be 1D or 2D, got shape {audio.shape}")
        
        # Compute mel spectrogram for each audio in batch
        mels_list = []
        for audio_sample in audio:
            mel = self._compute_mel_spectrogram(audio_sample)
            mels_list.append(mel)
        
        # Stack into batch
        max_len = max(m.shape[0] for m in mels_list)
        mels_padded = []
        for mel in mels_list:
            if mel.shape[0] < max_len:
                pad = np.zeros((max_len - mel.shape[0], mel.shape[1]))
                mel = np.concatenate([mel, pad], axis=0)
            mels_padded.append(mel)
        
        mels = np.stack(mels_padded, axis=0)  # (batch, time, n_mels)
        
        # Convert to tensor and run inference
        mels_tensor = torch.from_numpy(mels).float().to(device)
        
        with torch.no_grad():
            probs = self.forward(mels_tensor)
        
        # Convert to numpy if requested
        if return_numpy:
            probs = probs.cpu().numpy()
            if single_input:
                probs = probs[0]  # Remove batch dimension
        elif single_input:
            probs = probs[0]  # Remove batch dimension
        
        return probs
    
    def _compute_mel_spectrogram(
        self,
        audio: np.ndarray,
        n_fft: int = 512,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute mel spectrogram from raw audio.
        
        Args:
            audio: Raw audio samples (1D numpy array)
            n_fft: FFT window size
            win_length: Window length for STFT
            f_min: Minimum frequency
            f_max: Maximum frequency (default: sample_rate / 2)
        
        Returns:
            mel_spec: Mel spectrogram (time, n_mels)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for mel spectrogram computation. "
                "Install with: pip install librosa"
            )
        
        if f_max is None:
            f_max = self.sample_rate / 2
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=win_length,
            n_mels=self.n_mels,
            fmin=f_min,
            fmax=f_max,
            power=2.0,  # Power spectrogram
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to roughly [0, 1] range (typical for neural networks)
        mel_spec_norm = (mel_spec_db + 80) / 80  # Rough normalization
        mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
        
        # Transpose to (time, n_mels)
        mel_spec_norm = mel_spec_norm.T
        
        return mel_spec_norm
    
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
    
    def get_flops(self, input_shape: Optional[Tuple[int, int, int]] = None) -> dict:
        """
        Estimate FLOPs for the model.
        
        Args:
            input_shape: Input shape (batch, time, n_mels). If None, uses (1, 100, n_mels)
        
        Returns:
            Dictionary with FLOP estimates for each component
        """
        if input_shape is None:
            input_shape = (1, 100, self.n_mels)
        
        batch, time_steps, n_mels = input_shape
        
        # CNN FLOPs
        cnn_flops = 0
        time_dim = time_steps
        in_ch = 1
        for out_ch in [16, 32][:self.num_cnn_layers] if self.num_cnn_layers > 0 else []:
            # Conv2d: kernel_size=3, stride=1, padding=1
            # FLOPs ≈ 2 * kernel_h * kernel_w * in_ch * out_ch * out_h * out_w
            conv_flops = 2 * 3 * 3 * in_ch * out_ch * time_dim * n_mels
            cnn_flops += conv_flops
            
            # After pooling, time dimension is halved
            time_dim = time_dim // 2
            in_ch = out_ch
        
        # GRU FLOPs
        # GRU has 3 gates, each with 2 linear transforms (input and hidden)
        # FLOPs ≈ 3 * 2 * (input_dim * hidden + hidden * hidden) * time_steps * num_layers
        time_after_cnn = time_steps // self.cnn_time_stride
        gru_flops = (
            3 * 2 * (self.gru_input_dim * self.gru.hidden_size + 
                     self.gru.hidden_size * self.gru.hidden_size) *
            time_after_cnn * self.gru.num_layers
        )
        
        # Linear layer FLOPs
        linear_flops = 2 * self.gru.hidden_size * 1 * time_after_cnn
        
        total_flops = cnn_flops + gru_flops + linear_flops
        
        return {
            'cnn_flops': cnn_flops,
            'gru_flops': gru_flops,
            'linear_flops': linear_flops,
            'total_flops': total_flops,
            'total_macs': total_flops / 2,  # MACs ≈ FLOPs / 2
            'input_shape': input_shape,
            'output_frames': time_after_cnn,
        }
    
    def export_onnx(
        self,
        path: Union[str, Path],
        input_shape: Tuple[int, int, int] = (1, 100, 40),
        opset_version: int = 11
    ) -> Path:
        """
        Export model to ONNX format.
        
        Args:
            path: Path to save the ONNX model
            input_shape: Example input shape (batch, time, n_mels)
            opset_version: ONNX opset version
        
        Returns:
            Path to saved ONNX model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Suppress onnxscript warnings if onnxscript not installed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Export
                torch.onnx.export(
                    self,
                    dummy_input,
                    path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['mel_spectrogram'],
                    output_names=['speech_probability'],
                    dynamic_axes={
                        'mel_spectrogram': {0: 'batch_size', 1: 'time'},
                        'speech_probability': {0: 'batch_size', 1: 'time'}
                    }
                )
            except Exception as e:
                # If export fails due to missing dependencies, create a placeholder
                if 'onnxscript' in str(e).lower() or 'dynamo' in str(e).lower():
                    # Try with simpler settings
                    torch.onnx.export(
                        self,
                        dummy_input,
                        path,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=False,
                        input_names=['mel_spectrogram'],
                        output_names=['speech_probability'],
                    )
                else:
                    raise
        
        return path
    
    def export_torchscript(
        self,
        path: Union[str, Path],
        input_shape: Tuple[int, int, int] = (1, 100, 40),
        method: str = 'trace'
    ) -> Path:
        """
        Export model to TorchScript format.
        
        Args:
            path: Path to save the TorchScript model
            input_shape: Example input shape for tracing
            method: 'trace' or 'script'
        
        Returns:
            Path to saved TorchScript model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.eval()
        
        if method == 'trace':
            dummy_input = torch.randn(input_shape)
            scripted = torch.jit.trace(self, dummy_input)
        elif method == 'script':
            scripted = torch.jit.script(self)
        else:
            raise ValueError(f"Method must be 'trace' or 'script', got {method}")
        
        scripted.save(str(path))
        
        return path
    
    def get_model_info(self) -> dict:
        """Get comprehensive model information."""
        return {
            'parameters': self.count_parameters(),
            'size_kb': self.get_model_size_kb(),
            'size_mb': self.get_model_size_kb() / 1024,
            'cnn_layers': self.num_cnn_layers,
            'cnn_time_stride': self.cnn_time_stride,
            'gru_hidden': self.gru.hidden_size,
            'gru_layers': self.gru.num_layers,
            'n_mels': self.n_mels,
        }


def create_student_model(config: dict = None) -> TinyVAD:
    """
    Factory function to create student model.
    
    Default config targets ≤ 500 KB.
    
    Args:
        config: Dict with model hyperparameters
    
    Returns:
        TinyVAD model
    """
    default_config = {
        'n_mels': 40,
        'cnn_channels': [16, 24],  # Reduced from [16, 32] to meet size target
        'gru_hidden': 24,  # Reduced from 32 to meet size target
        'gru_layers': 2,
        'dropout': 0.0,
        'sample_rate': 16000,
        'hop_length': 128,
    }
    
    if config:
        default_config.update(config)
    
    model = TinyVAD(**default_config)
    return model


def create_student_small() -> TinyVAD:
    """
    Create a smaller student model (~300-400 KB).
    
    Config:
        - 2 CNN layers: [12, 24] channels
        - GRU: 20 hidden, 2 layers
    """
    config = {
        'n_mels': 40,
        'cnn_channels': [12, 24],
        'gru_hidden': 20,
        'gru_layers': 2,
        'dropout': 0.0,
    }
    return create_student_model(config)


def create_student_tiny() -> TinyVAD:
    """
    Create a tiny student model (~100-200 KB).
    
    Config:
        - 1 CNN layer: [16] channels
        - GRU: 16 hidden, 1 layer
    """
    config = {
        'n_mels': 40,
        'cnn_channels': [16],
        'gru_hidden': 16,
        'gru_layers': 1,
        'dropout': 0.0,
    }
    return create_student_model(config)


def create_student_micro() -> TinyVAD:
    """
    Create a micro student model (~50-100 KB).
    
    Config:
        - 1 CNN layer: [8] channels
        - GRU: 8 hidden, 1 layer
    """
    config = {
        'n_mels': 40,
        'cnn_channels': [8],
        'gru_hidden': 8,
        'gru_layers': 1,
        'dropout': 0.0,
    }
    return create_student_model(config)


def test_forward_pass(model: TinyVAD, batch_size: int = 2, time_steps: int = 100) -> bool:
    """Test that forward pass works correctly."""
    try:
        x = torch.randn(batch_size, time_steps, model.n_mels)
        probs = model(x)
        
        expected_time = time_steps // model.cnn_time_stride
        assert probs.shape == (batch_size, expected_time), \
            f"Expected shape {(batch_size, expected_time)}, got {probs.shape}"
        assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities not in [0, 1]"
        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False


def test_model_size(model: TinyVAD, max_size_kb: float = 500.0) -> bool:
    """Test that model size is under target."""
    size_kb = model.get_model_size_kb()
    if size_kb > max_size_kb:
        print(f"  ✗ Size {size_kb:.2f} KB exceeds target {max_size_kb:.2f} KB")
        return False
    return True


def test_export_onnx(model: TinyVAD, output_dir: Path) -> bool:
    """Test ONNX export."""
    try:
        onnx_path = output_dir / 'test_model.onnx'
        model.export_onnx(onnx_path)
        
        # Verify file exists and has content
        if not onnx_path.exists():
            print("  ✗ ONNX file not created")
            return False
        if onnx_path.stat().st_size == 0:
            print("  ✗ ONNX file is empty")
            return False
        
        # Clean up
        onnx_path.unlink()
        return True
    except ImportError as e:
        print(f"  ⚠ ONNX export skipped (missing dependency: {e})")
        return True  # Don't fail tests due to missing optional dependency
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        return False


def test_export_torchscript(model: TinyVAD, output_dir: Path) -> bool:
    """Test TorchScript export."""
    try:
        ts_path = output_dir / 'test_model.pt'
        model.export_torchscript(ts_path)
        
        # Verify file exists and has content
        if not ts_path.exists():
            print("  ✗ TorchScript file not created")
            return False
        if ts_path.stat().st_size == 0:
            print("  ✗ TorchScript file is empty")
            return False
        
        # Try to load it back
        loaded = torch.jit.load(str(ts_path))
        
        # Clean up
        ts_path.unlink()
        return True
    except Exception as e:
        print(f"  ✗ TorchScript export failed: {e}")
        return False


def test_predict_method(model: TinyVAD) -> bool:
    """Test the predict method with dummy audio."""
    try:
        # Skip if librosa not available
        try:
            import librosa
        except ImportError:
            print("  ⚠ librosa not installed, skipping predict test")
            return True
        
        # Create dummy audio (1 second @ 16kHz)
        duration = 1.0
        sample_rate = model.sample_rate
        samples = int(duration * sample_rate)
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        
        # Test prediction
        probs = model.predict(audio)
        
        # Check output
        assert probs.ndim == 1, f"Expected 1D output, got {probs.ndim}D"
        assert len(probs) > 0, "Empty output"
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities not in [0, 1]"
        
        return True
    except Exception as e:
        print(f"  ✗ Predict method failed: {e}")
        return False


def run_comprehensive_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("TinyVAD Student Model - Comprehensive Tests")
    print("=" * 70)
    
    # Create temporary directory for exports
    output_dir = Path(tempfile.mkdtemp(prefix="tinyvad_test_"))
    
    # Define model variants
    variants = [
        ('Default (≤500 KB)', create_student_model, 500),
        ('Small (≤400 KB)', create_student_small, 400),
        ('Tiny (≤200 KB)', create_student_tiny, 200),
        ('Micro (≤100 KB)', create_student_micro, 100),
    ]
    
    results = []
    
    for name, factory_fn, max_size in variants:
        print(f"\n{'─' * 70}")
        print(f"Testing: {name}")
        print('─' * 70)
        
        model = factory_fn()
        info = model.get_model_info()
        
        print(f"  Parameters: {info['parameters']:,}")
        print(f"  Size: {info['size_kb']:.2f} KB ({info['size_mb']:.3f} MB)")
        print(f"  CNN layers: {info['cnn_layers']} (stride: {info['cnn_time_stride']}x)")
        print(f"  GRU: {info['gru_layers']} layer(s), {info['gru_hidden']} hidden")
        
        # Run tests
        tests = [
            ('Forward Pass', test_forward_pass(model)),
            ('Size Check', test_model_size(model, max_size)),
            ('ONNX Export', test_export_onnx(model, output_dir)),
            ('TorchScript Export', test_export_torchscript(model, output_dir)),
            ('Predict Method', test_predict_method(model)),
        ]
        
        # Print test results
        for test_name, passed in tests:
            status = "✓" if passed else "✗"
            print(f"  {status} {test_name}")
        
        # Calculate FLOPs
        flops = model.get_flops()
        print(f"  FLOPs: {flops['total_flops']:,} ({flops['total_macs']:,} MACs)")
        
        all_passed = all(passed for _, passed in tests)
        results.append({
            'name': name,
            'params': info['parameters'],
            'size_kb': info['size_kb'],
            'flops': flops['total_flops'],
            'passed': all_passed,
        })
    
    # Print summary table
    print(f"\n{'=' * 70}")
    print("Summary Table")
    print('=' * 70)
    print(f"{'Variant':<20} {'Parameters':>12} {'Size (KB)':>12} {'FLOPs':>15} {'Status':>8}")
    print('-' * 70)
    
    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        flops_str = f"{r['flops']/1e6:.2f}M" if r['flops'] > 1e6 else f"{r['flops']/1e3:.2f}K"
        print(f"{r['name']:<20} {r['params']:>12,} {r['size_kb']:>12.2f} {flops_str:>15} {status:>8}")
    
    # Print frame alignment info
    print(f"\n{'=' * 70}")
    print("Frame Alignment Information")
    print('=' * 70)
    print("""
The CNN frontend downsamples time by factor of 2^N where N = number of CNN layers.

For matching teacher at 31.25 fps (32ms frames):
  - Default/Small (2 CNN layers, 4x stride): Use 125 fps input (hop_length=128 @ 16kHz)
  - Tiny/Micro (1 CNN layer, 2x stride): Use 62.5 fps input (hop_length=256 @ 16kHz)

Alternative: Interpolate student output to match teacher frame rate.
""")
    
    # Cleanup
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    
    print(f"\nAll tests complete!")
    
    # Return overall success
    return all(r['passed'] for r in results)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)
