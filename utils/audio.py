#!/usr/bin/env python3
"""
Audio loading and processing utilities.

This module provides unified audio loading functions used across the project
for consistent audio I/O operations.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio


def load_audio(
    audio_path: Union[str, Path],
    target_sr: int = 16000,
    return_tensor: bool = False,
    device: Optional[torch.device] = None,
) -> Union[
    Tuple[np.ndarray, int],
    Tuple[torch.Tensor, int],
    None,
]:
    """
    Load audio file with optional resampling and format conversion.

    This is the unified audio loading function used across the project.
    It replaces the duplicate implementations in:
    - scripts/data/cache_teacher.py
    - scripts/core/run_baseline.py
    - baselines/speechbrain_vad.py

    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate for resampling (default: 16000)
        return_tensor: If True, return waveform as torch.Tensor.
                      If False, return waveform as numpy.ndarray (default: False)
        device: Optional torch device to move tensor to (only used if return_tensor=True)

    Returns:
        Tuple of (waveform, sample_rate) where:
            - waveform: Audio data as numpy array [samples] or torch.Tensor [samples]
            - sample_rate: Sample rate of the returned audio (equals target_sr if resampled)
        Returns None if loading fails.

    Raises:
        No exceptions are raised. Errors are logged to stderr and None is returned.

    Examples:
        >>> # Load as numpy array (default)
        >>> waveform, sr = load_audio("audio.wav")
        >>> waveform.shape
        (16000,)

        >>> # Load as torch tensor
        >>> waveform, sr = load_audio("audio.wav", return_tensor=True)
        >>> waveform.shape
        torch.Size([16000])

        >>> # Load with specific target sample rate
        >>> waveform, sr = load_audio("audio.wav", target_sr=8000)
        >>> sr
        8000
    """
    audio_path = Path(audio_path)

    # Check if file exists
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        return None

    try:
        # Load audio using torchaudio
        waveform, orig_sr = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        else:
            sr = orig_sr

        # Remove channel dimension (convert to 1D)
        waveform = waveform.squeeze(0)

        # Move to device if specified (only for tensor output)
        if return_tensor:
            if device is not None:
                waveform = waveform.to(device)
            return waveform, sr
        else:
            # Convert to numpy array
            return waveform.numpy(), sr

    except (RuntimeError, IOError) as e:
        print(f"Error loading audio file {audio_path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error loading audio file {audio_path}: {e}", file=sys.stderr)
        return None


def get_audio_duration(audio_path: Union[str, Path]) -> float:
    """
    Get audio duration in seconds without loading the full file.

    Args:
        audio_path: Path to the audio file

    Returns:
        Duration in seconds, or 0.0 if file cannot be read
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return 0.0

    try:
        info = torchaudio.info(str(audio_path))
        return info.num_frames / info.sample_rate
    except (RuntimeError, IOError):
        return 0.0


def resample_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    orig_sr: int,
    target_sr: int,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Resample a waveform to a new sample rate.

    Args:
        waveform: Audio waveform as numpy array or torch tensor
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled waveform in the same format as input
    """
    if orig_sr == target_sr:
        return waveform

    if isinstance(waveform, np.ndarray):
        # Convert to tensor, resample, convert back
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(tensor).squeeze(0).numpy()
        return resampled
    else:
        # Already a tensor
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(waveform).squeeze(0)


# Backward compatibility aliases
load_audio_file = load_audio


if __name__ == "__main__":
    # Simple test when run directly
    import tempfile

    print("Testing audio utilities...")

    # Create a simple test audio file
    test_sr = 16000
    test_duration = 1.0  # 1 second
    test_freq = 440.0  # A4 note
    t = np.linspace(0, test_duration, int(test_sr * test_duration), endpoint=False)
    test_waveform = np.sin(2 * np.pi * test_freq * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        test_path = f.name

    # Save test audio
    torchaudio.save(test_path, torch.from_numpy(test_waveform).unsqueeze(0), test_sr)

    # Test load as numpy
    waveform, sr = load_audio(test_path, return_tensor=False)
    assert isinstance(waveform, np.ndarray), "Should return numpy array"
    assert sr == test_sr, f"Sample rate should be {test_sr}"
    assert len(waveform) == int(test_sr * test_duration), "Duration mismatch"
    print("✓ load_audio (numpy) works")

    # Test load as tensor
    waveform, sr = load_audio(test_path, return_tensor=True)
    assert isinstance(waveform, torch.Tensor), "Should return torch tensor"
    assert sr == test_sr, f"Sample rate should be {test_sr}"
    print("✓ load_audio (tensor) works")

    # Test get_audio_duration
    duration = get_audio_duration(test_path)
    assert abs(duration - test_duration) < 0.01, f"Duration should be ~{test_duration}"
    print("✓ get_audio_duration works")

    # Test resample
    waveform_8k = resample_waveform(test_waveform, test_sr, 8000)
    assert len(waveform_8k) == int(8000 * test_duration), "Resampled duration mismatch"
    print("✓ resample_waveform works")

    # Cleanup
    Path(test_path).unlink()
    print("\nAll tests passed!")
