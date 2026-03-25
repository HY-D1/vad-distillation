#!/usr/bin/env python3
"""
Shared audio loading utilities for personal milestone scripts.
"""

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio_mono(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono float32 and resample when needed.

    Args:
        audio_path: Path to waveform file.
        target_sr: Target sampling rate.

    Returns:
        Tuple of (mono waveform, sample rate).
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sample_rate = sf.read(str(path), always_2d=False)
    waveform = np.asarray(waveform)

    # Convert multi-channel audio to mono by averaging channels.
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    waveform = waveform.astype(np.float32, copy=False)

    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    return waveform, sample_rate
