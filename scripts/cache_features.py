#!/usr/bin/env python3
"""
Cache mel-spectrogram features for TORGO dataset.
Avoids recomputing features during training.

Usage:
    python scripts/cache_features.py \
        --manifest manifests/torgo_sentences.csv \
        --output_dir cached_features/
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchaudio


def compute_melspectrogram(
    audio_path: str,
    sr: int = 16000,
    n_mels: int = 40,
    n_fft: int = 512,
    hop_length: int = 512
) -> np.ndarray:
    """
    Compute mel spectrogram for audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Target sampling rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length
    
    Returns:
        mel: Mel spectrogram (n_mels, time)
    """
    # Load audio
    waveform, orig_sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)
    
    # Compute mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    
    mel = mel_transform(waveform)
    
    # Convert to log scale (dB)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    
    return mel.squeeze(0).numpy().T  # (time, n_mels)


def process_manifest(manifest_path: str, output_dir: str, n_mels: int = 40):
    """Process all files in manifest and cache features."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Processing {len(rows)} utterances...")
    
    for i, row in enumerate(rows):
        audio_path = row['path']
        utt_id = row['utt_id']
        speaker_id = row['speaker_id']
        
        # Output filename
        feat_file = output_path / f"{speaker_id}_{utt_id}.npy"
        
        # Skip if already cached
        if feat_file.exists():
            print(f"[{i+1}/{len(rows)}] Skipping {utt_id} (cached)")
            continue
        
        try:
            # Compute features
            mel = compute_melspectrogram(audio_path, n_mels=n_mels)
            
            # Save
            np.save(feat_file, mel)
            
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(rows)}] Processed {utt_id}, shape: {mel.shape}")
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    print(f"Done. Features saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Cache mel spectrogram features"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/torgo_sentences.csv",
        help="Path to manifest CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cached_features/",
        help="Output directory for features"
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=40,
        help="Number of mel frequency bins"
    )
    
    args = parser.parse_args()
    
    process_manifest(args.manifest, args.output_dir, args.n_mels)


if __name__ == "__main__":
    main()
