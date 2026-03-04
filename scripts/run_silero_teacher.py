#!/usr/bin/env python3
"""
Run Silero VAD as teacher model on TORGO dataset.
Caches frame-level speech probabilities.

Usage:
    python scripts/run_silero_teacher.py \
        --manifest manifests/torgo_sentences.csv \
        --output_dir teacher_probs/
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


def load_silero_vad():
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    return model, utils


def get_speech_probs(model, utils, audio_path: str, sampling_rate: int = 16000):
    """
    Get frame-level speech probabilities from Silero VAD.
    
    Returns:
        probs: numpy array of shape (num_frames,)
    """
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    # Load audio
    wav = read_audio(audio_path, sampling_rate=sampling_rate)
    
    # Get speech probabilities
    # Silero internally processes in 32ms frames
    model.reset_states()
    with torch.no_grad():
        probs = model(wav, sampling_rate)
    
    return probs.numpy()


def process_manifest(manifest_path: str, output_dir: str, model, utils):
    """Process all files in manifest and cache probabilities."""
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
        prob_file = output_path / f"{speaker_id}_{utt_id}.npy"
        
        # Skip if already cached
        if prob_file.exists():
            print(f"[{i+1}/{len(rows)}] Skipping {utt_id} (cached)")
            continue
        
        try:
            # Get probabilities
            probs = get_speech_probs(model, utils, audio_path)
            
            # Save
            np.save(prob_file, probs)
            
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(rows)}] Processed {utt_id}")
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    print(f"Done. Outputs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Silero VAD teacher on TORGO"
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
        default="teacher_probs/",
        help="Output directory for probabilities"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    print("Loading Silero VAD model...")
    model, utils = load_silero_vad()
    model.to(args.device)
    model.eval()
    
    process_manifest(args.manifest, args.output_dir, model, utils)


if __name__ == "__main__":
    main()
