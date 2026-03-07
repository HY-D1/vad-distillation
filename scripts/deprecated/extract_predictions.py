#!/usr/bin/env python3
"""Extract individual .npy files from predictions.npz for comparison script."""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def extract_predictions(predictions_npz, output_dir):
    """Extract .npy files from predictions.npz with frame-level data grouped by utterance."""
    print(f"Loading predictions from: {predictions_npz}")
    data = np.load(predictions_npz)
    
    # Expected keys: predictions, labels, probs, utt_ids (all frame-level)
    probs = data['probs']
    utt_ids = data['utt_ids']
    
    print(f"Total frames: {len(probs)}")
    print(f"Unique utterances: {len(set(utt_ids))}")
    
    # Create output directory
    frame_probs_dir = Path(output_dir) / 'frame_probs'
    frame_probs_dir.mkdir(parents=True, exist_ok=True)
    
    # Group frames by utterance ID
    utt_frames = defaultdict(list)
    for prob, utt_id in zip(probs, utt_ids):
        utt_frames[utt_id].append(prob)
    
    # Save individual .npy files (named exactly as utt_id from predictions)
    extracted = 0
    for utt_id, frames in utt_frames.items():
        # Sanitize filename (remove problematic characters)
        safe_id = str(utt_id).replace('/', '_').replace('\\', '_')
        prob_array = np.array(frames, dtype=np.float32)
        output_path = frame_probs_dir / f"{safe_id}.npy"
        np.save(output_path, prob_array)
        extracted += 1
    
    print(f"Extracted {extracted} .npy files to: {frame_probs_dir}")
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract .npy files from predictions.npz")
    parser.add_argument('--predictions', required=True, help="Path to predictions.npz file")
    parser.add_argument('--output-dir', required=True, help="Output directory for frame_probs/")
    
    args = parser.parse_args()
    
    extracted = extract_predictions(args.predictions, args.output_dir)
    print(f"\nDone! Extracted {extracted} utterance predictions.")


if __name__ == '__main__':
    main()
