#!/usr/bin/env python3
"""
Cache teacher (Silero VAD) outputs for TORGO dataset.

This is a wrapper around run_silero_teacher.py for consistency.
All teacher outputs are cached to avoid recomputation.

Usage:
    python scripts/cache_teacher.py \
        --manifest manifests/torgo_sentences.csv \
        --output_dir teacher_probs/
"""

import argparse
import sys
from pathlib import Path

# Import the main function from run_silero_teacher
try:
    from run_silero_teacher import load_silero_vad, process_manifest
except ImportError:
    # If imported as module
    from scripts.run_silero_teacher import load_silero_vad, process_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Cache Silero VAD teacher outputs"
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
    
    print("=" * 60)
    print("Caching Silero VAD Teacher Outputs")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Output: {args.output_dir}")
    print()
    
    print("Loading Silero VAD model...")
    model, utils = load_silero_vad()
    model.to(args.device)
    model.eval()
    
    process_manifest(args.manifest, args.output_dir, model, utils)
    
    print()
    print("=" * 60)
    print("Caching complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
