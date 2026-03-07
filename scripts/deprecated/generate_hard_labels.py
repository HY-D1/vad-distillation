#!/usr/bin/env python3
"""
Generate hard labels from teacher probabilities by thresholding.

Usage:
    # Generate hard labels at default threshold (0.5)
    python scripts/generate_hard_labels.py \
        --teacher_probs_dir teacher_probs/ \
        --output_dir teacher_hard_labels/
    
    # Generate at custom threshold
    python scripts/generate_hard_labels.py \
        --teacher_probs_dir teacher_probs/ \
        --output_dir teacher_hard_labels/thresh_0.3/ \
        --threshold 0.3
    
    # Generate multiple thresholds at once
    python scripts/generate_hard_labels.py \
        --teacher_probs_dir teacher_probs/ \
        --output_dir teacher_hard_labels/ \
        --thresholds 0.3 0.5 0.7
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def load_probabilities(probs_dir: Path) -> List[Path]:
    """Find all .npy probability files."""
    prob_files = sorted(probs_dir.glob('*.npy'))
    return prob_files


def threshold_probabilities(probs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert soft probabilities to hard binary labels.
    
    Args:
        probs: Soft probabilities (float32 array)
        threshold: Threshold for binarization
    
    Returns:
        Binary labels as int8 (0 or 1)
    """
    return (probs >= threshold).astype(np.int8)


def process_file(
    prob_file: Path,
    output_dir: Path,
    threshold: float
) -> Tuple[int, int, int]:
    """
    Process a single probability file and save hard labels.
    
    Returns:
        Tuple of (total_frames, speech_frames, silence_frames)
    """
    # Load probabilities
    probs = np.load(prob_file)
    
    # Threshold
    hard_labels = threshold_probabilities(probs, threshold)
    
    # Save
    output_file = output_dir / prob_file.name
    np.save(output_file, hard_labels)
    
    # Calculate statistics
    total_frames = len(hard_labels)
    speech_frames = int(np.sum(hard_labels == 1))
    silence_frames = int(np.sum(hard_labels == 0))
    
    return total_frames, speech_frames, silence_frames


def generate_hard_labels(
    probs_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict:
    """
    Generate and save hard labels from teacher probabilities.
    
    Args:
        probs_dir: Directory containing .npy probability files
        output_dir: Output directory for hard labels
        threshold: Threshold for binarization
        verbose: Whether to print progress
    
    Returns:
        Summary statistics dictionary
    """
    probs_path = Path(probs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all probability files
    prob_files = load_probabilities(probs_path)
    
    if len(prob_files) == 0:
        print(f"Error: No .npy files found in {probs_dir}")
        sys.exit(1)
    
    if verbose:
        print(f"Found {len(prob_files)} probability files")
        print(f"Threshold: {threshold}")
        print(f"Output directory: {output_path}")
        print()
    
    # Process all files
    total_stats = {
        'total_utterances': 0,
        'total_frames': 0,
        'speech_frames': 0,
        'silence_frames': 0,
    }
    
    iterator = tqdm(prob_files, desc=f"Processing (threshold={threshold})") if verbose else prob_files
    
    for prob_file in iterator:
        try:
            total_frames, speech_frames, silence_frames = process_file(
                prob_file, output_path, threshold
            )
            
            total_stats['total_utterances'] += 1
            total_stats['total_frames'] += total_frames
            total_stats['speech_frames'] += speech_frames
            total_stats['silence_frames'] += silence_frames
            
        except Exception as e:
            if verbose:
                print(f"Error processing {prob_file.name}: {e}")
            continue
    
    # Calculate percentages
    if total_stats['total_frames'] > 0:
        total_stats['speech_percentage'] = (
            total_stats['speech_frames'] / total_stats['total_frames'] * 100
        )
        total_stats['silence_percentage'] = (
            total_stats['silence_frames'] / total_stats['total_frames'] * 100
        )
    
    # Build summary
    summary = {
        'threshold': threshold,
        **total_stats,
        'teacher_probs_dir': str(probs_dir),
        'output_dir': str(output_dir),
    }
    
    # Save metadata
    meta_file = output_path / 'meta.json'
    with open(meta_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def print_summary(summary: Dict):
    """Print summary statistics."""
    print()
    print("=" * 50)
    print("HARD LABEL GENERATION SUMMARY")
    print("=" * 50)
    print(f"Threshold: {summary['threshold']}")
    print(f"Total utterances: {summary['total_utterances']:,}")
    print(f"Total frames: {summary['total_frames']:,}")
    print()
    print("Frame Distribution:")
    print(f"  Speech frames:  {summary['speech_frames']:,} ({summary['speech_percentage']:.2f}%)")
    print(f"  Silence frames: {summary['silence_frames']:,} ({summary['silence_percentage']:.2f}%)")
    print("=" * 50)


def verify_outputs(output_dir: str, num_samples: int = 5):
    """
    Verify generated hard labels.
    
    Checks:
    - Values are only 0 or 1
    - Files are smaller than float32 equivalents
    """
    output_path = Path(output_dir)
    label_files = list(output_path.glob('*.npy'))
    
    if len(label_files) == 0:
        print("No hard label files found for verification")
        return
    
    print()
    print("VERIFICATION")
    print("-" * 30)
    
    # Sample a few files
    sample_files = np.random.choice(label_files, min(num_samples, len(label_files)), replace=False)
    
    all_valid = True
    for label_file in sample_files:
        labels = np.load(label_file)
        
        # Check values
        unique_values = np.unique(labels)
        is_valid = set(unique_values).issubset({0, 1})
        
        status = "✓" if is_valid else "✗"
        print(f"{status} {label_file.name}: shape={labels.shape}, values={unique_values}")
        
        if not is_valid:
            all_valid = False
    
    # Check file sizes
    if len(label_files) > 0:
        total_size = sum(f.stat().st_size for f in label_files)
        avg_size = total_size / len(label_files)
        print(f"\nTotal files: {len(label_files)}")
        print(f"Average file size: {avg_size:.1f} bytes")
        print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    
    if all_valid:
        print("\n✓ All verified files contain only 0s and 1s")
    else:
        print("\n✗ Some files contain unexpected values")
    
    print("-" * 30)


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard labels from teacher probabilities"
    )
    parser.add_argument(
        '--teacher_probs_dir',
        type=str,
        required=True,
        help='Directory containing teacher probability .npy files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for hard labels'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binarization (default: 0.5)'
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=None,
        help='Multiple thresholds to generate (e.g., 0.3 0.5 0.7)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify generated hard labels'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Handle multiple thresholds
    if args.thresholds:
        if verbose:
            print(f"Generating hard labels for {len(args.thresholds)} thresholds: {args.thresholds}")
            print()
        
        all_summaries = []
        for threshold in args.thresholds:
            # Create subdirectory for each threshold
            thresh_output_dir = Path(args.output_dir) / f'thresh_{threshold}'
            
            summary = generate_hard_labels(
                args.teacher_probs_dir,
                str(thresh_output_dir),
                threshold,
                verbose=verbose
            )
            all_summaries.append(summary)
            
            if verbose:
                print_summary(summary)
                
            if args.verify:
                verify_outputs(str(thresh_output_dir))
        
        # Save combined summary
        combined_summary = {
            'thresholds': args.thresholds,
            'summaries': all_summaries
        }
        combined_meta_file = Path(args.output_dir) / 'meta_all_thresholds.json'
        with open(combined_meta_file, 'w') as f:
            json.dump(combined_summary, f, indent=2)
        
        if verbose:
            print(f"\nCombined metadata saved to: {combined_meta_file}")
    
    else:
        # Single threshold
        summary = generate_hard_labels(
            args.teacher_probs_dir,
            args.output_dir,
            args.threshold,
            verbose=verbose
        )
        
        if verbose:
            print_summary(summary)
        
        if args.verify:
            verify_outputs(args.output_dir)


if __name__ == '__main__':
    main()
