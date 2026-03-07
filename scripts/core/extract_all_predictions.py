#!/usr/bin/env python3
"""
Extract all frame_probs from fold_*_predictions.npz files.

This script processes all fold prediction files and extracts them to
individual .npy files organized by fold for comparison with baselines.

Usage:
    python scripts/core/extract_all_predictions.py
    python scripts/core/extract_all_predictions.py --results-dir outputs/production_cuda --output-dir outputs/our_model
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_predictions(predictions_npz, output_dir):
    """
    Extract .npy files from predictions.npz with frame-level data grouped by utterance.
    
    Args:
        predictions_npz: Path to predictions.npz file
        output_dir: Output directory for frame_probs/
        
    Returns:
        tuple: (number of extracted files, list of extracted utterance IDs)
    """
    print(f"  Loading predictions from: {predictions_npz}")
    data = np.load(predictions_npz)
    
    # Expected keys: predictions, labels, probs, utt_ids (all frame-level)
    probs = data['probs']
    utt_ids = data['utt_ids']
    
    print(f"  Total frames: {len(probs)}")
    unique_utts = set(utt_ids)
    print(f"  Unique utterances: {len(unique_utts)}")
    
    # Create output directory
    frame_probs_dir = Path(output_dir) / 'frame_probs'
    frame_probs_dir.mkdir(parents=True, exist_ok=True)
    
    # Group frames by utterance ID
    utt_frames = defaultdict(list)
    for prob, utt_id in zip(probs, utt_ids):
        utt_frames[utt_id].append(prob)
    
    # Save individual .npy files (named exactly as utt_id from predictions)
    extracted = 0
    extracted_utts = []
    for utt_id, frames in utt_frames.items():
        # Sanitize filename (remove problematic characters)
        safe_id = str(utt_id).replace('/', '_').replace('\\', '_')
        prob_array = np.array(frames, dtype=np.float32)
        output_path = frame_probs_dir / f"{safe_id}.npy"
        np.save(output_path, prob_array)
        extracted += 1
        extracted_utts.append(safe_id)
    
    print(f"  Extracted {extracted} .npy files to: {frame_probs_dir}")
    return extracted, extracted_utts


def extract_all_folds(results_dir, output_dir):
    """
    Extract predictions from all fold_*_predictions.npz files.
    
    Args:
        results_dir: Directory containing logs/ with fold_*_predictions.npz files
        output_dir: Base output directory for extracted files
        
    Returns:
        dict: Summary with fold mappings and statistics
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    
    # Find logs directory
    logs_dir = results_path / 'logs'
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return None
    
    # Find all prediction files
    pred_files = sorted(logs_dir.glob('fold_*_predictions.npz'))
    if not pred_files:
        print(f"Error: No fold_*_predictions.npz files found in {logs_dir}")
        return None
    
    print(f"Found {len(pred_files)} prediction files")
    print("=" * 60)
    
    # Summary data
    summary = {
        'source_dir': str(results_dir),
        'output_dir': str(output_dir),
        'folds': {},
        'total_utterances': 0
    }
    
    # Process each fold
    for pred_file in pred_files:
        # Extract fold ID from filename (e.g., fold_F01_predictions.npz -> F01)
        fold_id = pred_file.stem.replace('fold_', '').replace('_predictions', '')
        print(f"\nProcessing fold: {fold_id}")
        
        # Create fold-specific output directory
        fold_output_dir = output_path / f"fold_{fold_id}"
        
        # Extract predictions
        try:
            num_extracted, utterances = extract_predictions(pred_file, fold_output_dir)
            
            summary['folds'][fold_id] = {
                'source_file': str(pred_file.relative_to(results_path)),
                'output_dir': str(fold_output_dir.relative_to(output_path.parent if output_path.parent.exists() else Path('.'))),
                'num_utterances': num_extracted,
                'utterances': utterances
            }
            summary['total_utterances'] += num_extracted
            
        except Exception as e:
            print(f"  Error processing {pred_file}: {e}")
            summary['folds'][fold_id] = {
                'error': str(e)
            }
    
    # Save summary JSON
    print("\n" + "=" * 60)
    summary_path = output_path / 'extraction_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract all frame_probs from fold predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default extraction
  python scripts/core/extract_all_predictions.py
  
  # Custom directories
  python scripts/core/extract_all_predictions.py \\
      --results-dir outputs/production_cuda \\
      --output-dir outputs/our_model
  
  # Dry run (show what would be done)
  python scripts/core/extract_all_predictions.py --dry-run
        """
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='outputs/production_cuda',
        help='Directory containing logs/ with fold_*_predictions.npz files (default: outputs/production_cuda)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/our_model',
        help='Base output directory for extracted files (default: outputs/our_model)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without extracting files'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        results_path = Path(args.results_dir)
        logs_dir = results_path / 'logs'
        pred_files = list(logs_dir.glob('fold_*_predictions.npz'))
        print(f"Would process {len(pred_files)} files from {logs_dir}")
        for f in pred_files:
            fold_id = f.stem.replace('fold_', '').replace('_predictions', '')
            print(f"  - {f.name} -> {args.output_dir}/fold_{fold_id}/frame_probs/")
        return
    
    print("=" * 60)
    print("Frame Probabilities Extraction Pipeline")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    summary = extract_all_folds(args.results_dir, args.output_dir)
    
    if summary:
        print(f"\n{'=' * 60}")
        print("EXTRACTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total folds processed: {len(summary['folds'])}")
        print(f"Total utterances extracted: {summary['total_utterances']}")
        print(f"\nOutput structure:")
        print(f"  {args.output_dir}/")
        print(f"  ├── extraction_summary.json")
        for fold_id in sorted(summary['folds'].keys()):
            if 'error' not in summary['folds'][fold_id]:
                num_utts = summary['folds'][fold_id]['num_utterances']
                print(f"  ├── fold_{fold_id}/")
                print(f"  │   └── frame_probs/ ({num_utts} .npy files)")
        print(f"\nReady for comparison with baselines!")
    else:
        print("\nExtraction failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
