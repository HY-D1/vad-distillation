#!/usr/bin/env python3
"""
LOSO (Leave-One-Speaker-Out) Split Generator for TORGO Dataset.

Generates train/val/test splits where each fold holds out a different speaker
for testing. Validation speaker is rotated among remaining speakers for
reproducibility.

Usage:
    # Generate splits from manifest
    python scripts/generate_loso_splits.py \
        --manifest manifests/torgo_sentences.csv \
        --output_dir splits/

    # Generate splits for pilot (2-3 speakers only)
    python scripts/generate_loso_splits.py \
        --manifest manifests/torgo_sentences.csv \
        --output_dir splits/ \
        --pilot \
        --pilot_speakers F01 M01 F02

    # Generate from speaker list (no manifest)
    python scripts/generate_loso_splits.py \
        --speakers F01 M01 F02 F03 M02 M03 \
        --output_dir splits/

    # Verify existing splits
    python scripts/generate_loso_splits.py \
        --verify \
        --output_dir splits/
"""

import argparse
import csv
import os
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional


def load_manifest(manifest_path: str) -> tuple[list[dict], dict]:
    """
    Load manifest CSV and return list of utterances and speaker info.
    
    Returns:
        utterances: List of dicts with keys (speaker_id, session, utt_id, ...)
        speaker_stats: Dict mapping speaker_id to list of utterances
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    utterances = []
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            utterances.append(row)
    
    # Group by speaker
    speaker_stats = {}
    for u in utterances:
        speaker = u['speaker_id']
        if speaker not in speaker_stats:
            speaker_stats[speaker] = []
        speaker_stats[speaker].append(u)
    
    return utterances, speaker_stats


def generate_utterance_id(utt: dict) -> str:
    """Generate unique utterance ID from manifest entry."""
    return f"{utt['speaker_id']}_{utt['session']}_{utt['utt_id']}"


def create_loso_folds(
    speaker_stats: dict,
    val_speaker_rotation: Optional[str] = None
) -> list[dict]:
    """
    Create LOSO folds from speaker statistics.
    
    For each speaker:
    - Test = held-out speaker
    - Val = next speaker in sorted list (deterministic rotation)
    - Train = all remaining speakers
    
    Args:
        speaker_stats: Dict mapping speaker_id to list of utterances
        val_speaker_rotation: Optional fixed val speaker (for reproducibility)
        
    Returns:
        List of fold dictionaries
    """
    speakers = sorted(speaker_stats.keys())
    n_speakers = len(speakers)
    
    if n_speakers < 3:
        raise ValueError(
            f"Need at least 3 speakers for LOSO, got {n_speakers}"
        )
    
    folds = []
    
    for i, test_speaker in enumerate(speakers):
        # Test utterances
        test_utts = [
            generate_utterance_id(u) 
            for u in speaker_stats[test_speaker]
        ]
        
        # Validation speaker: next in rotation (deterministic)
        remaining_speakers = [s for s in speakers if s != test_speaker]
        if val_speaker_rotation and val_speaker_rotation in remaining_speakers:
            val_speaker = val_speaker_rotation
        else:
            # Use next speaker in sorted order, wrap around
            val_idx = i % len(remaining_speakers)
            val_speaker = remaining_speakers[val_idx]
        
        val_utts = [
            generate_utterance_id(u)
            for u in speaker_stats[val_speaker]
        ]
        
        # Train speakers: everyone else
        train_speakers = [s for s in speakers 
                         if s not in (test_speaker, val_speaker)]
        train_utts = []
        for spk in train_speakers:
            train_utts.extend([
                generate_utterance_id(u)
                for u in speaker_stats[spk]
            ])
        
        fold = {
            "fold_id": test_speaker,
            "test_speaker": test_speaker,
            "val_speaker": val_speaker,
            "train_speakers": train_speakers,
            "test_utterances": sorted(test_utts),
            "val_utterances": sorted(val_utts),
            "train_utterances": sorted(train_utts),
            "stats": {
                "n_test": len(test_utts),
                "n_val": len(val_utts),
                "n_train": len(train_utts),
                "n_speakers_train": len(train_speakers),
                "n_speakers_val": 1,
                "n_speakers_test": 1,
            }
        }
        folds.append(fold)
    
    return folds


def verify_splits(output_dir: str, verbose: bool = False) -> bool:
    """
    Verify that all splits are speaker-disjoint.
    
    Returns True if all checks pass, False otherwise.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"ERROR: Output directory does not exist: {output_dir}")
        return False
    
    split_files = sorted(output_path.glob("fold_*.json"))
    if not split_files:
        print(f"ERROR: No fold files found in {output_dir}")
        return False
    
    all_passed = True
    
    for split_file in split_files:
        with open(split_file, 'r') as f:
            fold = json.load(f)
        
        fold_id = fold['fold_id']
        test_spk = fold['test_speaker']
        val_spk = fold['val_speaker']
        train_spks = set(fold['train_speakers'])
        
        # Check 1: Test speaker not in val/train
        if test_spk == val_spk:
            print(f"ERROR [{fold_id}]: Test speaker == Val speaker")
            all_passed = False
        if test_spk in train_spks:
            print(f"ERROR [{fold_id}]: Test speaker in train")
            all_passed = False
        
        # Check 2: Val speaker not in train
        if val_spk in train_spks:
            print(f"ERROR [{fold_id}]: Val speaker in train")
            all_passed = False
        
        # Check 3: No utterance overlap
        test_utts = set(fold['test_utterances'])
        val_utts = set(fold['val_utterances'])
        train_utts = set(fold['train_utterances'])
        
        test_val_overlap = test_utts & val_utts
        test_train_overlap = test_utts & train_utts
        val_train_overlap = val_utts & train_utts
        
        if test_val_overlap:
            print(f"ERROR [{fold_id}]: Test/Val overlap: {len(test_val_overlap)} utterances")
            all_passed = False
        if test_train_overlap:
            print(f"ERROR [{fold_id}]: Test/Train overlap: {len(test_train_overlap)} utterances")
            all_passed = False
        if val_train_overlap:
            print(f"ERROR [{fold_id}]: Val/Train overlap: {len(val_train_overlap)} utterances")
            all_passed = False
        
        if verbose:
            stats = fold['stats']
            print(f"✓ {fold_id}: train={stats['n_train']}, val={stats['n_val']}, "
                  f"test={stats['n_test']} utterances")
    
    if all_passed:
        print(f"\n✓ All {len(split_files)} folds verified: speaker-disjoint")
    else:
        print(f"\n✗ Verification FAILED")
    
    return all_passed


def write_folds(folds: list[dict], output_dir: str):
    """Write fold JSON files to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fold in folds:
        fold_id = fold['fold_id']
        output_file = output_path / f"fold_{fold_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(fold, f, indent=2)
        
        print(f"  Written: {output_file}")
    
    # Write summary
    summary = {
        "n_folds": len(folds),
        "fold_ids": [f['fold_id'] for f in folds],
        "total_utterances": sum(f['stats']['n_test'] + f['stats']['n_val'] + 
                                f['stats']['n_train'] for f in folds) // len(folds),
    }
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary written to: {summary_file}")
    print(f"Total folds: {summary['n_folds']}")
    print(f"Estimated unique utterances: {summary['total_utterances']}")


def create_speaker_list_manifest(speakers: list[str], output_path: str):
    """Create a minimal manifest from just speaker names (for testing)."""
    # Create dummy utterances: 200 per speaker (typical TORGO)
    utterances = []
    for speaker in speakers:
        for i in range(200):
            utterances.append({
                'speaker_id': speaker,
                'session': f'Session{(i // 100) + 1}',
                'utt_id': f'{i+1:03d}',
                'path': os.path.join('data', speaker, f'Session{(i // 100) + 1}', f'{i+1:03d}.wav'),
                'duration': None,
                'text': None,
            })
    
    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=utterances[0].keys())
        writer.writeheader()
        writer.writerows(utterances)
    
    print(f"Created dummy manifest: {output_path}")
    print(f"  Speakers: {len(speakers)}")
    print(f"  Utterances per speaker: 200")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LOSO splits for TORGO dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --manifest manifests/torgo_sentences.csv --output_dir splits/
    %(prog)s --manifest manifests/torgo_sentences.csv --output_dir splits/ --pilot
    %(prog)s --speakers F01 M01 F02 F03 M02 M03 --output_dir splits/
    %(prog)s --verify --output_dir splits/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--manifest",
        type=str,
        help="Path to manifest CSV file"
    )
    input_group.add_argument(
        "--speakers",
        nargs='+',
        help="List of speaker IDs (no manifest needed)"
    )
    input_group.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing splits instead of generating"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="splits/",
        help="Output directory for split JSON files"
    )
    
    # Pilot mode
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Generate only pilot subset (2-3 speakers)"
    )
    parser.add_argument(
        "--pilot_speakers",
        nargs='+',
        default=["F01", "M01", "F02"],
        help="Speakers to use in pilot mode (default: F01 M01 F02)"
    )
    
    # Validation speaker selection
    parser.add_argument(
        "--val_speaker",
        type=str,
        default=None,
        help="Fixed validation speaker for all folds (for reproducibility)"
    )
    
    # Verification
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during verification"
    )
    
    args = parser.parse_args()
    
    # Verification mode
    if args.verify:
        success = verify_splits(args.output_dir, args.verbose)
        sys.exit(0 if success else 1)
    
    # Generate mode
    print("=" * 60)
    print("LOSO Split Generator")
    print("=" * 60)
    
    # Determine input source
    if args.speakers:
        # Create temporary manifest from speaker list
        print(f"Using speaker list: {args.speakers}")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            manifest_path = f.name
        create_speaker_list_manifest(args.speakers, manifest_path)
    else:
        manifest_path = args.manifest
        print(f"Using manifest: {manifest_path}")
    
    # Load data
    try:
        utterances, speaker_stats = load_manifest(manifest_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"\nDataset statistics:")
    print(f"  Total speakers: {len(speaker_stats)}")
    print(f"  Total utterances: {len(utterances)}")
    for speaker in sorted(speaker_stats.keys()):
        print(f"    {speaker}: {len(speaker_stats[speaker])} utterances")
    
    # Pilot mode: filter speakers
    if args.pilot:
        pilot_speakers = [s for s in args.pilot_speakers 
                         if s in speaker_stats]
        if len(pilot_speakers) < 2:
            print(f"ERROR: Pilot mode needs at least 2 speakers, found {len(pilot_speakers)}")
            print(f"Available: {list(speaker_stats.keys())}")
            sys.exit(1)
        
        speaker_stats = {s: speaker_stats[s] for s in pilot_speakers}
        print(f"\n[PILOT MODE] Using {len(pilot_speakers)} speakers: {pilot_speakers}")
    
    # Generate folds
    print(f"\nGenerating LOSO folds...")
    folds = create_loso_folds(speaker_stats, args.val_speaker)
    
    # Write outputs
    print(f"\nWriting splits to: {args.output_dir}")
    write_folds(folds, args.output_dir)
    
    # Verify
    print(f"\nVerifying splits...")
    verify_splits(args.output_dir, verbose=True)
    
    print("\n" + "=" * 60)
    print("LOSO split generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
