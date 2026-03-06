#!/usr/bin/env python3
"""
Create symlinks in teacher_probs_fixed/ for ALL speakers with proper naming.
Maps from: F01_1.npy, F01_2.npy, etc.
Maps to:   F01_Session1_0001.npy, F01_Session1_0002.npy, etc.
"""

import os
import json
import csv
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TEACHER_PROBS = PROJECT_ROOT / "teacher_probs"
TEACHER_PROBS_FIXED = PROJECT_ROOT / "teacher_probs_fixed"
SPLITS_DIR = PROJECT_ROOT / "splits"
MANIFEST = PROJECT_ROOT / "manifests" / "torgo_pilot.csv"

def get_speaker_files(teacher_probs_dir):
    """Get all speaker .npy files grouped by speaker."""
    speaker_files = {}
    for f in os.listdir(teacher_probs_dir):
        if f.endswith('.npy') and '_' in f:
            # Skip files that already have the proper naming format
            if 'Session' in f:
                continue
            # Extract speaker and number (e.g., "F01_1.npy" -> "F01", 1)
            parts = f.replace('.npy', '').split('_')
            if len(parts) == 2 and parts[1].isdigit():
                speaker = parts[0]
                num = int(parts[1])
                if speaker not in speaker_files:
                    speaker_files[speaker] = []
                speaker_files[speaker].append((num, f))
    
    # Sort by number for each speaker
    for speaker in speaker_files:
        speaker_files[speaker].sort(key=lambda x: x[0])
    
    return speaker_files

def load_manifest(manifest_path):
    """Load manifest and create ordered list per speaker."""
    speaker_utterances = {}
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker = row['speaker_id']
            session = row['session']
            utt_id = row['utt_id']
            
            # Create utterance ID in the format used in splits
            # F01 + Session1 + 0001 -> F01_Session1_0001
            utt_name = f"{speaker}_{session}_{int(utt_id):04d}"
            
            if speaker not in speaker_utterances:
                speaker_utterances[speaker] = []
            speaker_utterances[speaker].append(utt_name)
    
    return speaker_utterances

def create_symlinks():
    """Create symlinks for all speakers."""
    # Ensure teacher_probs_fixed exists
    TEACHER_PROBS_FIXED.mkdir(exist_ok=True)
    
    # Get speaker files from teacher_probs
    speaker_files = get_speaker_files(TEACHER_PROBS)
    print(f"Found files for {len(speaker_files)} speakers: {sorted(speaker_files.keys())}")
    
    # Load manifest to get utterance ordering
    speaker_utterances = load_manifest(MANIFEST)
    print(f"Loaded manifest with {len(speaker_utterances)} speakers")
    
    # Track stats
    total_created = 0
    total_missing = 0
    speaker_stats = {}
    
    for speaker in sorted(speaker_files.keys()):
        files = speaker_files[speaker]
        utterances = speaker_utterances.get(speaker, [])
        
        created = 0
        missing = 0
        
        print(f"\nProcessing {speaker}: {len(files)} files, {len(utterances)} utterances in manifest")
        
        # Create symlinks for as many files as we have (up to utterance count)
        for i, (num, src_file) in enumerate(files):
            if i >= len(utterances):
                # More files than expected utterances
                missing += 1
                continue
                
            utt_name = utterances[i]
            src_path = TEACHER_PROBS / src_file
            dst_path = TEACHER_PROBS_FIXED / f"{utt_name}.npy"
            
            # Create symlink
            try:
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                dst_path.symlink_to(src_path.absolute())
                created += 1
            except Exception as e:
                print(f"  Error creating symlink for {src_file}: {e}")
                missing += 1
        
        speaker_stats[speaker] = {'created': created, 'missing': missing, 'total_files': len(files), 'total_utterances': len(utterances)}
        total_created += created
        total_missing += missing
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Speaker':<10} {'Created':>8} {'Missing':>8} {'Files':>8} {'Utter.':>8}")
    print("-"*60)
    for speaker in sorted(speaker_stats.keys()):
        stats = speaker_stats[speaker]
        print(f"{speaker:<10} {stats['created']:>8} {stats['missing']:>8} {stats['total_files']:>8} {stats['total_utterances']:>8}")
    print("-"*60)
    print(f"{'TOTAL':<10} {total_created:>8} {total_missing:>8}")
    print("="*60)
    
    return total_created, speaker_stats

if __name__ == "__main__":
    create_symlinks()
