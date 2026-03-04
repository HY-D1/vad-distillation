#!/usr/bin/env python3
"""
Build TORGO dataset manifest for sentence-level audio files.

Usage:
    python scripts/build_torgo_manifest.py \
        --data_dir data/torgo_raw \
        --output manifests/torgo_sentences.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def find_audio_files(data_dir: str) -> list[dict]:
    """
    Find all .wav files in TORGO structure.
    
    Expected structure:
        data_dir/
        ├── F01/
        │   ├── Session1/
        │   │   ├── wav_headMic/
        │   │   └── ...
    
    Returns list of dicts with speaker_id, session, utt_id, path
    """
    data_path = Path(data_dir)
    audio_files = []
    
    if not data_path.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    # Walk through speaker directories
    for speaker_dir in sorted(data_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
            
        speaker_id = speaker_dir.name
        
        # Walk through sessions
        for session_dir in sorted(speaker_dir.iterdir()):
            if not session_dir.is_dir():
                continue
                
            session = session_dir.name
            
            # Look for wav_headMic directory
            wav_dir = session_dir / "wav_headMic"
            if not wav_dir.exists():
                continue
            
            # Find all .wav files
            for wav_file in sorted(wav_dir.glob("*.wav")):
                utt_id = wav_file.stem
                
                audio_files.append({
                    'speaker_id': speaker_id,
                    'session': session,
                    'utt_id': utt_id,
                    'path': str(wav_file.relative_to(Path.cwd())),
                    'duration': None,  # To be filled
                    'text': None,      # To be filled if transcript available
                })
    
    return audio_files


def write_manifest(audio_files: list[dict], output_path: str):
    """Write manifest to CSV."""
    if not audio_files:
        print("Warning: No audio files found")
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['speaker_id', 'session', 'utt_id', 'path', 'duration', 'text']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audio_files)
    
    print(f"Manifest written to {output_path}")
    print(f"Total utterances: {len(audio_files)}")
    
    # Print speaker stats
    speakers = sorted(set(f['speaker_id'] for f in audio_files))
    print(f"Speakers: {len(speakers)}")
    for speaker in speakers:
        count = sum(1 for f in audio_files if f['speaker_id'] == speaker)
        print(f"  {speaker}: {count} utterances")


def main():
    parser = argparse.ArgumentParser(
        description="Build TORGO dataset manifest"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/torgo_raw",
        help="Path to TORGO data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="manifests/torgo_sentences.csv",
        help="Output CSV path"
    )
    
    args = parser.parse_args()
    
    print(f"Scanning {args.data_dir}...")
    audio_files = find_audio_files(args.data_dir)
    write_manifest(audio_files, args.output)


if __name__ == "__main__":
    main()
