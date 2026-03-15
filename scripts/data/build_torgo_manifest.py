#!/usr/bin/env python3
"""
Build TORGO dataset manifest for sentence-level audio files.

This script scans the TORGO dataset, extracts metadata including audio duration
and transcriptions, validates audio files, and outputs a manifest CSV.

Usage:
    python scripts/build_torgo_manifest.py \
        --data_dir data/torgo_raw \
        --output manifests/torgo_sentences.csv

Features:
    - Duration computation using torchaudio
    - Transcript parsing from prompts/ directories
    - Progress bar with tqdm for large datasets
    - Validation to check for corrupt audio files
    - Graceful handling when data is not present
"""

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

# Try importing optional dependencies
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    warnings.warn("soundfile not available. Duration computation will be skipped.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def find_transcript(wav_file: Path) -> tuple[str | None, Path | None]:
    """
    Find and read transcript for a given audio file.
    
    TORGO structure places transcripts in prompts/ directory alongside wav_headMic/.
    
    Args:
        wav_file: Path to the audio file
        
    Returns:
        Tuple of (transcript_text, transcript_path)
    """
    # Expected structure: SessionX/wav_headMic/001.wav -> SessionX/prompts/001.txt
    session_dir = wav_file.parent.parent  # Go up from wav_headMic to SessionX
    prompts_dir = session_dir / "prompts"
    
    if not prompts_dir.exists():
        return None, None
    
    # Look for transcript file with same name as audio
    transcript_file = prompts_dir / f"{wav_file.stem}.txt"
    
    if transcript_file.exists():
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text, transcript_file
        except (IOError, UnicodeDecodeError) as e:
            return None, transcript_file
    
    return None, None


def compute_duration(wav_file: Path) -> float | None:
    """
    Compute audio duration in seconds using soundfile.
    
    Args:
        wav_file: Path to the audio file
        
    Returns:
        Duration in seconds, or None if computation fails
    """
    if not SOUNDFILE_AVAILABLE:
        return None
    
    try:
        info = sf.info(str(wav_file))
        duration = info.frames / info.samplerate
        return duration
    except Exception as e:
        return None


def validate_audio_file(wav_file: Path) -> tuple[bool, str | None]:
    """
    Validate that an audio file can be loaded.
    
    Args:
        wav_file: Path to the audio file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not SOUNDFILE_AVAILABLE:
        # Can't validate without soundfile, assume valid
        return True, None
    
    try:
        info = sf.info(str(wav_file))
        if info.frames == 0:
            return False, "Empty audio file (0 frames)"
        if info.samplerate == 0:
            return False, "Invalid sample rate (0)"
        return True, None
    except Exception as e:
        return False, str(e)


def find_audio_files(data_dir: str, compute_durations: bool = True, 
                     parse_transcripts: bool = True, 
                     validate_files: bool = True) -> list[dict]:
    """
    Find all .wav files in TORGO structure with optional metadata extraction.
    
    Expected structure:
        data_dir/
        ├── F01/
        │   ├── Session1/
        │   │   ├── wav_headMic/
        │   │   │   ├── 001.wav
        │   │   │   └── ...
        │   │   └── prompts/
        │   │       ├── 001.txt
        │   │       └── ...
        │   └── ...
        └── ...
    
    Args:
        data_dir: Root directory of TORGO dataset
        compute_durations: Whether to compute audio durations
        parse_transcripts: Whether to parse transcript files
        validate_files: Whether to validate audio files
        
    Returns:
        List of dicts with speaker_id, session, utt_id, path, duration, text
    """
    data_path = Path(data_dir)
    audio_files = []
    corrupt_files = []
    
    if not data_path.exists():
        print(f"Warning: Data directory {data_dir} does not exist")
        print("Returning empty manifest. Run this script again after downloading TORGO.")
        return audio_files
    
    # Collect all wav files first for progress bar
    wav_files_to_process = []
    for speaker_dir in sorted(data_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
            
        speaker_id = speaker_dir.name.upper()  # Normalize to uppercase
        
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
            for wav_file in sorted([f for f in wav_dir.iterdir() if f.suffix.lower() == '.wav']):
                wav_files_to_process.append({
                    'speaker_id': speaker_id,
                    'session': session,
                    'wav_file': wav_file
                })
    
    # Process files with progress bar if available
    iterator = tqdm(wav_files_to_process, desc="Processing audio files") if TQDM_AVAILABLE else wav_files_to_process
    
    for item in iterator:
        speaker_id = item['speaker_id']
        session = item['session']
        wav_file = item['wav_file']
        utt_id = wav_file.stem
        
        # Validate audio file
        is_valid, error_msg = True, None
        if validate_files and SOUNDFILE_AVAILABLE:
            is_valid, error_msg = validate_audio_file(wav_file)
        
        if not is_valid:
            corrupt_files.append({
                'path': str(wav_file),
                'error': error_msg
            })
            continue
        
        # Compute duration
        duration = None
        if compute_durations:
            duration = compute_duration(wav_file)
        
        # Parse transcript
        text = None
        transcript_path = None
        if parse_transcripts:
            text, transcript_path = find_transcript(wav_file)
        
        audio_files.append({
            'speaker_id': speaker_id,
            'session': session,
            'utt_id': utt_id,
            'path': str(wav_file),
            'duration': duration,
            'text': text,
            'transcript_path': str(transcript_path) if transcript_path else None,
        })
    
    # Report corrupt files
    if corrupt_files:
        print(f"\nWarning: Found {len(corrupt_files)} corrupt audio files:")
        for cf in corrupt_files[:10]:  # Show first 10
            print(f"  - {cf['path']}: {cf['error']}")
        if len(corrupt_files) > 10:
            print(f"  ... and {len(corrupt_files) - 10} more")
    
    return audio_files


def write_manifest(audio_files: list[dict], output_path: str):
    """
    Write manifest to CSV.
    
    Args:
        audio_files: List of audio file metadata dicts
        output_path: Path to output CSV file
    """
    if not audio_files:
        print("Warning: No audio files found")
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['speaker_id', 'session', 'utt_id', 'path', 'duration', 'text']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, 
                               extrasaction='ignore')  # Ignore extra fields like transcript_path
        writer.writeheader()
        writer.writerows(audio_files)
    
    print(f"\nManifest written to {output_path}")
    print(f"Total utterances: {len(audio_files)}")
    
    # Print speaker stats
    speakers = sorted(set(f['speaker_id'] for f in audio_files))
    print(f"Speakers: {len(speakers)}")
    for speaker in speakers:
        count = sum(1 for f in audio_files if f['speaker_id'] == speaker)
        durations = [f['duration'] for f in audio_files 
                    if f['speaker_id'] == speaker and f['duration'] is not None]
        total_duration = sum(durations) if durations else 0
        duration_str = f"{total_duration/3600:.1f}h" if durations else "N/A"
        print(f"  {speaker}: {count} utterances, {duration_str}")
    
    # Print duration stats if available
    all_durations = [f['duration'] for f in audio_files if f['duration'] is not None]
    if all_durations:
        total_hours = sum(all_durations) / 3600
        print(f"\nTotal duration: {total_hours:.1f} hours")
        print(f"Mean duration: {sum(all_durations)/len(all_durations):.2f}s")
        print(f"Min duration: {min(all_durations):.2f}s")
        print(f"Max duration: {max(all_durations):.2f}s")
    
    # Print transcript coverage
    with_transcripts = sum(1 for f in audio_files if f['text'] is not None)
    if with_transcripts > 0:
        coverage = 100 * with_transcripts / len(audio_files)
        print(f"\nTranscript coverage: {with_transcripts}/{len(audio_files)} ({coverage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Build TORGO dataset manifest with optional duration computation and transcript parsing"
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
    parser.add_argument(
        "--no_duration",
        action="store_true",
        help="Skip duration computation"
    )
    parser.add_argument(
        "--no_transcripts",
        action="store_true",
        help="Skip transcript parsing"
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip audio file validation"
    )
    
    args = parser.parse_args()
    
    print(f"Scanning {args.data_dir}...")
    print(f"  Duration computation: {'disabled' if args.no_duration else 'enabled'}")
    print(f"  Transcript parsing: {'disabled' if args.no_transcripts else 'enabled'}")
    print(f"  File validation: {'disabled' if args.no_validate else 'enabled'}")
    
    audio_files = find_audio_files(
        args.data_dir,
        compute_durations=not args.no_duration,
        parse_transcripts=not args.no_transcripts,
        validate_files=not args.no_validate
    )
    
    write_manifest(audio_files, args.output)


if __name__ == "__main__":
    main()
