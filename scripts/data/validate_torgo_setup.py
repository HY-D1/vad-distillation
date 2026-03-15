#!/usr/bin/env python3
"""
Validate TORGO dataset setup.

This script checks:
1. If TORGO data exists in the correct structure
2. What files/directories are missing
3. Validates that a sample of audio files can be loaded

Usage:
    python scripts/validate_torgo_setup.py [--data_dir data/torgo_raw]
"""

import argparse
import sys
from pathlib import Path


try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


def check_directory_structure(data_dir: Path) -> dict:
    """
    Check if TORGO directory structure exists.
    
    Expected structure:
        data_dir/
        ├── F01/
        │   ├── Session1/
        │   │   ├── wav_headMic/
        │   │   └── prompts/
        │   └── ...
        └── ...
    
    Returns:
        Dictionary with structure check results
    """
    results = {
        'exists': False,
        'speakers': [],
        'sessions': {},
        'audio_dirs': [],
        'prompt_dirs': [],
        'wav_files': [],
        'missing_components': []
    }
    
    if not data_dir.exists():
        results['missing_components'].append(f"Data directory not found: {data_dir}")
        return results
    
    results['exists'] = True
    
    # Look for speaker directories
    speaker_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    
    if not speaker_dirs:
        results['missing_components'].append("No speaker directories found")
        return results
    
    results['speakers'] = [d.name for d in speaker_dirs]
    
    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        results['sessions'][speaker_id] = []
        
        # Look for session directories
        session_dirs = [d for d in sorted(speaker_dir.iterdir()) if d.is_dir()]
        
        if not session_dirs:
            results['missing_components'].append(
                f"{speaker_id}: No session directories found"
            )
            continue
        
        for session_dir in session_dirs:
            session_name = session_dir.name
            results['sessions'][speaker_id].append(session_name)
            
            # Check for wav_headMic
            wav_dir = session_dir / "wav_headMic"
            if wav_dir.exists():
                results['audio_dirs'].append(str(wav_dir.relative_to(data_dir)))
                wav_files = [f for f in wav_dir.iterdir() if f.suffix.lower() == '.wav']
                results['wav_files'].extend([
                    str(f.relative_to(data_dir)) for f in wav_files
                ])
            else:
                results['missing_components'].append(
                    f"{speaker_id}/{session_name}: wav_headMic/ not found"
                )
            
            # Check for prompts (optional but recommended)
            prompts_dir = session_dir / "prompts"
            if prompts_dir.exists():
                results['prompt_dirs'].append(str(prompts_dir.relative_to(data_dir)))
    
    return results


def validate_audio_sample(wav_files: list[str], data_dir: Path, 
                          sample_size: int = 5) -> dict:
    """
    Validate that a sample of audio files can be loaded.
    
    Args:
        wav_files: List of relative paths to wav files
        data_dir: Base data directory
        sample_size: Number of files to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'checked': 0,
        'valid': 0,
        'corrupt': [],
        'can_load': False
    }
    
    if not SOUNDFILE_AVAILABLE:
        results['error'] = "soundfile not available, cannot validate audio files"
        return results
    
    if not wav_files:
        results['error'] = "No wav files to validate"
        return results
    
    # Sample files evenly from the list
    if len(wav_files) <= sample_size:
        sample = wav_files
    else:
        indices = [int(i * len(wav_files) / sample_size) for i in range(sample_size)]
        sample = [wav_files[i] for i in indices]
    
    for wav_file in sample:
        wav_path = data_dir / wav_file
        results['checked'] += 1
        
        try:
            info = sf.info(str(wav_path))
            if info.frames == 0:
                results['corrupt'].append({
                    'file': wav_file,
                    'error': 'Empty audio file (0 frames)'
                })
            elif info.samplerate == 0:
                results['corrupt'].append({
                    'file': wav_file,
                    'error': 'Invalid sample rate (0)'
                })
            else:
                results['valid'] += 1
        except Exception as e:
            results['corrupt'].append({
                'file': wav_file,
                'error': str(e)
            })
    
    results['can_load'] = results['valid'] > 0 and len(results['corrupt']) == 0
    return results


def print_report(structure_results: dict, audio_results: dict, data_dir: Path):
    """Print validation report."""
    print("=" * 60)
    print("TORGO DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"\nData directory: {data_dir.absolute()}")
    
    # Structure summary
    print(f"\n{'─' * 60}")
    print("DIRECTORY STRUCTURE")
    print("─" * 60)
    
    if not structure_results['exists']:
        print("❌ Data directory does not exist")
        print("\nTo set up TORGO data:")
        print("  1. Download from https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html")
        print(f"  2. Extract to: {data_dir}")
        print("  3. Ensure structure: data/torgo_raw/<speaker>/<session>/wav_headMic/")
        return
    
    print(f"✓ Data directory exists")
    print(f"\nSpeakers found: {len(structure_results['speakers'])}")
    for speaker in structure_results['speakers']:
        sessions = structure_results['sessions'].get(speaker, [])
        print(f"  - {speaker}: {len(sessions)} session(s)")
    
    print(f"\nAudio directories: {len(structure_results['audio_dirs'])}")
    print(f"WAV files found: {len(structure_results['wav_files'])}")
    
    if structure_results['prompt_dirs']:
        print(f"Prompt directories: {len(structure_results['prompt_dirs'])}")
    else:
        print("⚠ No prompt directories found (transcripts unavailable)")
    
    # Missing components
    if structure_results['missing_components']:
        print(f"\n⚠ Missing/Issues ({len(structure_results['missing_components'])}):")
        for issue in structure_results['missing_components'][:10]:
            print(f"  - {issue}")
        if len(structure_results['missing_components']) > 10:
            print(f"  ... and {len(structure_results['missing_components']) - 10} more")
    
    # Audio validation
    if structure_results['wav_files']:
        print(f"\n{'─' * 60}")
        print("AUDIO FILE VALIDATION")
        print("─" * 60)
        
        if 'error' in audio_results:
            print(f"⚠ {audio_results['error']}")
        else:
            print(f"Sample checked: {audio_results['checked']} files")
            print(f"Valid: {audio_results['valid']}")
            print(f"Corrupt: {len(audio_results['corrupt'])}")
            
            if audio_results['can_load']:
                print("✓ Audio files can be loaded successfully")
            else:
                print("❌ Some audio files could not be loaded")
            
            if audio_results['corrupt']:
                print("\nCorrupt files:")
                for item in audio_results['corrupt']:
                    print(f"  - {item['file']}: {item['error']}")
    
    # Overall status
    print(f"\n{'=' * 60}")
    print("OVERALL STATUS")
    print("=" * 60)
    
    all_good = (
        structure_results['exists'] and
        len(structure_results['speakers']) > 0 and
        len(structure_results['wav_files']) > 0 and
        audio_results.get('can_load', False)
    )
    
    if all_good:
        print("✅ TORGO data is properly set up and ready to use!")
        print("\nNext steps:")
        print("  python scripts/build_torgo_manifest.py")
        print("  jupyter notebook notebooks/eda_torgo_sentences.ipynb")
    elif structure_results['exists'] and len(structure_results['wav_files']) > 0:
        print("⚠️  TORGO data exists but may have issues")
        print("\nRecommendations:")
        print("  - Review missing components listed above")
        print("  - Consider re-downloading corrupt files")
    else:
        print("❌ TORGO data is not set up")
        print("\nTo download TORGO:")
        print("  1. Visit: https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html")
        print("  2. Complete the license agreement")
        print("  3. Download and extract to data/torgo_raw/")
        print("\nExpected structure:")
        print("""
    data/torgo_raw/
    ├── F01/
    │   ├── Session1/
    │   │   ├── wav_headMic/
    │   │   │   ├── 001.wav
    │   │   │   └── ...
    │   │   └── prompts/
    │   │       ├── 001.txt
    │   │       └── ...
    │   └── ...
    ├── F03/
    └── ...
        """)


def main():
    parser = argparse.ArgumentParser(
        description="Validate TORGO dataset setup"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/torgo_raw",
        help="Path to TORGO data directory"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5,
        help="Number of audio files to validate"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Check structure
    print("Checking directory structure...")
    structure_results = check_directory_structure(data_dir)
    
    # Validate audio files
    audio_results = {}
    if structure_results['wav_files']:
        print("Validating audio files...")
        audio_results = validate_audio_sample(
            structure_results['wav_files'],
            data_dir,
            args.sample_size
        )
    
    # Print report
    print_report(structure_results, audio_results, data_dir)
    
    # Return exit code
    if not structure_results['exists']:
        return 1
    if structure_results['wav_files'] and not audio_results.get('can_load', False):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
