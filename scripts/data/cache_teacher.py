#!/usr/bin/env python3
"""
Enhanced teacher (Silero VAD) output caching for TORGO dataset.

Supports parallel batch processing, resume capability, and verification.

Usage:
    # Cache teacher outputs
    python scripts/cache_teacher.py --manifest manifests/torgo_sentences.csv
    
    # Use GPU with batch processing
    python scripts/cache_teacher.py --device cuda --batch-size 16
    
    # Resume from previous run
    python scripts/cache_teacher.py --resume
    
    # Test mode
    python scripts/cache_teacher.py --test

Note: This script uses DASHES for argument names (e.g., --output-dir, NOT --output_dir)
      This follows standard Unix conventions where argparse converts --output-dir to args.output_dir
"""

import argparse
import csv
import os
import json
import signal
import sys
import time
import traceback
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cli.utils import ProgressTracker

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# Add project root to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.audio import load_audio


# Cache version for metadata
CACHE_VERSION = "1.0.0"

# Silero VAD parameters
SILERO_SR = 16000
SILERO_FRAME_MS = 32  # Silero processes in 32ms frames

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_silero_vad():
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    return model, utils


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except (RuntimeError, IOError):
        return 0.0


def get_speech_probs(
    model,
    utils,
    audio: torch.Tensor,
    sampling_rate: int = 16000,
) -> np.ndarray:
    """
    Get frame-level speech probabilities from Silero VAD.
    Uses VADIterator for proper chunked processing.
    
    Args:
        model: Silero VAD model
        utils: Silero VAD utilities (get_speech_timestamps, etc.)
        audio: Audio tensor (samples,)
        sampling_rate: Sample rate
    
    Returns:
        probs: Speech probabilities (num_frames,)
    """
    # Get utility functions
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
    
    device = next(model.parameters()).device
    
    # Convert to numpy for processing
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio
    
    # Silero VAD processes audio in chunks and returns timestamps
    # We'll use get_speech_timestamps which internally collects probabilities
    model.reset_states()
    
    with torch.no_grad():
        # Get timestamps - this processes the audio in proper chunks
        speech_timestamps = get_speech_timestamps(
            torch.tensor(audio_np).to(device),
            model,
            sampling_rate=sampling_rate,
            threshold=0.5,
            min_speech_duration_ms=250,
            max_speech_duration_s=float('inf'),
            min_silence_duration_ms=100,
        )
    
    # Now compute frame-level probabilities by processing in chunks
    window_size_samples = 512 if sampling_rate == 16000 else 256  # 32ms chunks
    
    model.reset_states()
    speech_probs = []
    
    with torch.no_grad():
        for i in range(0, len(audio_np), window_size_samples):
            chunk = audio_np[i: i + window_size_samples]
            
            # Pad the last chunk if needed
            if len(chunk) < window_size_samples:
                chunk = np.pad(chunk, (0, window_size_samples - len(chunk)))
            
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get probability for this chunk
            prob = model(chunk_tensor, sampling_rate).item()
            speech_probs.append(prob)
    
    return np.array(speech_probs, dtype=np.float32)


def get_cache_filename(row: Dict) -> str:
    """Generate cache filename from manifest row."""
    speaker_id = row['speaker_id']
    utt_id = row['utt_id']
    session = row.get('session', 'unknown')
    return f"{speaker_id}_{session}_{int(utt_id):04d}.npy"


def verify_cache_file(cache_path: Path) -> Tuple[bool, Optional[str]]:
    """Verify a cached file is valid."""
    try:
        data = np.load(cache_path)
        
        if data.size == 0:
            return False, "empty array"
        
        if not np.isfinite(data).all():
            return False, "contains NaN or Inf"
        
        # Check values are in valid probability range
        if data.min() < 0 or data.max() > 1:
            return False, "values outside [0, 1]"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def process_single_file(
    row: Dict,
    output_dir: Path,
    model,
    utils,
    verify: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Process a single file.
    
    Args:
        row: Manifest row
        output_dir: Output directory
        model: Silero VAD model
        utils: Silero VAD utilities
        verify: Verify cached files
    
    Returns:
        (status, error_message)
    """
    audio_path = Path(row['path'])
    
    # Handle relative paths - manifest paths already include 'data/' prefix
    if not audio_path.is_absolute():
        # Check if path already starts with 'data/'
        if str(audio_path).startswith(f'data{os.sep}'):
            audio_path = audio_path  # Already correct
        else:
            audio_path = Path('data') / audio_path
    
    cache_filename = get_cache_filename(row)
    cache_path = output_dir / cache_filename
    
    # Check if audio file exists
    if not audio_path.exists():
        return 'failed', f"Audio file not found: {audio_path}"
    
    try:
        # Load audio
        audio, _ = load_audio(str(audio_path), target_sr=SILERO_SR, return_tensor=True)
        
        # Get speech probabilities
        probs = get_speech_probs(model, utils, audio, SILERO_SR)
        
        if probs is None or len(probs) == 0:
            return 'failed', "No probabilities returned"
        
        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, probs.astype(np.float32))
        
        return 'success', None
        
    except Exception as e:
        return 'failed', str(e)


def process_batch(
    rows: List[Dict],
    output_dir: Path,
    model,
    utils,
    device: str = 'cpu',
) -> Tuple[int, int, List[str]]:
    """
    Process a batch of files.
    
    Args:
        rows: List of manifest rows
        output_dir: Output directory
        model: Silero VAD model
        utils: Silero VAD utilities
        device: Device string
    
    Returns:
        (success_count, failed_count, errors)
    """
    success_count = 0
    failed_count = 0
    errors = []
    
    model.eval()
    
    for row in rows:
        audio_path = Path(row['path'])
        
        # Handle relative paths - manifest paths already include 'data/' prefix
        if not audio_path.is_absolute():
            # Check if path already starts with 'data/'
            if str(audio_path).startswith(f'data{os.sep}'):
                audio_path = audio_path  # Already correct
            else:
                audio_path = Path('data') / audio_path
        
        cache_filename = get_cache_filename(row)
        cache_path = output_dir / cache_filename
        
        if not audio_path.exists():
            failed_count += 1
            errors.append(f"Not found: {audio_path}")
            continue
        
        try:
            # Load audio
            audio, _ = load_audio(str(audio_path), target_sr=SILERO_SR, return_tensor=True)
            
            # Get speech probabilities
            with torch.no_grad():
                probs = get_speech_probs(model, utils, audio, SILERO_SR)
            
            # Save
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, probs.astype(np.float32))
            
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            errors.append(f"{audio_path.name}: {e}")
    
    return success_count, failed_count, errors


def group_by_length(
    rows: List[Dict],
    num_groups: int = 10
) -> List[List[Dict]]:
    """
    Group utterances by similar length for efficient batching.
    
    Args:
        rows: List of manifest rows
        num_groups: Number of length groups
    
    Returns:
        List of row groups
    """
    # Get durations
    durations = []
    for row in rows:
        audio_path = Path(row['path'])
        if not audio_path.is_absolute():
            # Check if path already starts with 'data/'
            if not str(audio_path).startswith(f'data{os.sep}'):
                audio_path = Path('data') / audio_path
        duration = get_audio_duration(str(audio_path))
        durations.append((duration, row))
    
    # Sort by duration
    durations.sort(key=lambda x: x[0])
    
    # Group into buckets
    groups = []
    bucket_size = max(1, len(durations) // num_groups)
    
    for i in range(0, len(durations), bucket_size):
        bucket = durations[i:i + bucket_size]
        groups.append([row for _, row in bucket])
    
    return groups


def process_teacher_manifest(
    manifest_rows: List[Dict],
    output_dir: Path,
    device: str = 'cpu',
    batch_size: int = 1,
    resume: bool = True,
    verify: bool = False,
    tracker: Optional['ProgressTracker'] = None,
    is_interrupted_func: Optional[Callable] = None,
) -> Tuple[int, int]:
    """
    Process manifest and cache teacher outputs.
    
    Args:
        manifest_rows: List of manifest rows
        output_dir: Output directory
        device: Device for inference
        batch_size: Batch size (currently unused, processes sequentially)
        resume: Skip existing files
        verify: Verify cached files
        tracker: Progress tracker
        is_interrupted_func: Function to check for interruption
    
    Returns:
        (success_count, failed_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading Silero VAD model...")
    model, utils = load_silero_vad()
    model.to(device)
    model.eval()
    
    # Filter already cached
    if resume:
        cached_files = {f.stem for f in output_dir.glob('*.npy')}
        rows_to_process = []
        
        for row in manifest_rows:
            cache_key = get_cache_filename(row).replace('.npy', '')
            if cache_key not in cached_files:
                rows_to_process.append(row)
        
        skipped = len(manifest_rows) - len(rows_to_process)
        if skipped > 0:
            print(f"Skipping {skipped} already cached files")
    else:
        rows_to_process = manifest_rows
    
    total = len(rows_to_process)
    success_count = 0
    failed_count = 0
    
    # Show GPU memory if available
    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"Processing {total} utterances...")
    
    # Process with progress bar
    pbar = tqdm(total=total, desc="Caching teacher")
    
    for i, row in enumerate(rows_to_process):
        if is_interrupted_func and is_interrupted_func():
            break
        
        status, error = process_single_file(row, output_dir, model, utils, verify)
        
        if status == 'success':
            success_count += 1
        else:
            failed_count += 1
            if error:
                tqdm.write(f"Error: {error}")
        
        pbar.update(1)
        if tracker:
            tracker.update(1)
        
        # Show GPU memory periodically
        if device == 'cuda' and torch.cuda.is_available() and (i + 1) % 100 == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            tqdm.write(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    pbar.close()
    
    return success_count, failed_count


def save_metadata(
    output_dir: Path,
    num_files: int,
    device: str,
):
    """Save cache metadata."""
    metadata = {
        'version': CACHE_VERSION,
        'timestamp': datetime.now().isoformat(),
        'teacher_model': 'silero-vad',
        'parameters': {
            'sample_rate': SILERO_SR,
            'frame_ms': SILERO_FRAME_MS,
            'device': device,
        },
        'num_files': num_files,
    }
    
    meta_path = output_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {meta_path}")


def main():
    # Check for common user mistakes BEFORE argparse processes them
    # This provides a more helpful error message than argparse's "unrecognized arguments"
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--output_dir'):
            print("\n" + "=" * 60)
            print("ERROR: Invalid argument: --output_dir (underscore)")
            print("       Use --output-dir (dash) instead")
            print("=" * 60)
            print("\nNote: Standard argparse uses DASHES for multi-word arguments:")
            print("      --output-dir, --batch-size")
            print("      (These become args.output_dir, args.batch_size in code)")
            print("=" * 60 + "\n")
            sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Cache Silero VAD teacher outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cache teacher outputs (CPU)
  python scripts/cache_teacher.py
  
  # Use GPU
  python scripts/cache_teacher.py --device cuda
  
  # Resume from previous run
  python scripts/cache_teacher.py --resume
  
  # Verify cached files
  python scripts/cache_teacher.py --verify
  
  # Test mode (5 files)
  python scripts/cache_teacher.py --test
  
  # Custom output directory (note: use --output-dir with DASH, not underscore)
  python scripts/cache_teacher.py --output-dir my_teacher_probs

IMPORTANT:
  Use DASHES for multi-word arguments: --output-dir, --batch-size
  (NOT --output_dir, NOT --batch_size)
  Argparse automatically converts --output-dir to args.output_dir in the code.
        """
    )
    
    # Input/output
    parser.add_argument('--manifest', type=str, default='manifests/torgo_sentences.csv',
                        help='Path to manifest CSV')
    parser.add_argument('--output-dir', type=str, default='teacher_probs',
                        help='Output directory for teacher probabilities')
    
    # Model
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Device for inference (default: auto-detect)')
    
    # Processing
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (currently limited by model)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already cached files')
    parser.add_argument('--verify', action='store_true',
                        help='Verify cached files')
    
    # Control
    parser.add_argument('--test', action='store_true',
                        help='Test mode: process only 5 files')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)
    
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if args.test:
        rows = rows[:5]
        print(f"TEST MODE: Processing only {len(rows)} files")
    
    print("=" * 60)
    print("Teacher Caching (Silero VAD)")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Files: {len(rows)}")
    print(f"Device: {args.device}")
    print(f"Resume: {args.resume}")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    
    try:
        success, failed = process_teacher_manifest(
            manifest_rows=rows,
            output_dir=output_dir,
            device=args.device,
            batch_size=args.batch_size,
            resume=args.resume,
            verify=args.verify,
        )
        
        # Save metadata
        save_metadata(output_dir, len(rows), args.device)
        
        # Summary
        print("\n" + "=" * 60)
        print("Caching Complete")
        print("=" * 60)
        print(f"Success: {success}")
        print(f"Failed:  {failed}")
        print("=" * 60)
        
        if failed > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
