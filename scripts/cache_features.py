#!/usr/bin/env python3
"""
Enhanced feature caching for TORGO dataset.

Caches mel spectrograms, MFCCs, and raw audio with parallel processing,
resume capability, and verification.

Usage:
    # Cache mel spectrograms
    python scripts/cache_features.py --manifest manifests/torgo_sentences.csv
    
    # Cache multiple features in parallel
    python scripts/cache_features.py --features mel mfcc --parallel 4
    
    # Resume from previous run
    python scripts/cache_features.py --resume
    
    # Verify cached files
    python scripts/cache_features.py --verify
    
    # Test mode (5 files only)
    python scripts/cache_features.py --test
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


# Cache version for metadata
CACHE_VERSION = "1.0.0"

# Default parameters
DEFAULTS = {
    'sample_rate': 16000,
    'n_mels': 40,
    'n_mfcc': 13,
    'n_fft': 512,
    'hop_length': 160,  # 10ms at 16kHz
    'win_length': 400,  # 25ms at 16kHz
}

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate
    
    Returns:
        audio: Audio array (samples,)
        sr: Sample rate
    """
    try:
        waveform, orig_sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0).numpy(), target_sr
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {audio_path}: {e}")


def compute_melspectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Compute mel spectrogram from audio.
    
    Args:
        audio: Audio array (samples,)
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length
        win_length: Window length
    
    Returns:
        mel: Mel spectrogram (time, n_mels)
    """
    # Use torchaudio for consistency
    waveform = torch.from_numpy(audio).unsqueeze(0)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    
    mel = mel_transform(waveform)
    
    # Convert to log scale (dB)
    mel = torchaudio.transforms.AmplitudeToDB(st_ref=1.0, top_db=80.0)(mel)
    
    # Transpose to (time, n_mels)
    mel = mel.squeeze(0).numpy().T
    
    return mel.astype(np.float32)


def compute_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Compute MFCC features from audio.
    
    Args:
        audio: Audio array (samples,)
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length
        win_length: Window length
    
    Returns:
        mfcc: MFCC features (time, n_mfcc)
    """
    waveform = torch.from_numpy(audio).unsqueeze(0)
    
    # Mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=40,  # Standard for MFCC
        power=2.0,
    )
    
    mel = mel_transform(waveform)
    
    # Convert to dB
    mel = torchaudio.transforms.AmplitudeToDB(st_ref=1.0, top_db=80.0)(mel)
    
    # MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        log_mels=False,
        melkwargs={
            'n_fft': n_fft,
            'win_length': win_length,
            'hop_length': hop_length,
            'n_mels': 40,
        }
    )
    
    mfcc = mfcc_transform(waveform)
    
    return mfcc.squeeze(0).numpy().T.astype(np.float32)


def compute_features(
    audio_path: str,
    feature_type: str,
    params: Dict,
) -> Optional[np.ndarray]:
    """
    Compute features for an audio file.
    
    Args:
        audio_path: Path to audio file
        feature_type: Type of features ('mel', 'mfcc', 'raw')
        params: Feature parameters
    
    Returns:
        features: Computed features or None if failed
    """
    try:
        # Load audio
        audio, sr = load_audio(audio_path, params['sample_rate'])
        
        if feature_type == 'mel':
            return compute_melspectrogram(
                audio, sr,
                n_mels=params['n_mels'],
                n_fft=params['n_fft'],
                hop_length=params['hop_length'],
                win_length=params['win_length'],
            )
        elif feature_type == 'mfcc':
            return compute_mfcc(
                audio, sr,
                n_mfcc=params['n_mfcc'],
                n_fft=params['n_fft'],
                hop_length=params['hop_length'],
                win_length=params['win_length'],
            )
        elif feature_type == 'raw':
            return audio.astype(np.float32)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to compute {feature_type} for {audio_path}: {e}")


def get_cache_filename(row: Dict) -> str:
    """Generate cache filename from manifest row."""
    speaker_id = row['speaker_id']
    utt_id = row['utt_id']
    session = row.get('session', 'unknown')
    return f"{speaker_id}_{session}_{int(utt_id):04d}.npy"


def should_recompute(
    audio_path: Path,
    cache_path: Path,
    check_modified: bool = True
) -> bool:
    """
    Check if cache should be recomputed.
    
    Args:
        audio_path: Path to source audio file
        cache_path: Path to cached file
        check_modified: Whether to check if source is newer than cache
    
    Returns:
        True if recomputation is needed
    """
    if not cache_path.exists():
        return True
    
    if check_modified:
        # Check if source is newer than cache
        audio_mtime = audio_path.stat().st_mtime
        cache_mtime = cache_path.stat().st_mtime
        if audio_mtime > cache_mtime:
            return True
    
    return False


def verify_cache_file(cache_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Verify a cached file is valid.
    
    Args:
        cache_path: Path to cached file
    
    Returns:
        (is_valid, error_message)
    """
    try:
        data = np.load(cache_path)
        
        # Check for empty array
        if data.size == 0:
            return False, "empty array"
        
        # Check for NaN/Inf
        if not np.isfinite(data).all():
            return False, "contains NaN or Inf"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def process_single_file(
    row: Dict,
    output_dir: Path,
    feature_type: str,
    params: Dict,
    resume: bool = True,
    verify: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Process a single file (for parallel execution).
    
    Args:
        row: Manifest row
        output_dir: Output directory
        feature_type: Type of features
        params: Feature parameters
        resume: Skip existing files
        verify: Verify cached files
    
    Returns:
        (status, error_message)
        status: 'success', 'skipped', 'failed'
    """
    audio_path = Path(row['path'])
    
    # Ensure path is relative to project root
    # If path already starts with 'data/', use it as-is
    # Otherwise, prepend 'data/' for backward compatibility
    if not audio_path.is_absolute():
        path_str = str(audio_path)
        if not path_str.startswith('data/'):
            audio_path = Path('data') / audio_path
    
    cache_filename = get_cache_filename(row)
    cache_path = output_dir / cache_filename
    
    # Check if already cached
    if resume and cache_path.exists():
        if not should_recompute(audio_path, cache_path):
            if verify:
                is_valid, error = verify_cache_file(cache_path)
                if is_valid:
                    return 'skipped', None
                else:
                    # Invalid cache, delete and recompute
                    try:
                        cache_path.unlink()
                    except:
                        pass
            else:
                return 'skipped', None
    
    # Check if audio file exists
    if not audio_path.exists():
        return 'failed', f"Audio file not found: {audio_path}"
    
    try:
        # Compute features
        features = compute_features(str(audio_path), feature_type, params)
        
        if features is None:
            return 'failed', "Feature computation returned None"
        
        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
        
        return 'success', None
        
    except Exception as e:
        return 'failed', str(e)


def init_worker():
    """Initialize worker process (ignore SIGINT)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_manifest_parallel(
    manifest_rows: List[Dict],
    output_dir: Path,
    feature_type: str,
    parallel: int = 1,
    resume: bool = True,
    verify: bool = False,
    params: Optional[Dict] = None,
    tracker: Optional['ProgressTracker'] = None,
    is_interrupted_func: Optional[Callable] = None,
) -> Tuple[int, int]:
    """
    Process manifest with parallel workers.
    
    Args:
        manifest_rows: List of manifest rows
        output_dir: Output directory
        feature_type: Type of features
        parallel: Number of parallel workers
        resume: Skip existing files
        verify: Verify cached files
        params: Feature parameters
        tracker: Progress tracker (optional)
        is_interrupted_func: Function to check for interruption
    
    Returns:
        (success_count, failed_count)
    """
    if params is None:
        params = DEFAULTS.copy()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(manifest_rows)
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Prepare arguments for parallel processing
    process_args = [
        (row, output_dir, feature_type, params, resume, verify)
        for row in manifest_rows
    ]
    
    # Process with progress bar
    pbar = tqdm(total=total, desc=f"Caching {feature_type}")
    
    if parallel > 1:
        # Use multiprocessing pool
        pool = mp.Pool(processes=parallel, initializer=init_worker)
        
        try:
            results = pool.starmap_async(process_single_file, process_args)
            
            # Monitor progress
            while not results.ready():
                if is_interrupted_func and is_interrupted_func():
                    pool.terminate()
                    pool.join()
                    break
                
                # Update progress bar
                ready = results._number_left
                completed = total - ready
                pbar.n = completed
                pbar.refresh()
                
                if tracker:
                    tracker.update(completed - tracker.completed)
                
                time.sleep(0.1)
            
            # Get results
            if not is_interrupted_func or not is_interrupted_func():
                results = results.get()
                for status, error in results:
                    if status == 'success':
                        success_count += 1
                    elif status == 'skipped':
                        skipped_count += 1
                    else:
                        failed_count += 1
                        if error:
                            tqdm.write(f"Error: {error}")
            
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        finally:
            pool.close()
            pool.join()
    else:
        # Sequential processing
        for args in process_args:
            if is_interrupted_func and is_interrupted_func():
                break
            
            status, error = process_single_file(*args)
            
            if status == 'success':
                success_count += 1
            elif status == 'skipped':
                skipped_count += 1
            else:
                failed_count += 1
                if error:
                    tqdm.write(f"Error: {error}")
            
            pbar.update(1)
            if tracker:
                tracker.update(1)
    
    pbar.close()
    
    return success_count, failed_count


def save_metadata(
    output_dir: Path,
    feature_types: List[str],
    params: Dict,
    num_files: int,
):
    """Save cache metadata."""
    metadata = {
        'version': CACHE_VERSION,
        'timestamp': datetime.now().isoformat(),
        'feature_types': feature_types,
        'parameters': {
            'sample_rate': params['sample_rate'],
            'n_mels': params.get('n_mels', 40),
            'n_mfcc': params.get('n_mfcc', 13),
            'n_fft': params['n_fft'],
            'hop_length': params['hop_length'],
            'win_length': params['win_length'],
        },
        'num_files': num_files,
    }
    
    meta_path = output_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cache features for TORGO dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cache mel spectrograms
  python scripts/cache_features.py
  
  # Cache mel and MFCC in parallel
  python scripts/cache_features.py --features mel mfcc --parallel 4
  
  # Resume interrupted run
  python scripts/cache_features.py --resume
  
  # Verify cached files
  python scripts/cache_features.py --verify
  
  # Test mode (5 files)
  python scripts/cache_features.py --test
        """
    )
    
    # Input/output
    parser.add_argument('--manifest', type=str, default='manifests/torgo_sentences.csv',
                        help='Path to manifest CSV')
    parser.add_argument('--output-dir', type=str, default='cached_features',
                        help='Output directory for cached features')
    
    # Features
    parser.add_argument('--features', nargs='+', choices=['mel', 'mfcc', 'raw', 'all'],
                        default=['mel'],
                        help='Features to cache (default: mel)')
    
    # Parameters
    parser.add_argument('--n-mels', type=int, default=DEFAULTS['n_mels'],
                        help='Number of mel bins')
    parser.add_argument('--n-mfcc', type=int, default=DEFAULTS['n_mfcc'],
                        help='Number of MFCC coefficients')
    parser.add_argument('--sample-rate', type=int, default=DEFAULTS['sample_rate'],
                        help='Target sample rate')
    
    # Processing
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already cached files')
    parser.add_argument('--verify', action='store_true',
                        help='Verify cached files')
    
    # Control
    parser.add_argument('--test', action='store_true',
                        help='Test mode: process only 5 files')
    
    args = parser.parse_args()
    
    # Expand 'all' features
    if 'all' in args.features:
        args.features = ['mel', 'mfcc', 'raw']
    
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
    print("Feature Caching")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Files: {len(rows)}")
    print(f"Features: {', '.join(args.features)}")
    print(f"Parallel: {args.parallel}")
    print(f"Resume: {args.resume}")
    print("=" * 60)
    
    # Feature parameters
    params = {
        'sample_rate': args.sample_rate,
        'n_mels': args.n_mels,
        'n_mfcc': args.n_mfcc,
        'n_fft': DEFAULTS['n_fft'],
        'hop_length': DEFAULTS['hop_length'],
        'win_length': DEFAULTS['win_length'],
    }
    
    output_dir = Path(args.output_dir)
    
    # Process each feature type
    all_success = 0
    all_failed = 0
    
    for feature_type in args.features:
        print(f"\n{'=' * 60}")
        print(f"Caching {feature_type.upper()}")
        print('=' * 60)
        
        feature_dir = output_dir / feature_type
        
        try:
            success, failed = process_manifest_parallel(
                manifest_rows=rows,
                output_dir=feature_dir,
                feature_type=feature_type,
                parallel=args.parallel,
                resume=args.resume,
                verify=args.verify,
                params=params,
            )
            
            all_success += success
            all_failed += failed
            
            print(f"\n{feature_type.upper()} Summary:")
            print(f"  Success: {success}")
            print(f"  Failed:  {failed}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved.")
            sys.exit(130)
        except Exception as e:
            print(f"\nError during {feature_type} caching: {e}")
            traceback.print_exc()
            all_failed += len(rows)
    
    # Save metadata
    save_metadata(output_dir, args.features, params, len(rows))
    
    # Final summary
    print("\n" + "=" * 60)
    print("Caching Complete")
    print("=" * 60)
    print(f"Total success: {all_success}")
    print(f"Total failed:  {all_failed}")
    print("=" * 60)
    
    if all_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
