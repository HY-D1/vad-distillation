#!/usr/bin/env python3
"""
Unified baseline execution script for VAD methods.

Supports Energy-based VAD, SpeechBrain VAD, and Silero VAD.

Usage:
    python scripts/run_baseline.py \
        --method energy \
        --manifest manifests/torgo_sentences.csv \
        --output-dir outputs/baseline_energy/
    
    python scripts/run_baseline.py \
        --method silero \
        --manifest manifests/torgo_pilot.csv \
        --output-dir outputs/baseline_silero/ \
        --device cuda
    
    python scripts/run_baseline.py \
        --method speechbrain \
        --manifest manifests/torgo_sentences.csv \
        --output-dir outputs/baseline_speechbrain/ \
        --test  # Process only 5 files for testing
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for baselines import
sys.path.insert(0, str(Path(__file__).parent.parent))
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device, auto-detecting if not specified.
    
    Args:
        device: Device string ('cpu', 'cuda', or None for auto)
        
    Returns:
        torch.device object
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    return torch.device(device)


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Load manifest CSV file.
    
    Args:
        manifest_path: Path to CSV manifest
        
    Returns:
        List of dictionaries containing utterance metadata
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    logger.info(f"Loaded {len(rows)} utterances from {manifest_path}")
    return rows


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if necessary.
    
    NOTE: This is a standalone copy in run_baseline.py. A similar but
    incompatible function exists in cache_teacher.py that returns only the
    waveform tensor (not a tuple). Kept separate due to different return types.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (waveform tensor, sample rate)
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    
    return waveform.squeeze(0), sr


def probs_to_segments(
    frame_probs: np.ndarray,
    frame_hop_ms: float,
    threshold: float = 0.5,
    min_duration_ms: float = 0.0
) -> List[Tuple[float, float]]:
    """
    Convert frame probabilities to speech segments.
    
    Args:
        frame_probs: Array of frame probabilities
        frame_hop_ms: Frame hop size in milliseconds
        threshold: Probability threshold for speech detection
        min_duration_ms: Minimum segment duration in milliseconds
        
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    speech_frames = frame_probs >= threshold
    segments = []
    
    start_frame = None
    for i, is_speech in enumerate(speech_frames):
        if is_speech and start_frame is None:
            start_frame = i
        elif not is_speech and start_frame is not None:
            end_frame = i
            start_time = start_frame * frame_hop_ms / 1000.0
            end_time = end_frame * frame_hop_ms / 1000.0
            duration_ms = (end_frame - start_frame) * frame_hop_ms
            
            if duration_ms >= min_duration_ms:
                segments.append((start_time, end_time))
            start_frame = None
    
    # Handle case where speech continues to the end
    if start_frame is not None:
        end_frame = len(speech_frames)
        start_time = start_frame * frame_hop_ms / 1000.0
        end_time = end_frame * frame_hop_ms / 1000.0
        duration_ms = (end_frame - start_frame) * frame_hop_ms
        
        if duration_ms >= min_duration_ms:
            segments.append((start_time, end_time))
    
    return segments


def save_segments(segments: List[Tuple[float, float]], output_path: Path) -> None:
    """
    Save speech segments to text file.
    
    Args:
        segments: List of (start_time, end_time) tuples
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for start, end in segments:
            f.write(f"{start:.3f}\t{end:.3f}\n")


class SileroVADRunner:
    """Runner for Silero VAD."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.utils = None
        self.frame_hop_ms = 32  # Silero uses 512 samples @ 16kHz = 32ms
        
    def load(self):
        """Load Silero VAD model."""
        logger.info("Loading Silero VAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Silero VAD model loaded successfully")
        
    def process(self, audio_path: str) -> np.ndarray:
        """
        Process audio file and return frame probabilities.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Array of frame probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load audio
        waveform, sr = load_audio(audio_path, target_sr=16000)
        
        # Silero processes audio in 512-sample chunks for 16kHz (32ms frames)
        chunk_size = 512
        num_chunks = (len(waveform) + chunk_size - 1) // chunk_size
        
        probs = []
        self.model.reset_states()
        
        with torch.no_grad():
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(waveform))
                chunk = waveform[start:end]
                
                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Get probability for this chunk
                prob = self.model(chunk.unsqueeze(0).to(self.device), sr)
                probs.append(prob.item())
        
        return np.array(probs)


class EnergyVADRunner:
    """Runner for Energy-based VAD."""
    
    def __init__(self, frame_hop_ms: float = 10, device: torch.device = None):
        self.frame_hop_ms = frame_hop_ms
        self.device = device
        self.sample_rate = 16000
        self.frame_length = int(self.sample_rate * frame_hop_ms / 1000)
        
    def load(self):
        """Energy VAD doesn't require model loading."""
        try:
            from baselines.energy_vad import EnergyVAD
            self.EnergyVAD = EnergyVAD
            logger.info("Energy VAD module loaded successfully")
        except ImportError as e:
            logger.warning(f"Could not import EnergyVAD from baselines.energy_vad: {e}")
            logger.info("Using built-in energy VAD implementation")
            self.EnergyVAD = None
        
    def process(self, audio_path: str) -> np.ndarray:
        """
        Process audio file and return frame probabilities.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Array of frame probabilities (normalized energy)
        """
        waveform, sr = load_audio(audio_path, target_sr=self.sample_rate)
        
        if self.EnergyVAD is not None:
            # Use imported EnergyVAD if available
            vad = self.EnergyVAD(frame_hop_ms=int(self.frame_hop_ms))
            audio = waveform.numpy().squeeze()
            probs, _ = vad.get_frame_probs(audio, sr=self.sample_rate)
            return probs
        else:
            # Built-in simple energy-based VAD
            return self._compute_energy_probs(waveform.numpy())
    
    def _compute_energy_probs(self, waveform: np.ndarray) -> np.ndarray:
        """Compute normalized energy-based probabilities."""
        frame_shift = self.frame_length // 2
        num_frames = (len(waveform) - self.frame_length) // frame_shift + 1
        
        energies = []
        for i in range(num_frames):
            start = i * frame_shift
            end = start + self.frame_length
            frame = waveform[start:end]
            energy = np.sum(frame ** 2)
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Normalize to probabilities using adaptive threshold
        mean_energy = np.mean(energies)
        std_energy = np.std(energies) + 1e-10
        probs = 1 / (1 + np.exp(-(energies - mean_energy) / std_energy))
        
        return probs


class SpeechBrainVADRunner:
    """Runner for SpeechBrain VAD."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.frame_hop_ms = 10  # SpeechBrain typically uses 10ms
        
    def load(self):
        """Load SpeechBrain VAD model."""
        logger.info("Loading SpeechBrain VAD model...")
        try:
            from baselines.speechbrain_vad import SpeechBrainVAD
            self.SpeechBrainVAD = SpeechBrainVAD
            # Initialize the VAD
            self.model = SpeechBrainVAD(device=str(self.device))
            logger.info("SpeechBrain VAD model loaded successfully")
        except ImportError as e:
            logger.error(f"Could not import SpeechBrainVAD: {e}")
            raise RuntimeError(
                "SpeechBrain VAD not available. "
                "Please ensure baselines.speechbrain_vad is implemented."
            )
        
    def process(self, audio_path: str) -> np.ndarray:
        """
        Process audio file and return frame probabilities.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Array of frame probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Use the SpeechBrain VAD to get probabilities
        return self.model.process_file(audio_path)


def create_runner(method: str, frame_hop_ms: float, device: torch.device):
    """
    Create appropriate VAD runner based on method.
    
    Args:
        method: VAD method name ('energy', 'silero', 'speechbrain')
        frame_hop_ms: Frame hop size in milliseconds
        device: Torch device
        
    Returns:
        VAD runner instance
    """
    method = method.lower()
    
    if method == 'silero':
        runner = SileroVADRunner(device)
    elif method == 'energy':
        runner = EnergyVADRunner(frame_hop_ms=frame_hop_ms, device=device)
    elif method == 'speechbrain':
        runner = SpeechBrainVADRunner(device)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: energy, silero, speechbrain")
    
    runner.load()
    return runner


def get_utt_id(row: Dict[str, str]) -> str:
    """
    Generate unique utterance ID from manifest row.
    
    Args:
        row: Manifest row dictionary
        
    Returns:
        Unique utterance ID string
    """
    speaker_id = row.get('speaker_id', '')
    session = row.get('session', '')
    utt_id = row.get('utt_id', '')
    
    # Format: F01_Session1_0001
    if speaker_id and session and utt_id:
        try:
            utt_num = int(utt_id)
            return f"{speaker_id}_{session}_{utt_num:04d}"
        except ValueError:
            return f"{speaker_id}_{session}_{utt_id}"
    
    # Fallback: use path
    return Path(row.get('path', '')).stem


def save_metadata(
    output_dir: Path,
    method: str,
    config: Dict[str, Any],
    failed_files: List[str]
) -> None:
    """
    Save metadata and configuration files.
    
    Args:
        output_dir: Output directory
        method: VAD method used
        config: Configuration dictionary
        failed_files: List of failed file paths
    """
    timestamp = datetime.now().isoformat()
    
    # Save meta.json
    meta = {
        'method': method,
        'timestamp': timestamp,
        'config': config,
        'failed_files': failed_files,
        'total_failed': len(failed_files),
    }
    
    meta_path = output_dir / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")
    
    # Save config.yaml
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Configuration saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline VAD methods on manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run energy-based VAD
    python scripts/run_baseline.py --method energy --manifest manifests/torgo.csv --output-dir outputs/energy/
    
    # Run Silero VAD on GPU
    python scripts/run_baseline.py --method silero --manifest manifests/torgo.csv --output-dir outputs/silero/ --device cuda
    
    # Test with only 5 files
    python scripts/run_baseline.py --method speechbrain --manifest manifests/torgo.csv --output-dir outputs/sb/ --test
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['energy', 'speechbrain', 'silero'],
        help='VAD method to use'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to CSV manifest'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--frame-hop-ms',
        type=float,
        default=10,
        help='Frame hop size in milliseconds (default: 10)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use (cpu/cuda, auto-detect if not given)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Process only 5 files for testing'
    )
    parser.add_argument(
        '--segment-threshold',
        type=float,
        default=0.5,
        help='Threshold for converting probabilities to segments (default: 0.5)'
    )
    parser.add_argument(
        '--min-segment-duration',
        type=float,
        default=0.0,
        help='Minimum segment duration in milliseconds (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    frame_probs_dir = output_dir / 'frame_probs'
    segments_dir = output_dir / 'segments'
    
    frame_probs_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Frame probabilities: {frame_probs_dir}")
    logger.info(f"Segments: {segments_dir}")
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load manifest
    try:
        manifest = load_manifest(args.manifest)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
    
    # Limit to 5 files if testing
    if args.test:
        manifest = manifest[:5]
        logger.info(f"TEST MODE: Processing only {len(manifest)} files")
    
    # Create VAD runner
    try:
        runner = create_runner(args.method, args.frame_hop_ms, device)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error(e)
        sys.exit(1)
    
    # Process utterances
    failed_files = []
    processed_count = 0
    
    pbar = tqdm(manifest, desc=f"Processing with {args.method}")
    for row in pbar:
        audio_path = row['path']
        utt_id = get_utt_id(row)
        
        # Set progress bar description
        pbar.set_postfix({'utt': utt_id})
        
        # Define output paths
        frame_prob_path = frame_probs_dir / f"{utt_id}.npy"
        segment_path = segments_dir / f"{utt_id}.txt"
        
        # Skip if already processed
        if frame_prob_path.exists() and segment_path.exists():
            logger.debug(f"Skipping {utt_id} (already processed)")
            continue
        
        try:
            # Check if audio file exists
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Get frame probabilities
            frame_probs = runner.process(audio_path)
            
            # Convert to segments
            segments = probs_to_segments(
                frame_probs,
                runner.frame_hop_ms,
                threshold=args.segment_threshold,
                min_duration_ms=args.min_segment_duration
            )
            
            # Save frame probabilities
            np.save(frame_prob_path, frame_probs)
            
            # Save segments
            save_segments(segments, segment_path)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            failed_files.append({
                'utt_id': utt_id,
                'path': audio_path,
                'error': str(e)
            })
            continue
    
    # Prepare configuration
    config = {
        'method': args.method,
        'manifest': args.manifest,
        'output_dir': args.output_dir,
        'frame_hop_ms': args.frame_hop_ms,
        'device': str(device),
        'segment_threshold': args.segment_threshold,
        'min_segment_duration_ms': args.min_segment_duration,
        'test_mode': args.test,
        'total_utterances': len(manifest),
        'processed_count': processed_count,
        'failed_count': len(failed_files),
    }
    
    # Save metadata
    save_metadata(output_dir, args.method, config, failed_files)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("BASELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Method: {args.method}")
    logger.info(f"Total utterances: {len(manifest)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed: {len(failed_files)}")
    logger.info(f"Output directory: {output_dir}")
    
    if failed_files:
        logger.warning(f"\nFailed files ({len(failed_files)}):")
        for failed in failed_files[:10]:  # Show first 10
            logger.warning(f"  - {failed['utt_id']}: {failed['error']}")
        if len(failed_files) > 10:
            logger.warning(f"  ... and {len(failed_files) - 10} more")
    
    logger.info("=" * 60)
    
    # Exit with error code if all files failed
    if len(failed_files) == len(manifest) and len(manifest) > 0:
        logger.error("All files failed to process!")
        sys.exit(1)


if __name__ == "__main__":
    main()
