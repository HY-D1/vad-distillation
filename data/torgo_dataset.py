#!/usr/bin/env python3
"""
TORGO dataset class for VAD knowledge distillation.

This module provides a PyTorch Dataset for loading TORGO dysarthric speech data
with mel spectrograms, hard labels (from transcripts), and teacher probabilities
(from a pre-trained teacher model).

Enhanced with comprehensive caching support.
"""

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import librosa
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    warnings.warn("torchaudio/librosa not available. Mel spectrogram computation will fail.")


# Setup logging
logger = logging.getLogger(__name__)


def load_mel_spectrogram(
    audio_path: Union[str, Path],
    n_mels: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    sr: int = 16000,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute mel spectrogram from audio file.
    
    Args:
        audio_path: Path to audio file
        n_mels: Number of mel frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT (default 160 = 10ms at 16kHz)
        sr: Target sample rate
        normalize: Whether to normalize the mel spectrogram
    
    Returns:
        mel_spec: Mel spectrogram array (time, n_mels)
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio loading fails
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Load audio
        y, orig_sr = librosa.load(str(audio_path), sr=sr, mono=True)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=sr // 2
        )
        
        # Convert to log scale (dB)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time, n_mels)
        mel_spec = mel_spec.T
        
        # Normalize to roughly zero mean, unit variance
        if normalize:
            mel_mean = mel_spec.mean()
            mel_std = mel_spec.std() + 1e-8
            mel_spec = (mel_spec - mel_mean) / mel_std
        
        return mel_spec.astype(np.float32)
    
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {audio_path}: {e}")


def create_hard_labels_from_transcript(
    text: Union[str, float, None],
    num_frames: int,
    silence_tokens: Optional[List[str]] = None
) -> np.ndarray:
    """
    Create frame-level hard labels from transcript text.
    
    TODO: This is a simplified version that assumes the entire utterance is speech.
    In reality, we need proper timestamp alignment from forced alignment or similar.
    Future improvements:
        - Use Montreal Forced Aligner (MFA) or similar for word-level timestamps
        - Parse phoneme-level alignments if available
        - Use silence detection heuristics
    
    Args:
        text: Transcript text (can be NaN/None)
        num_frames: Number of frames in the utterance
        silence_tokens: List of tokens indicating silence (e.g., [SIL], <s>, etc.)
    
    Returns:
        labels: Binary array (num_frames,) where 1=speech, 0=silence
    """
    if silence_tokens is None:
        # Common silence markers in transcripts
        silence_tokens = ['[silence]', '[sil]', '<s>', '</s>', '[noise]', '[pause]']
    
    # Handle NaN/None values
    if text is None or (isinstance(text, float) and pd.isna(text)):
        text = ""
    
    text = str(text)
    text_lower = text.lower().strip()
    
    # Check if the entire utterance is marked as silence
    is_silence = any(token in text_lower for token in silence_tokens)
    
    if is_silence or len(text.strip()) == 0:
        # All silence
        labels = np.zeros(num_frames, dtype=np.float32)
    else:
        # TODO: Currently assumes all speech. In the future, use proper alignment
        # to mark actual speech vs silence regions within the utterance.
        labels = np.ones(num_frames, dtype=np.float32)
    
    return labels


def pad_sequence(sequences: List[torch.Tensor], padding_value: float = 0.0) -> torch.Tensor:
    """
    Pad a list of variable-length tensors to the same length.
    
    Args:
        sequences: List of tensors, each of shape (time, ...)
        padding_value: Value to use for padding
    
    Returns:
        padded: Tensor of shape (batch, max_time, ...)
    """
    # Find max length
    max_len = max(s.shape[0] for s in sequences)
    
    # Pad each sequence
    padded = []
    for seq in sequences:
        if seq.shape[0] < max_len:
            # Create padding
            pad_len = max_len - seq.shape[0]
            pad_shape = (pad_len,) + seq.shape[1:]
            padding = torch.full(pad_shape, padding_value, dtype=seq.dtype, device=seq.device)
            seq = torch.cat([seq, padding], dim=0)
        padded.append(seq)
    
    return torch.stack(padded, dim=0)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching variable-length sequences.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Dict with padded tensors:
            - mels: (batch, max_time, n_mels)
            - hard_labels: (batch, max_time)
            - teacher_probs: (batch, max_time)
            - utt_ids: List of utterance IDs
            - speaker_ids: List of speaker IDs
            - lengths: (batch,) - actual lengths before padding
    """
    # Extract fields
    mels = [b['mels'] for b in batch]
    hard_labels = [b['hard_labels'] for b in batch]
    teacher_probs = [b['teacher_probs'] for b in batch]
    utt_ids = [b['utt_id'] for b in batch]
    speaker_ids = [b['speaker_id'] for b in batch]
    
    # Record actual lengths
    lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    
    # Pad sequences
    mels_padded = pad_sequence(mels, padding_value=0.0)
    hard_labels_padded = pad_sequence(hard_labels, padding_value=0.0)
    teacher_probs_padded = pad_sequence(teacher_probs, padding_value=0.0)
    
    return {
        'mels': mels_padded,
        'hard_labels': hard_labels_padded,
        'teacher_probs': teacher_probs_padded,
        'utt_ids': utt_ids,
        'speaker_ids': speaker_ids,
        'lengths': lengths,
    }


class TORGODataset(Dataset):
    """
    TORGO dataset for VAD knowledge distillation.
    
    Loads mel spectrograms, hard labels (from transcripts), and teacher probabilities
    for training a student VAD model.
    
    Enhanced caching features:
    - Check for cached features first (new structure with session)
    - Fall back to on-the-fly computation
    - Optionally auto-cache computed features
    - Support for multiple cache types (mel, mfcc, raw)
    
    Example usage:
        >>> dataset = TORGODataset(
        ...     manifest_path='manifests/torgo_pilot.csv',
        ...     teacher_probs_dir='teacher_probs',
        ...     fold_config='splits/fold_F01.json',
        ...     mode='train',
        ...     cache_dir='cached_features',  # Enable caching
        ...     auto_cache=True,  # Auto-save computed features
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        teacher_probs_dir: Union[str, Path],
        fold_config: Optional[Union[str, Path, Dict]] = None,
        mode: str = 'train',
        feature_type: str = 'mel',
        n_mels: int = 40,
        max_seq_len: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        auto_cache: bool = False,
        sr: int = 16000,
        hop_length: int = 160,
    ):
        """
        Initialize TORGO dataset.
        
        Args:
            manifest_path: Path to CSV manifest with columns:
                - speaker_id: Speaker identifier (e.g., 'F01')
                - session: Session identifier (e.g., 'Session1')
                - utt_id: Utterance identifier (e.g., '0001')
                - path: Path to audio file
                - duration: Duration in seconds (optional)
                - text: Transcript text
            teacher_probs_dir: Directory containing teacher probability .npy files
            fold_config: Optional fold configuration for LOSO splits. Can be:
                - Path to JSON file (e.g., 'splits/fold_F01.json')
                - Dict with 'train_utterances', 'val_utterances', 'test_utterances'
            mode: 'train', 'val', or 'test'
            feature_type: Type of features ('mel' or 'teacher_probs')
            n_mels: Number of mel frequency bins
            max_seq_len: Maximum sequence length (truncate/pad to this)
            cache_dir: Optional directory to cache features (e.g., 'cached_features')
            auto_cache: Whether to automatically cache computed features
            sr: Sample rate for audio loading
            hop_length: Hop length for STFT (affects time resolution)
        
        Raises:
            ValueError: If mode is invalid or manifest is malformed
            FileNotFoundError: If required files don't exist
        """
        super().__init__()
        
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"mode must be 'train', 'val', or 'test', got {mode}")
        
        if feature_type not in ['mel', 'teacher_probs']:
            raise ValueError(f"feature_type must be 'mel' or 'teacher_probs', got {feature_type}")
        
        self.manifest_path = Path(manifest_path)
        self.teacher_probs_dir = Path(teacher_probs_dir)
        self.mode = mode
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.max_seq_len = max_seq_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.auto_cache = auto_cache
        self.sr = sr
        self.hop_length = hop_length
        
        # Create cache directories if specified
        if self.cache_dir:
            (self.cache_dir / 'mel').mkdir(parents=True, exist_ok=True)
            (self.cache_dir / 'mfcc').mkdir(parents=True, exist_ok=True)
            (self.cache_dir / 'raw').mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        self.manifest = pd.read_csv(self.manifest_path)
        
        # Validate manifest columns
        required_cols = ['speaker_id', 'utt_id', 'path', 'text']
        missing_cols = [c for c in required_cols if c not in self.manifest.columns]
        if missing_cols:
            raise ValueError(f"Manifest missing required columns: {missing_cols}")
        
        # Apply fold split if provided
        if fold_config is not None:
            self.manifest = self._apply_fold_split(self.manifest, fold_config, mode)
        
        # Create unique utterance ID only if not already present
        if 'unique_utt_id' not in self.manifest.columns:
            self.manifest['unique_utt_id'] = self.manifest.apply(
                lambda row: f"{row['speaker_id']}_{row.get('session', 'unknown')}_{int(row['utt_id']):04d}",
                axis=1
            )
        
        # Statistics
        logger.info(f"TORGO {mode} dataset: {len(self.manifest)} utterances")
        if len(self.manifest) > 0:
            speaker_counts = self.manifest['speaker_id'].value_counts()
            logger.info(f"  Speakers: {len(speaker_counts)} ({list(speaker_counts.index)})")
    
    def _apply_fold_split(
        self,
        manifest: pd.DataFrame,
        fold_config: Union[str, Path, Dict],
        mode: str
    ) -> pd.DataFrame:
        """
        Apply fold-based train/val/test split to manifest.
        
        Args:
            manifest: Full manifest DataFrame
            fold_config: Fold configuration
            mode: Which split to use
        
        Returns:
            Filtered manifest for the specified mode
        """
        # Load fold config
        if isinstance(fold_config, (str, Path)):
            import json
            with open(fold_config, 'r') as f:
                fold_config = json.load(f)
        
        # Get utterance list for this mode
        key_map = {'train': 'train_utterances', 'val': 'val_utterances', 'test': 'test_utterances'}
        utterance_key = key_map[mode]
        
        if utterance_key not in fold_config:
            raise ValueError(f"Fold config missing '{utterance_key}'")
        
        valid_utterances = set(fold_config[utterance_key])
        
        # Create unique_utt_id in the same format as fold config (only if not present)
        manifest = manifest.copy()
        if 'unique_utt_id' not in manifest.columns:
            manifest['unique_utt_id'] = manifest.apply(
                lambda row: f"{row['speaker_id']}_{row.get('session', 'unknown')}_{int(row['utt_id']):04d}",
                axis=1
            )
        
        # Filter
        filtered = manifest[manifest['unique_utt_id'].isin(valid_utterances)].copy()
        
        logger.info(f"Applied {mode} split: {len(filtered)}/{len(manifest)} utterances")
        
        return filtered
    
    def _get_cache_filename(self, row: pd.Series) -> str:
        """Generate cache filename from manifest row."""
        speaker_id = row['speaker_id']
        utt_id = row['utt_id']
        session = row.get('session', 'unknown')
        return f"{speaker_id}_{session}_{int(utt_id):04d}.npy"
    
    def _get_teacher_prob_path(self, speaker_id: str, utt_id: str, session: str = None) -> Path:
        """
        Get path to teacher probability file.
        
        Tries multiple naming conventions:
        1. {speaker_id}_{session}_{utt_id}.npy (new format)
        2. {speaker_id}_{utt_id}.npy (legacy format)
        """
        # Format utterance ID with leading zeros
        utt_id_str = str(int(utt_id)).zfill(4)
        
        # Try different naming patterns
        if session:
            patterns = [
                f"{speaker_id}_{session}_{utt_id_str}.npy",
                f"{speaker_id}_{session}_{utt_id}.npy",
            ]
        else:
            patterns = [f"{speaker_id}_{utt_id_str}.npy", f"{speaker_id}_{utt_id}.npy"]
        
        for pattern in patterns:
            path = self.teacher_probs_dir / pattern
            if path.exists():
                return path
        
        # Return first pattern (will fail later with warning)
        return self.teacher_probs_dir / patterns[0]
    
    def _load_teacher_probs(self, prob_path: Path, expected_frames: int) -> np.ndarray:
        """
        Load teacher probabilities from .npy file.
        
        Args:
            prob_path: Path to .npy file
            expected_frames: Expected number of frames (from mel spectrogram)
        
        Returns:
            teacher_probs: Array of shape (expected_frames,)
        
        Handles:
            - Missing files (returns zeros with warning)
            - Mismatched lengths (interpolates or truncates)
        """
        if not prob_path.exists():
            logger.warning(f"Teacher probs not found: {prob_path}. Using zeros.")
            return np.zeros(expected_frames, dtype=np.float32)
        
        try:
            teacher_probs = np.load(prob_path).astype(np.float32)
            
            # Handle mismatched lengths
            if len(teacher_probs) != expected_frames:
                logger.debug(
                    f"Length mismatch: teacher={len(teacher_probs)}, mels={expected_frames}"
                )
                
                if len(teacher_probs) == 0:
                    return np.zeros(expected_frames, dtype=np.float32)
                
                # Interpolate or truncate
                if len(teacher_probs) < expected_frames:
                    # Interpolate up
                    teacher_probs = np.interp(
                        np.linspace(0, len(teacher_probs) - 1, expected_frames),
                        np.arange(len(teacher_probs)),
                        teacher_probs
                    )
                else:
                    # Truncate
                    teacher_probs = teacher_probs[:expected_frames]
            
            return teacher_probs
        
        except Exception as e:
            logger.warning(f"Failed to load teacher probs {prob_path}: {e}. Using zeros.")
            return np.zeros(expected_frames, dtype=np.float32)
    
    def _get_cached_feature_path(self, row: pd.Series, feature_type: str = 'mel') -> Optional[Path]:
        """
        Get path to cached features.
        
        Args:
            row: Manifest row
            feature_type: Type of features ('mel', 'mfcc', 'raw')
        
        Returns:
            Path to cached file or None if not cached
        """
        if self.cache_dir is None:
            return None
        
        cache_file = self._get_cache_filename(row)
        cache_path = self.cache_dir / feature_type / cache_file
        
        if cache_path.exists():
            return cache_path
        
        # Try legacy naming (without session)
        speaker_id = row['speaker_id']
        utt_id = row['utt_id']
        legacy_file = f"{speaker_id}_{int(utt_id):04d}.npy"
        legacy_path = self.cache_dir / feature_type / legacy_file
        
        if legacy_path.exists():
            return legacy_path
        
        return None
    
    def _load_cached_features(self, row: pd.Series, feature_type: str = 'mel') -> Optional[np.ndarray]:
        """
        Load features from cache if available.
        
        Args:
            row: Manifest row
            feature_type: Type of features
        
        Returns:
            Features array or None if not cached
        """
        cache_path = self._get_cached_feature_path(row, feature_type)
        
        if cache_path is None:
            return None
        
        try:
            features = np.load(cache_path).astype(np.float32)
            logger.debug(f"Loaded cached {feature_type} from {cache_path}")
            return features
        except Exception as e:
            logger.warning(f"Failed to load cached features from {cache_path}: {e}")
            return None
    
    def _save_features_to_cache(self, features: np.ndarray, row: pd.Series, feature_type: str = 'mel'):
        """
        Save computed features to cache.
        
        Args:
            features: Feature array
            row: Manifest row
            feature_type: Type of features
        """
        if self.cache_dir is None:
            return
        
        try:
            cache_file = self._get_cache_filename(row)
            cache_path = self.cache_dir / feature_type / cache_file
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(cache_path, features)
            logger.debug(f"Saved {feature_type} features to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save features to cache: {e}")
    
    def _load_mel(self, audio_path: Path, row: pd.Series) -> np.ndarray:
        """
        Load mel spectrogram, using cache if available.
        
        Args:
            audio_path: Path to audio file
            row: Manifest row
        
        Returns:
            mel_spec: Mel spectrogram (time, n_mels)
        """
        # Try to load from cache first
        mel_spec = self._load_cached_features(row, 'mel')
        
        if mel_spec is not None:
            return mel_spec
        
        # Compute mel spectrogram
        mel_spec = load_mel_spectrogram(
            audio_path,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            sr=self.sr
        )
        
        # Save to cache if auto_cache is enabled
        if self.auto_cache:
            self._save_features_to_cache(mel_spec, row, 'mel')
        
        return mel_spec
    
    def __len__(self) -> int:
        """Return number of utterances in dataset."""
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
        
        Returns:
            Dict with:
                - mels: Tensor (time, n_mels)
                - hard_labels: Tensor (time,) - binary speech/silence labels
                - teacher_probs: Tensor (time,) - teacher model probabilities
                - utt_id: Utterance ID
                - speaker_id: Speaker ID
        """
        row = self.manifest.iloc[idx]
        
        speaker_id = row['speaker_id']
        utt_id = str(int(row['utt_id']))
        session = row.get('session', None)
        audio_path = Path(row['path'])
        text = row.get('text', '')
        
        # Ensure audio path is correct
        # If path is relative and doesn't start with 'data/', prepend it
        if not audio_path.is_absolute():
            path_str = str(audio_path)
            if not path_str.startswith(f'data{os.sep}'):
                # Try relative to manifest directory
                audio_path = self.manifest_path.parent / audio_path
        
        # Load mel spectrogram (with caching)
        try:
            mel_spec = self._load_mel(audio_path, row)
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            # Return dummy data on failure
            mel_spec = np.zeros((100, self.n_mels), dtype=np.float32)
        
        num_frames = mel_spec.shape[0]
        
        # Create hard labels from transcript (handle NaN text values)
        text_value = row.get('text', '')
        hard_labels = create_hard_labels_from_transcript(text_value, num_frames)
        
        # Load teacher probabilities
        prob_path = self._get_teacher_prob_path(speaker_id, utt_id, session)
        teacher_probs = self._load_teacher_probs(prob_path, num_frames)
        
        # Apply max sequence length limit
        if self.max_seq_len is not None and num_frames > self.max_seq_len:
            mel_spec = mel_spec[:self.max_seq_len]
            hard_labels = hard_labels[:self.max_seq_len]
            teacher_probs = teacher_probs[:self.max_seq_len]
        
        # Convert to tensors
        return {
            'mels': torch.from_numpy(mel_spec),
            'hard_labels': torch.from_numpy(hard_labels),
            'teacher_probs': torch.from_numpy(teacher_probs),
            'utt_id': row.get('unique_utt_id', f"{speaker_id}_{utt_id}"),
            'speaker_id': speaker_id,
        }
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about cache usage.
        
        Returns:
            Dict with cache statistics
        """
        if self.cache_dir is None:
            return {'enabled': False}
        
        stats = {'enabled': True, 'auto_cache': self.auto_cache, 'types': {}}
        
        for feature_type in ['mel', 'mfcc', 'raw']:
            cache_subdir = self.cache_dir / feature_type
            if cache_subdir.exists():
                num_files = len(list(cache_subdir.glob('*.npy')))
                stats['types'][feature_type] = num_files
            else:
                stats['types'][feature_type] = 0
        
        return stats
    
    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics.
        
        Returns:
            Dict with statistics:
                - num_utterances: Total number of utterances
                - num_speakers: Number of unique speakers
                - duration_stats: Min/max/mean duration
                - label_distribution: Speech vs silence ratio
        """
        stats = {
            'num_utterances': len(self.manifest),
            'num_speakers': self.manifest['speaker_id'].nunique(),
            'speakers': sorted(self.manifest['speaker_id'].unique().tolist()),
        }
        
        # Add cache stats
        stats['cache'] = self.get_cache_stats()
        
        # Duration statistics
        if 'duration' in self.manifest.columns:
            durations = self.manifest['duration'].dropna()
            if len(durations) > 0:
                stats['duration_stats'] = {
                    'min': float(durations.min()),
                    'max': float(durations.max()),
                    'mean': float(durations.mean()),
                    'total': float(durations.sum()),
                }
        
        # Sample a few utterances to estimate label distribution
        # (doing all would be too slow)
        sample_size = min(100, len(self.manifest))
        if sample_size > 0:
            sample_indices = np.random.choice(len(self.manifest), sample_size, replace=False)
            total_speech = 0
            total_frames = 0
            
            for idx in sample_indices:
                row = self.manifest.iloc[idx]
                text = row.get('text', '')
                
                # Estimate frames from duration or use default
                if 'duration' in row and pd.notna(row['duration']):
                    est_frames = int(row['duration'] * self.sr / self.hop_length)
                else:
                    est_frames = 100  # Default estimate
                
                labels = create_hard_labels_from_transcript(text, est_frames)
                total_speech += labels.sum()
                total_frames += len(labels)
            
            if total_frames > 0:
                stats['label_distribution'] = {
                    'speech_ratio': float(total_speech / total_frames),
                    'silence_ratio': float(1 - total_speech / total_frames),
                }
        
        return stats


def create_dataloader(
    dataset: TORGODataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader with proper collate function.
    
    Args:
        dataset: TORGODataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (for GPU training)
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == '__main__':
    """
    Test script for TORGODataset.
    
    Usage:
        python data/torgo_dataset.py
    """
    import json
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Paths
    manifest_path = Path("../manifests/torgo_pilot.csv")
    teacher_probs_dir = Path("../teacher_probs")
    fold_config_path = Path("../splits/fold_F01.json")
    cache_dir = Path("../cached_features")
    
    # Check if files exist (handle both absolute and relative paths)
    if not manifest_path.exists():
        manifest_path = Path("manifests/torgo_pilot.csv")
    if not fold_config_path.exists():
        fold_config_path = Path("splits/fold_F01.json")
    
    print("=" * 60)
    print("TORGO Dataset Testing")
    print("=" * 60)
    
    # Test 1: Basic dataset creation with caching
    print("\n1. Testing basic dataset creation with caching...")
    try:
        dataset = TORGODataset(
            manifest_path=manifest_path,
            teacher_probs_dir=teacher_probs_dir,
            n_mels=40,
            max_seq_len=500,  # ~5 seconds at 10ms hop
            cache_dir=cache_dir if cache_dir.exists() else None,
            auto_cache=False,  # Don't auto-cache in test
        )
        print(f"   ✓ Dataset created with {len(dataset)} utterances")
        
        # Get statistics
        stats = dataset.get_statistics()
        print(f"   ✓ Speakers: {stats['num_speakers']} ({stats['speakers']})")
        if 'duration_stats' in stats:
            d = stats['duration_stats']
            print(f"   ✓ Duration: min={d['min']:.2f}s, max={d['max']:.2f}s, mean={d['mean']:.2f}s")
        if 'label_distribution' in stats:
            ld = stats['label_distribution']
            print(f"   ✓ Labels: speech={ld['speech_ratio']:.2%}, silence={ld['silence_ratio']:.2%}")
        if stats['cache']['enabled']:
            print(f"   ✓ Cache: {stats['cache']['types']}")
        else:
            print(f"   ⚠ Cache: Not enabled (directory not found)")
    
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        dataset = None
    
    # Test 2: Load individual samples
    if dataset and len(dataset) > 0:
        print("\n2. Testing __getitem__...")
        try:
            sample = dataset[0]
            print(f"   ✓ Sample loaded:")
            print(f"     - mels shape: {sample['mels'].shape}")
            print(f"     - hard_labels shape: {sample['hard_labels'].shape}")
            print(f"     - teacher_probs shape: {sample['teacher_probs'].shape}")
            print(f"     - utt_id: {sample['utt_id']}")
            print(f"     - speaker_id: {sample['speaker_id']}")
            print(f"     - hard_labels unique: {torch.unique(sample['hard_labels']).tolist()}")
            print(f"     - teacher_probs range: [{sample['teacher_probs'].min():.3f}, {sample['teacher_probs'].max():.3f}]")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 3: Fold-based split
    print("\n3. Testing fold-based split...")
    try:
        if fold_config_path.exists():
            dataset_train = TORGODataset(
                manifest_path=manifest_path,
                teacher_probs_dir=teacher_probs_dir,
                fold_config=fold_config_path,
                mode='train',
                n_mels=40,
                cache_dir=cache_dir if cache_dir.exists() else None,
            )
            dataset_val = TORGODataset(
                manifest_path=manifest_path,
                teacher_probs_dir=teacher_probs_dir,
                fold_config=fold_config_path,
                mode='val',
                n_mels=40,
                cache_dir=cache_dir if cache_dir.exists() else None,
            )
            dataset_test = TORGODataset(
                manifest_path=manifest_path,
                teacher_probs_dir=teacher_probs_dir,
                fold_config=fold_config_path,
                mode='test',
                n_mels=40,
                cache_dir=cache_dir if cache_dir.exists() else None,
            )
            print(f"   ✓ Train: {len(dataset_train)} utterances")
            print(f"   ✓ Val: {len(dataset_val)} utterances")
            print(f"   ✓ Test: {len(dataset_test)} utterances")
            
            # Verify no overlap
            train_utts = set(dataset_train.manifest['unique_utt_id'])
            val_utts = set(dataset_val.manifest['unique_utt_id'])
            test_utts = set(dataset_test.manifest['unique_utt_id'])
            
            overlap_train_val = train_utts & val_utts
            overlap_train_test = train_utts & test_utts
            overlap_val_test = val_utts & test_utts
            
            if not overlap_train_val and not overlap_train_test and not overlap_val_test:
                print("   ✓ No overlap between splits")
            else:
                print(f"   ⚠ Overlap detected: train/val={len(overlap_train_val)}, train/test={len(overlap_train_test)}, val/test={len(overlap_val_test)}")
        else:
            print(f"   ⚠ Fold config not found: {fold_config_path}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: DataLoader with collate function
    if dataset and len(dataset) > 0:
        print("\n4. Testing DataLoader with custom collate_fn...")
        try:
            # Use smaller dataset for testing
            test_dataset = TORGODataset(
                manifest_path=manifest_path,
                teacher_probs_dir=teacher_probs_dir,
                fold_config=fold_config_path if fold_config_path.exists() else None,
                mode='train',
                n_mels=40,
                max_seq_len=200,  # Short sequences for testing
                cache_dir=cache_dir if cache_dir.exists() else None,
            )
            
            if len(test_dataset) > 0:
                dataloader = create_dataloader(
                    test_dataset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=0,
                )
                
                batch = next(iter(dataloader))
                print(f"   ✓ Batch loaded:")
                print(f"     - mels shape: {batch['mels'].shape}")
                print(f"     - hard_labels shape: {batch['hard_labels'].shape}")
                print(f"     - teacher_probs shape: {batch['teacher_probs'].shape}")
                print(f"     - lengths: {batch['lengths'].tolist()}")
                print(f"     - utt_ids: {batch['utt_ids']}")
                
                # Verify padding worked correctly
                max_len = batch['lengths'].max().item()
                assert batch['mels'].shape[1] == max_len, "Padding mismatch"
                print("   ✓ Padding verified")
            else:
                print("   ⚠ No samples in test dataset")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 5: Error handling
    print("\n5. Testing error handling...")
    try:
        # Test missing teacher probs
        temp_dir = Path(tempfile.mkdtemp(prefix="empty_teacher_probs_"))
        
        test_dataset = TORGODataset(
            manifest_path=manifest_path,
            teacher_probs_dir=temp_dir,
            n_mels=40,
            cache_dir=cache_dir if cache_dir.exists() else None,
        )
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            # Should have zeros for teacher_probs
            assert torch.allclose(sample['teacher_probs'], torch.zeros_like(sample['teacher_probs'])), \
                "Should return zeros for missing teacher probs"
            print("   ✓ Missing teacher probs handled (returns zeros)")
        
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
