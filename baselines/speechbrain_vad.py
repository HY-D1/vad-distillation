"""
SpeechBrain VAD baseline wrapper.

This module provides a wrapper around SpeechBrain's pretrained VAD model
for voice activity detection.
"""

import os
import warnings
from typing import List, Tuple, Optional
import numpy as np

# Handle import error gracefully
try:
    import torch
    from speechbrain.inference.VAD import VAD
    SPEECHBRAIN_AVAILABLE = True
except ImportError as e:
    SPEECHBRAIN_AVAILABLE = False
    VAD = None
    torch = None
    warnings.warn(
        f"SpeechBrain not available: {e}. "
        "Please install it with: pip install speechbrain"
    )


class SpeechBrainVAD:
    """
    SpeechBrain VAD baseline wrapper.
    
    Wraps the pretrained VAD model from SpeechBrain:
    "speechbrain/vad-crdnn-libriparty"
    
    Attributes:
        vad: The underlying SpeechBrain VAD model
        device: The device used for inference (cuda/cpu)
        sample_rate: Expected sample rate (16kHz for this model)
    """
    
    DEFAULT_SOURCE = "speechbrain/vad-crdnn-libriparty"
    DEFAULT_SAVEDIR = "pretrained_models/vad-crdnn-libriparty"
    EXPECTED_SAMPLE_RATE = 16000
    
    def __init__(
        self,
        source: str = DEFAULT_SOURCE,
        savedir: str = DEFAULT_SAVEDIR,
        device: Optional[str] = None
    ):
        """
        Initialize the SpeechBrain VAD wrapper.
        
        Args:
            source: HuggingFace model identifier or local path
            savedir: Directory to save/load pretrained model
            device: Device to use for inference ('cuda', 'cpu', or None for auto)
        
        Raises:
            ImportError: If SpeechBrain is not installed
            RuntimeError: If model loading fails
        """
        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError(
                "SpeechBrain is required. Install with: pip install speechbrain"
            )
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.source = source
        self.savedir = savedir
        self.sample_rate = self.EXPECTED_SAMPLE_RATE
        
        # Load the VAD model
        try:
            self.vad = VAD.from_hparams(
                source=source,
                savedir=savedir,
                run_opts={"device": device}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load VAD model: {e}")
    
    def _get_audio_info(self, audio_file: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data with sample rate.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Use torchaudio for loading (SpeechBrain's dependency)
        try:
            import torchaudio
        except ImportError:
            raise ImportError(
                "torchaudio is required for audio loading. "
                "Install with: pip install torchaudio"
            )
        
        audio, sr = torchaudio.load(audio_file)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate
            )
            audio = resampler(audio)
        
        return audio.squeeze().numpy(), self.sample_rate
    
    def get_frame_probs(self, audio_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frame-level speech probabilities for an audio file.
        
        Uses VAD.get_speech_prob_file() to get frame posteriors.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Tuple of (probs, times) as numpy arrays:
                - probs: Frame-level speech probabilities (shape: [n_frames])
                - times: Time in seconds for each frame (shape: [n_frames])
                
        Raises:
            FileNotFoundError: If audio file does not exist
            RuntimeError: If VAD inference fails
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        try:
            # Get frame posteriors using SpeechBrain's method
            # This returns a tensor of shape [n_chunks, frames_per_chunk, 1]
            prob_chunks = self.vad.get_speech_prob_file(audio_file)
            
            # Convert to numpy and flatten
            if isinstance(prob_chunks, torch.Tensor):
                prob_chunks = prob_chunks.cpu().numpy()
            
            # Flatten to get frame-level probabilities
            # Shape: [n_chunks, frames_per_chunk, 1] -> [n_frames]
            probs = prob_chunks.squeeze().flatten()
            
            # Calculate frame times
            # SpeechBrain uses 10ms frames by default (0.01s hop length)
            hop_length = 0.01  # 10ms
            times = np.arange(len(probs)) * hop_length
            
            return probs, times
            
        except Exception as e:
            raise RuntimeError(f"Failed to get frame probabilities: {e}")
    
    def process_file(self, audio_file: str) -> np.ndarray:
        """Return frame-level speech probabilities for run_baseline.py compatibility."""
        probs, _ = self.get_frame_probs(audio_file)
        return probs
    
    def get_segments(
        self,
        audio_file: str,
        apply_energy_vad: bool = False,
        merge_close: bool = True,
        remove_short: bool = True,
        threshold: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Get speech segments from an audio file.
        
        Args:
            audio_file: Path to audio file
            apply_energy_vad: Whether to apply energy-based VAD refinement
            merge_close: Whether to merge close segments
            remove_short: Whether to remove short segments
            threshold: Probability threshold for speech detection
            
        Returns:
            List of (start, end) tuples in seconds
            
        Raises:
            FileNotFoundError: If audio file does not exist
            RuntimeError: If VAD inference fails
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        try:
            # Get speech boundaries using SpeechBrain's method
            # This returns a tensor of shape [n_segments, 2] with (start, end) in seconds
            boundaries = self.vad.get_speech_segments(
                audio_file,
                apply_energy_VAD=apply_energy_vad
            )
            
            # Convert to numpy
            if isinstance(boundaries, torch.Tensor):
                boundaries = boundaries.cpu().numpy()
            
            # Convert to list of tuples
            segments = [(float(start), float(end)) for start, end in boundaries]
            
            # Apply post-processing if requested
            if merge_close:
                segments = self._merge_close_segments(segments)
            
            if remove_short:
                segments = self._remove_short_segments(segments)
            
            return segments
            
        except Exception as e:
            raise RuntimeError(f"Failed to get speech segments: {e}")
    
    def _merge_close_segments(
        self,
        segments: List[Tuple[float, float]],
        max_gap: float = 0.3
    ) -> List[Tuple[float, float]]:
        """
        Merge speech segments that are close to each other.
        
        Args:
            segments: List of (start, end) tuples
            max_gap: Maximum gap in seconds to merge segments
            
        Returns:
            List of merged segments
        """
        if len(segments) <= 1:
            return segments
        
        # Sort by start time
        segments = sorted(segments, key=lambda x: x[0])
        
        merged = [segments[0]]
        for current_start, current_end in segments[1:]:
            prev_start, prev_end = merged[-1]
            
            # Check if segments are close enough to merge
            if current_start - prev_end <= max_gap:
                # Merge segments
                merged[-1] = (prev_start, max(prev_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged
    
    def _remove_short_segments(
        self,
        segments: List[Tuple[float, float]],
        min_duration: float = 0.25
    ) -> List[Tuple[float, float]]:
        """
        Remove short speech segments.
        
        Args:
            segments: List of (start, end) tuples
            min_duration: Minimum duration in seconds to keep a segment
            
        Returns:
            List of filtered segments
        """
        return [
            (start, end) for start, end in segments
            if (end - start) >= min_duration
        ]
    
    def predict(self, audio_file: str) -> dict:
        """
        Run full VAD prediction on an audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with keys:
                - 'frame_probs': Frame-level probabilities
                - 'frame_times': Time for each frame
                - 'segments': List of speech segments
                - 'total_speech_duration': Total duration of speech
        """
        probs, times = self.get_frame_probs(audio_file)
        segments = self.get_segments(audio_file)
        
        total_speech = sum(end - start for start, end in segments)
        
        return {
            'frame_probs': probs,
            'frame_times': times,
            'segments': segments,
            'total_speech_duration': total_speech
        }


def test_speechbrain_vad():
    """
    Test the SpeechBrainVAD wrapper.
    
    Creates a synthetic audio file and runs VAD on it.
    """
    print("=" * 60)
    print("Testing SpeechBrain VAD Baseline")
    print("=" * 60)
    
    # Check if SpeechBrain is available
    if not SPEECHBRAIN_AVAILABLE:
        print("\nERROR: SpeechBrain is not installed.")
        print("Install it with: pip install speechbrain")
        return False
    
    print("\n1. Loading VAD model...")
    try:
        vad = SpeechBrainVAD()
        print(f"   Model loaded successfully!")
        print(f"   Device: {vad.device}")
        print(f"   Source: {vad.source}")
        print(f"   Savedir: {vad.savedir}")
    except Exception as e:
        print(f"   ERROR: Failed to load model: {e}")
        return False
    
    # Create a test audio file
    print("\n2. Creating test audio file...")
    try:
        import torch
        import torchaudio
        
        # Create a 3-second test signal with speech-like characteristics
        sample_rate = 16000
        duration = 3.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create a signal with alternating speech-like (modulated) and silence regions
        # 0-1s: speech-like
        # 1-2s: silence
        # 2-3s: speech-like
        signal = torch.zeros_like(t)
        
        # Speech regions: modulated sine wave
        speech_mask_1 = (t >= 0.0) & (t < 1.0)
        speech_mask_2 = (t >= 2.0) & (t < 3.0)
        
        # Modulated sine wave to simulate speech
        carrier = torch.sin(2 * np.pi * 200 * t)  # 200 Hz carrier
        modulator = 0.5 + 0.5 * torch.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
        signal[speech_mask_1 | speech_mask_2] = (
            carrier * modulator
        )[speech_mask_1 | speech_mask_2]
        
        # Add some noise
        signal += 0.01 * torch.randn_like(signal)
        
        # Normalize
        signal = signal / (signal.abs().max() + 1e-8)
        
        # Save as wav file
        test_file = "test_audio.wav"
        torchaudio.save(test_file, signal.unsqueeze(0), sample_rate)
        print(f"   Created: {test_file}")
        print(f"   Duration: {duration}s, Sample rate: {sample_rate}Hz")
        
    except Exception as e:
        print(f"   ERROR: Failed to create test audio: {e}")
        return False
    
    # Test get_frame_probs
    print("\n3. Testing get_frame_probs()...")
    try:
        probs, times = vad.get_frame_probs(test_file)
        print(f"   Success!")
        print(f"   Frame probabilities shape: {probs.shape}")
        print(f"   Times shape: {times.shape}")
        print(f"   Prob range: [{probs.min():.3f}, {probs.max():.3f}]")
        print(f"   Mean prob: {probs.mean():.3f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test get_segments
    print("\n4. Testing get_segments()...")
    try:
        segments = vad.get_segments(test_file)
        print(f"   Success!")
        print(f"   Found {len(segments)} speech segment(s):")
        for i, (start, end) in enumerate(segments, 1):
            print(f"     Segment {i}: {start:.3f}s - {end:.3f}s "
                  f"(duration: {end-start:.3f}s)")
        
        # Test with different options
        print("\n   Testing with apply_energy_vad=True...")
        segments_energy = vad.get_segments(
            test_file,
            apply_energy_vad=True
        )
        print(f"   Found {len(segments_energy)} segment(s) with energy VAD")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test full prediction
    print("\n5. Testing predict()...")
    try:
        result = vad.predict(test_file)
        print(f"   Success!")
        print(f"   Total speech duration: {result['total_speech_duration']:.3f}s")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Cleanup
    print("\n6. Cleaning up...")
    try:
        os.remove(test_file)
        print(f"   Removed: {test_file}")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_speechbrain_vad()
