"""Energy-based Voice Activity Detection baseline."""

import numpy as np
import librosa
from scipy.ndimage import median_filter
from typing import List, Tuple


class EnergyVAD:
    """Energy-based Voice Activity Detection.
    
    Uses short-term energy with hysteresis thresholding for speech detection.
    
    Parameters
    ----------
    frame_hop_ms : int, default 10
        Frame hop length in milliseconds.
    threshold : float, default 0.5
        Base energy threshold for speech detection (normalized energy).
    hysteresis_high : float, default 0.6
        High threshold for hysteresis (enter speech state).
    hysteresis_low : float, default 0.4
        Low threshold for hysteresis (exit speech state).
    min_speech_dur : float, default 0.25
        Minimum duration of speech segments in seconds.
    min_silence_dur : float, default 0.25
        Minimum silence duration between segments; gaps smaller than this
        will be merged.
    smoothing_window : int, default 3
        Window size for median smoothing of energy values.
    """
    
    def __init__(
        self,
        frame_hop_ms: int = 10,
        threshold: float = 0.5,
        hysteresis_high: float = 0.6,
        hysteresis_low: float = 0.4,
        min_speech_dur: float = 0.25,
        min_silence_dur: float = 0.25,
        smoothing_window: int = 3,
    ):
        self.frame_hop_ms = frame_hop_ms
        self.threshold = threshold
        self.hysteresis_high = hysteresis_high
        self.hysteresis_low = hysteresis_low
        self.min_speech_dur = min_speech_dur
        self.min_silence_dur = min_silence_dur
        self.smoothing_window = smoothing_window
    
    def get_frame_probs(self, audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        """Compute frame-level speech probabilities from audio energy.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal.
        sr : int, default 16000
            Sample rate.
        
        Returns
        -------
        probs : np.ndarray
            Normalized energy values (speech probabilities) for each frame.
        times : np.ndarray
            Timestamp (in seconds) for each frame center.
        """
        # Frame parameters: 25ms window, specified hop
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(self.frame_hop_ms / 1000 * sr)
        
        # Compute short-term energy (RMS)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Normalize energy to [0, 1]
        if rms.max() > 0:
            probs = rms / rms.max()
        else:
            probs = rms.copy()
        
        # Apply median smoothing
        if self.smoothing_window > 1:
            probs = median_filter(probs, size=self.smoothing_window)
        
        # Compute frame timestamps (center of each frame)
        times = librosa.frames_to_time(
            np.arange(len(probs)),
            sr=sr,
            hop_length=hop_length
        )
        
        return probs, times
    
    def get_segments(self, audio: np.ndarray, sr: int = 16000) -> List[Tuple[float, float]]:
        """Detect speech segments using hysteresis thresholding.
        
        Parameters
        ----------
        audio : np.ndarray
            Audio signal.
        sr : int, default 16000
            Sample rate.
        
        Returns
        -------
        segments : List[Tuple[float, float]]
            List of (start_time, end_time) tuples for detected speech segments.
        """
        probs, times = self.get_frame_probs(audio, sr)
        hop_length = int(self.frame_hop_ms / 1000 * sr)
        frame_dur = hop_length / sr
        
        # Hysteresis thresholding
        is_speech = np.zeros(len(probs), dtype=bool)
        in_speech = False
        
        for i, prob in enumerate(probs):
            if in_speech:
                if prob < self.hysteresis_low:
                    in_speech = False
            else:
                if prob > self.hysteresis_high:
                    in_speech = True
            is_speech[i] = in_speech
        
        # Convert to segments
        segments = []
        start_idx = None
        
        for i, speech in enumerate(is_speech):
            if speech and start_idx is None:
                # Speech start
                start_idx = i
            elif not speech and start_idx is not None:
                # Speech end
                segments.append((times[start_idx], times[i]))
                start_idx = None
        
        # Handle case where audio ends during speech
        if start_idx is not None:
            segments.append((times[start_idx], times[-1] + frame_dur))
        
        # Merge close segments (gaps smaller than min_silence_dur)
        if len(segments) > 1:
            merged = [segments[0]]
            for start, end in segments[1:]:
                prev_start, prev_end = merged[-1]
                if start - prev_end < self.min_silence_dur:
                    # Merge with previous segment
                    merged[-1] = (prev_start, end)
                else:
                    merged.append((start, end))
            segments = merged
        
        # Remove short segments (shorter than min_speech_dur)
        segments = [
            (start, end) for start, end in segments
            if end - start >= self.min_speech_dur
        ]
        
        return segments


if __name__ == "__main__":
    # Test code - runs on a sample file
    import sys
    
    # Try to find a sample audio file
    test_files = [
        "test.wav",
        "sample.wav", 
        "audio.wav",
        "../test.wav",
        "../sample.wav",
        "../audio.wav",
    ]
    
    # Also check for any wav files in current directory
    try:
        import glob
        wav_files = glob.glob("*.wav") + glob.glob("../*.wav") + glob.glob("data/**/*.wav", recursive=True)
        test_files = list(dict.fromkeys(test_files + wav_files))  # Remove duplicates
    except Exception:
        pass
    
    audio_file = None
    for f in test_files:
        try:
            import os
            if os.path.exists(f):
                audio_file = f
                break
        except Exception:
            continue
    
    if audio_file is None:
        print("No test audio file found. Please provide a .wav file.")
        print("Usage: python energy_vad.py <path_to_audio.wav>")
        
        # Create synthetic test audio for demonstration
        print("\nGenerating synthetic test audio...")
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create signal with alternating speech-like and silence regions
        # Speech: higher energy with some modulation
        # Silence: low energy noise
        audio = np.zeros_like(t)
        
        # Segment 1: 0.5-1.5s (speech)
        mask = (t >= 0.5) & (t < 1.5)
        audio[mask] = 0.3 * np.sin(2 * np.pi * 200 * t[mask]) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t[mask]))
        
        # Segment 2: 2.0-3.0s (speech)
        mask = (t >= 2.0) & (t < 3.0)
        audio[mask] = 0.4 * np.sin(2 * np.pi * 250 * t[mask]) * (1 + 0.3 * np.sin(2 * np.pi * 7 * t[mask]))
        
        # Segment 3: 3.5-4.2s (speech)
        mask = (t >= 3.5) & (t < 4.2)
        audio[mask] = 0.25 * np.sin(2 * np.pi * 180 * t[mask]) * (1 + 0.4 * np.sin(2 * np.pi * 4 * t[mask]))
        
        # Add background noise
        audio += 0.01 * np.random.randn(len(t))
        
        print(f"Synthetic audio: {duration}s duration, 3 speech segments")
    else:
        print(f"Loading audio file: {audio_file}")
        audio, sr = librosa.load(audio_file, sr=16000)
    
    # Run VAD
    print("\nRunning EnergyVAD...")
    vad = EnergyVAD()
    
    # Get frame probabilities
    probs, times = vad.get_frame_probs(audio, sr)
    print(f"  Frames: {len(probs)}, Duration: {len(audio)/sr:.2f}s")
    print(f"  Energy range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Get segments
    segments = vad.get_segments(audio, sr)
    print(f"\nDetected {len(segments)} speech segment(s):")
    for i, (start, end) in enumerate(segments):
        print(f"  Segment {i+1}: {start:.3f}s - {end:.3f}s (duration: {end-start:.3f}s)")
    
    # Summary statistics
    total_speech = sum(end - start for start, end in segments)
    print(f"\nTotal speech time: {total_speech:.3f}s ({100*total_speech/(len(audio)/sr):.1f}%)")
