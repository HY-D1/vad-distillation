"""Baseline VAD implementations."""

from .energy_vad import EnergyVAD
from .speechbrain_vad import SpeechBrainVAD

__all__ = ["EnergyVAD", "SpeechBrainVAD"]
