#!/usr/bin/env python3
"""
Utility package for VAD distillation project.

This package contains shared utility modules used across the project
for audio processing, configuration loading, and common operations.
"""

from utils.audio import load_audio, get_audio_duration

__all__ = ["load_audio", "get_audio_duration"]
