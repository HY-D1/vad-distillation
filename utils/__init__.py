#!/usr/bin/env python3
"""
Utility package for VAD distillation project.

This package contains shared utility modules used across the project
for audio processing, configuration loading, and common operations.
"""

# Audio utilities
from utils.audio import load_audio, get_audio_duration

# Common utilities (migrated from root utils.py)
from utils.common import (
    setup_project_paths,
    get_project_root,
    load_config,
    save_config,
    load_json,
    save_json,
    ensure_dir,
    format_duration,
    format_size,
    count_parameters,
    get_model_size_mb,
    format_model_size,
    get_device,
    compute_metrics,
)

__all__ = [
    # Audio utilities
    "load_audio",
    "get_audio_duration",
    # Common utilities
    "setup_project_paths",
    "get_project_root",
    "load_config",
    "save_config",
    "load_json",
    "save_json",
    "ensure_dir",
    "format_duration",
    "format_size",
    "count_parameters",
    "get_model_size_mb",
    "format_model_size",
    "get_device",
    "compute_metrics",
]
