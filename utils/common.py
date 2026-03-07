#!/usr/bin/env python3
"""
Common utility functions for VAD distillation project.

This module contains shared utility functions used across the project
for configuration loading, path handling, and common operations.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def setup_project_paths() -> Path:
    """
    Setup sys.path to allow imports from project root.
    
    This function should be called at the beginning of scripts that
    need to import from data/ or models/ packages.
    
    Returns:
        Path to project root directory
    """
    # Get the project root (parent of the current script's directory)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    # Add to sys.path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.resolve()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Parsed JSON data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is invalid JSON
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], output_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        output_path: Path to save JSON file
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "1h 30m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_size(size_bytes: float) -> str:
    """
    Format size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# Import torch for model utility functions (with fallback if not available)
try:
    import torch
    import torch.nn as nn
    
    def count_parameters(model: nn.Module) -> int:
        """Count trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_model_size_mb(model: nn.Module) -> float:
        """Get model size in megabytes."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def format_model_size(model: nn.Module) -> str:
        """
        Format model size as a human-readable string.
        
        Args:
            model: PyTorch model
            
        Returns:
            Formatted string (e.g., "473.25 KB")
        """
        size_mb = get_model_size_mb(model)
        return format_size(size_mb * 1024 * 1024)
        
except ImportError:
    torch = None
    
    def count_parameters(model) -> int:
        """Count trainable parameters (torch not available)."""
        raise ImportError("torch is required for count_parameters")
    
    def get_model_size_mb(model) -> float:
        """Get model size in MB (torch not available)."""
        raise ImportError("torch is required for get_model_size_mb")
    
    def format_model_size(model) -> str:
        """Format model size (torch not available)."""
        raise ImportError("torch is required for format_model_size")


__all__ = [
    'setup_project_paths',
    'get_project_root',
    'load_config',
    'save_config',
    'load_json',
    'save_json',
    'ensure_dir',
    'format_duration',
    'format_size',
    'count_parameters',
    'get_model_size_mb',
    'format_model_size',
]
