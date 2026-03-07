"""
Configuration management for VAD CLI.

Implements configuration hierarchy:
    CLI args > Environment vars > Config file > Defaults
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# Default configuration
DEFAULT_CONFIG = {
    'seed': 6140,
    'device': 'auto',
    'num_workers': 4,
    'data_dir': 'data/torgo_raw',
    'manifest': 'manifests/torgo_sentences.csv',
    'teacher_probs_dir': 'teacher_probs/',
    'output_dir': 'outputs/',
    'splits_dir': 'splits/',
}

# Environment variable mapping
ENV_VAR_MAP = {
    'VAD_CONFIG': 'config_path',
    'VAD_OUTPUT_DIR': 'output_dir',
    'VAD_DEVICE': 'device',
    'VAD_DATA_DIR': 'data_dir',
    'VAD_SEED': 'seed',
}


class Config:
    """Configuration manager with hierarchy support."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file
            overrides: Dictionary of override values (highest priority)
        """
        self._config = {}
        self._load_defaults()
        self._load_from_file(config_path)
        self._load_from_env()
        if overrides:
            self._apply_overrides(overrides)
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self._config.update(DEFAULT_CONFIG)
    
    def _load_from_file(self, config_path: Optional[Union[str, Path]]) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try to find default config
            default_paths = [
                'configs/production.yaml',
                'configs/pilot.yaml',
            ]
            for path in default_paths:
                if Path(path).exists():
                    config_path = path
                    break
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._config.update(file_config)
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        for env_var, config_key in ENV_VAR_MAP.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
                self._config[config_key] = value
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply command-line overrides (highest priority)."""
        self._config.update(overrides)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
    
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Get nested configuration value.
        
        Example:
            config.get_nested('model', 'gru_hidden')
        """
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style assignment."""
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config


def get_all_folds(splits_dir: Union[str, Path] = 'splits') -> List[str]:
    """
    Get list of all available fold IDs.
    
    Args:
        splits_dir: Directory containing fold JSON files
        
    Returns:
        List of fold IDs (e.g., ['F01', 'F03', ...])
    """
    splits_path = Path(splits_dir)
    if not splits_path.exists():
        return []
    
    folds = []
    for file_path in splits_path.glob('fold_*.json'):
        # Extract fold ID from filename (e.g., fold_F01.json -> F01)
        fold_id = file_path.stem.replace('fold_', '')
        folds.append(fold_id)
    
    return sorted(folds)


def get_fold_config(fold_id: str, splits_dir: Union[str, Path] = 'splits') -> Dict[str, Any]:
    """
    Load fold configuration from JSON file.
    
    Args:
        fold_id: Fold identifier (e.g., 'F01')
        splits_dir: Directory containing fold files
        
    Returns:
        Fold configuration dictionary
    """
    import json
    
    fold_path = Path(splits_dir) / f'fold_{fold_id}.json'
    if not fold_path.exists():
        raise FileNotFoundError(f"Fold configuration not found: {fold_path}")
    
    with open(fold_path, 'r') as f:
        return json.load(f)
