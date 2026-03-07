"""
Validation command for VAD distillation.

Validates data, configurations, and environment setup.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from cli.config import Config, get_all_folds, get_fold_config
from cli.utils import (
    ensure_project_root,
    print_error,
    print_info,
    print_success,
    print_warning,
    with_project_root
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add validate-specific arguments."""
    parser.add_argument(
        '--data',
        action='store_true',
        help='Validate data only'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Validate specific config file'
    )
    parser.add_argument(
        '--fold',
        type=str,
        default=None,
        help='Validate specific fold structure'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full validation (data + config + environment)'
    )


@with_project_root
def main(args: argparse.Namespace) -> int:
    """
    Execute validation command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("="*60)
    print("VAD DISTILLATION - VALIDATION")
    print("="*60)
    
    all_valid = True
    
    # Determine what to validate
    validate_data = args.data or args.full or (not args.config and not args.fold)
    validate_config = args.config or args.full
    validate_fold = args.fold or args.full
    validate_env = args.full or (not args.data and not args.config and not args.fold)
    
    # Environment validation
    if validate_env:
        print("\n[1/5] Environment Validation")
        print("-"*40)
        env_valid = validate_environment()
        all_valid = all_valid and env_valid
    
    # Data validation
    if validate_data:
        print("\n[2/5] Data Validation")
        print("-"*40)
        data_valid = validate_data_setup()
        all_valid = all_valid and data_valid
    
    # Config validation
    if validate_config:
        print("\n[3/5] Configuration Validation")
        print("-"*40)
        config_valid = validate_configuration(args.config)
        all_valid = all_valid and config_valid
    
    # Fold validation
    if validate_fold:
        print("\n[4/5] Fold Validation")
        print("-"*40)
        fold_valid = validate_folds(args.fold)
        all_valid = all_valid and fold_valid
    
    # Teacher probabilities validation
    if validate_data:
        print("\n[5/5] Teacher Probabilities Validation")
        print("-"*40)
        teacher_valid = validate_teacher_probs()
        all_valid = all_valid and teacher_valid
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if all_valid:
        print_success("All checks passed! Project is ready.")
        return 0
    else:
        print_error("Some checks failed. Please review the output above.")
        return 5


def validate_environment() -> bool:
    """Validate Python environment and dependencies."""
    valid = True
    
    # Check Python version
    if sys.version_info < (3, 8):
        print_error("Python 3.8+ required", "Please upgrade Python")
        valid = False
    else:
        print_success(f"Python {sys.version.split()[0]} OK")
    
    # Check required packages
    required = ['torch', 'torchaudio', 'numpy', 'yaml', 'tqdm']
    for package in required:
        try:
            __import__(package)
            print_success(f"Package '{package}' OK")
        except ImportError:
            print_error(f"Package '{package}' not found")
            valid = False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print_info("CUDA not available (CPU mode)")
    except ImportError:
        pass
    
    return valid


def validate_data_setup() -> bool:
    """Validate TORGO data setup."""
    valid = True
    
    # Import existing validation script
    try:
        import scripts.data.validate_torgo_setup as validator
        
        # Use the existing validation logic
        data_dir = Path('data/torgo_raw')
        if not data_dir.exists():
            print_error(
                "Data directory not found",
                "Download TORGO dataset to data/torgo_raw/"
            )
            return False
        
        structure = validator.check_directory_structure(data_dir)
        
        if not structure['exists']:
            print_error("Data directory does not exist")
            return False
        
        print_success(f"Found {len(structure['speakers'])} speakers")
        print_info(f"Found {len(structure['wav_files'])} audio files")
        
        if structure['missing_components']:
            print_warning(f"{len(structure['missing_components'])} missing components")
        
    except Exception as e:
        print_error(f"Data validation failed: {e}")
        valid = False
    
    return valid


def validate_configuration(config_path: Optional[str]) -> bool:
    """Validate configuration file."""
    valid = True
    
    if config_path:
        configs_to_check = [config_path]
    else:
        configs_to_check = ['configs/production.yaml', 'configs/pilot.yaml']
    
    for path in configs_to_check:
        config_file = Path(path)
        if not config_file.exists():
            if config_path:  # Only error if explicitly specified
                print_error(f"Config not found: {path}")
                valid = False
            continue
        
        try:
            config = Config(path)
            print_success(f"Config valid: {path}")
            
            # Check required fields
            required = ['model', 'alpha', 'temperature']
            for field in required:
                if field not in config.to_dict():
                    print_warning(f"Missing field '{field}' in {path}")
                    
        except Exception as e:
            print_error(f"Invalid config {path}: {e}")
            valid = False
    
    return valid


def validate_folds(fold_id: Optional[str]) -> bool:
    """Validate fold configurations."""
    valid = True
    
    splits_dir = Path('splits')
    if not splits_dir.exists():
        print_error(
            "Splits directory not found",
            "Run 'python scripts/data/generate_loso_splits.py'"
        )
        return False
    
    if fold_id:
        # Validate specific fold
        try:
            config = get_fold_config(fold_id)
            print_success(f"Fold {fold_id} valid")
            print_info(f"  Train: {len(config.get('train', []))} speakers")
            print_info(f"  Val: {config.get('val', 'N/A')}")
            print_info(f"  Test: {config.get('test', 'N/A')}")
        except FileNotFoundError:
            print_error(f"Fold {fold_id} not found")
            valid = False
    else:
        # Validate all folds
        folds = get_all_folds(splits_dir)
        print_info(f"Found {len(folds)} fold configurations")
        
        for fold in folds:
            try:
                get_fold_config(fold)
            except Exception as e:
                print_error(f"Fold {fold} invalid: {e}")
                valid = False
        
        if valid:
            print_success(f"All {len(folds)} folds valid")
    
    return valid


def validate_teacher_probs() -> bool:
    """Validate teacher probability files."""
    valid = True
    
    teacher_dir = Path('teacher_probs')
    if not teacher_dir.exists():
        print_error(
            "Teacher probabilities directory not found",
            "Run 'python scripts/data/cache_teacher.py' to generate"
        )
        return False
    
    # Count .npz files
    prob_files = list(teacher_dir.glob('*.npz'))
    if len(prob_files) == 0:
        print_error("No teacher probability files found")
        return False
    
    print_success(f"Found {len(prob_files)} teacher probability files")
    
    # Sample check
    import numpy as np
    sample_file = prob_files[0]
    try:
        data = np.load(sample_file)
        print_info(f"Sample file {sample_file.name}: {data['probs'].shape}")
    except Exception as e:
        print_warning(f"Could not read sample file: {e}")
    
    return valid
