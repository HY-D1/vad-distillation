"""
Export command for VAD distillation.

Export trained models to various formats.
"""

import argparse
import sys
from pathlib import Path

from cli.utils import (
    ensure_project_root,
    print_error,
    print_info,
    print_success,
    with_project_root
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add export-specific arguments."""
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Config path (alternative to checkpoint)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['onnx', 'torchscript', 'all'],
        required=True,
        help='Export format'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path'
    )
    parser.add_argument(
        '--input-shape',
        nargs=3,
        type=int,
        default=[1, 100, 40],
        help='Example input shape (batch time n_mels)'
    )


@with_project_root
def main(args: argparse.Namespace) -> int:
    """
    Execute export command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("="*60)
    print("MODEL EXPORT")
    print("="*60)
    
    # Validate arguments
    if not args.checkpoint and not args.config:
        print_error(
            "Must specify either --checkpoint or --config",
            "Example: vad.py export --checkpoint outputs/checkpoints/fold_F01_best.pt --format onnx"
        )
        return 2
    
    # Import model
    try:
        from models.tinyvad_student import TinyVAD, create_student_model
        import torch
        import yaml
    except ImportError as e:
        print_error(f"Failed to import required modules: {e}")
        return 3
    
    # Load model
    model, model_name = load_model(args.checkpoint, args.config)
    if model is None:
        return 1
    
    print_success(f"Model loaded: {model_name}")
    print_info(f"Parameters: {model.count_parameters():,}")
    print_info(f"Size: {model.get_model_size_kb():.2f} KB")
    
    # Determine output paths
    output_base = args.output or f"exports/{model_name}"
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    input_shape = tuple(args.input_shape)
    success = True
    
    if args.format in ('onnx', 'all'):
        onnx_path = f"{output_base}.onnx"
        print(f"\nExporting to ONNX: {onnx_path}")
        try:
            model.export_onnx(onnx_path, input_shape=input_shape)
            print_success(f"ONNX export complete: {Path(onnx_path).stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print_error(f"ONNX export failed: {e}")
            success = False
    
    if args.format in ('torchscript', 'all'):
        ts_path = f"{output_base}.pt"
        print(f"\nExporting to TorchScript: {ts_path}")
        try:
            model.export_torchscript(ts_path, input_shape=input_shape)
            print_success(f"TorchScript export complete: {Path(ts_path).stat().st_size / 1024:.2f} KB")
        except Exception as e:
            print_error(f"TorchScript export failed: {e}")
            success = False
    
    return 0 if success else 1


def load_model(checkpoint_path: str, config_path: str):
    """
    Load model from checkpoint or config.
    
    Args:
        checkpoint_path: Path to checkpoint
        config_path: Path to config
        
    Returns:
        Tuple of (model, name)
    """
    from models.tinyvad_student import create_student_model
    import torch
    import yaml
    
    model = None
    model_name = "model"
    
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            print_error(f"Checkpoint not found: {checkpoint_path}")
            return None, None
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get config from checkpoint or separate file
        config = checkpoint.get('config', {})
        if not config and config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Create model
        model_config = config.get('model', {})
        model = create_student_model(model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract name from path
        model_name = checkpoint_file.stem.replace('_best', '').replace('_latest', '')
        
    elif config_path:
        # Create from config only
        config_file = Path(config_path)
        if not config_file.exists():
            print_error(f"Config not found: {config_path}")
            return None, None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config.get('model', {})
        model = create_student_model(model_config)
        model_name = config_file.stem
    
    model.eval()
    return model, model_name
