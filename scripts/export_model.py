#!/usr/bin/env python3
"""
Export trained TinyVAD model to TorchScript and ONNX formats.

Usage:
    python scripts/export_model.py --checkpoint outputs/production_cuda/checkpoints/fold_F01_latest_best.pt --format torchscript
    python scripts/export_model.py --checkpoint outputs/production_cuda/checkpoints/fold_F01_latest_best.pt --format onnx
    python scripts/export_model.py --checkpoint outputs/production_cuda/checkpoints/fold_F01_latest_best.pt --format both
"""

import argparse
import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import create_student_model


def export_torchscript(checkpoint_path, output_path):
    """Export model to TorchScript format."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = create_student_model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 100, 40)
    
    # Trace and save
    print("Tracing model...")
    traced = torch.jit.trace(model, example_input)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    traced.save(output_path)
    print(f"✓ Exported to: {output_path}")
    
    # Verify
    loaded = torch.jit.load(output_path)
    test_output = loaded(example_input)
    print(f"✓ Verification passed: output shape = {test_output.shape}")
    
    return output_path


def export_onnx(checkpoint_path, output_path):
    """Export model to ONNX format."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = create_student_model()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 100, 40)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=['mel_spectrogram'],
        output_names=['speech_probability'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch', 1: 'time'},
            'speech_probability': {0: 'batch', 1: 'time'}
        },
        opset_version=11
    )
    print(f"✓ Exported to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export TinyVAD model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., outputs/production_cuda/checkpoints/fold_F01_latest_best.pt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: deploy/model.pt or deploy/model.onnx)')
    parser.add_argument('--format', type=str, choices=['torchscript', 'onnx', 'both'], default='both',
                        help='Export format')
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        return 1
    
    exported = []
    
    # Export TorchScript
    if args.format in ['torchscript', 'both']:
        output_path = args.output or 'deploy/model.pt'
        export_torchscript(args.checkpoint, output_path)
        exported.append(output_path)
    
    # Export ONNX
    if args.format in ['onnx', 'both']:
        output_path = args.output.replace('.pt', '.onnx') if args.output else 'deploy/model.onnx'
        export_onnx(args.checkpoint, output_path)
        exported.append(output_path)
    
    print("\n" + "="*50)
    print("Export completed successfully!")
    print("="*50)
    for path in exported:
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path} ({size_kb:.1f} KB)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
