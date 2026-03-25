#!/usr/bin/env python3
"""
Personal VAD Milestone: Student Model vs Energy Baseline Comparison

This script:
1. Loads the trained TinyVAD student model from checkpoint
2. Runs inference on the same utterances as the energy baseline
3. Saves frame-level predictions as .npy files
4. Compares student model metrics with energy baseline
5. Generates a summary report

Usage:
    python milestones/personal_vad/scripts/run_my_experiment.py \
        --checkpoint outputs/quick_test/checkpoints/fold_F01_latest_best.pt \
        --manifest manifests/torgo_pilot.csv \
        --baseline-dir outputs/personal_vad/baseline_energy_50/frame_probs \
        --output-dir outputs/personal_vad/comparison
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from audio_loader import load_audio_mono
from models.tinyvad_student import TinyVAD


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[TinyVAD, Dict]:
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {
        'n_mels': 40,
        'cnn_channels': [14, 28],
        'gru_hidden': 32,
        'gru_layers': 2,
        'dropout': 0.1
    })
    
    # Create model
    model = TinyVAD(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.count_parameters():,} parameters")
    print(f"Model size: {model.get_model_size_kb():.1f} KB")
    
    return model, checkpoint


def load_manifest(manifest_path: str, max_utterances: Optional[int] = None) -> List[Dict]:
    """Load manifest CSV."""
    print(f"Loading manifest: {manifest_path}")
    
    utterances = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            utterances.append(row)
    
    if max_utterances:
        utterances = utterances[:max_utterances]
    
    print(f"Loaded {len(utterances)} utterances")
    return utterances


def run_inference(
    model: TinyVAD,
    utterances: List[Dict],
    device: torch.device,
    output_dir: Path,
    baseline_dir: Path,
    overwrite: bool = False
) -> Dict:
    """Run inference on utterances and save predictions."""
    
    frame_probs_dir = output_dir / 'student_frame_probs'
    frame_probs_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'predictions': []
    }
    
    # Find which utterances have baseline predictions
    baseline_files = set()
    if baseline_dir.exists():
        baseline_files = {f.stem for f in baseline_dir.glob('*.npy')}
        print(f"Found {len(baseline_files)} baseline predictions")
    
    # Filter to utterances that have baseline predictions
    utterances_to_process = []
    for utt in utterances:
        utt_id = f"{utt['speaker_id']}_{utt['session']}_{int(utt['utt_id']):04d}"
        if utt_id in baseline_files:
            utterances_to_process.append((utt, utt_id))
    
    print(f"Processing {len(utterances_to_process)} utterances with baseline matches...")
    
    for i, (utt, utt_id) in enumerate(utterances_to_process):
        audio_path = utt['path']
        
        try:
            # Check if already processed
            output_path = frame_probs_dir / f"{utt_id}.npy"
            if output_path.exists() and not overwrite:
                results['skipped'] += 1
                continue
            
            # Load audio with soundfile/librosa to avoid torchaudio runtime codec issues.
            audio_np, _ = load_audio_mono(audio_path, target_sr=16000)
            
            with torch.no_grad():
                probs = model.predict(audio_np, device=device, return_numpy=True)
            
            # Save predictions
            np.save(output_path, probs)
            
            results['processed'] += 1
            results['predictions'].append({
                'utt_id': utt_id,
                'audio_path': audio_path,
                'num_frames': len(probs),
                'mean_prob': float(np.mean(probs)),
                'max_prob': float(np.max(probs))
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(utterances_to_process)} utterances")
                
        except Exception as e:
            print(f"  Error processing {utt_id}: {e}")
            results['failed'] += 1
    
    return results


def compare_with_baseline(
    student_dir: Path,
    baseline_dir: Path,
    output_dir: Path
) -> Dict:
    """Compare student predictions with baseline."""
    
    print("\nComparing student model with energy baseline...")
    
    student_files = list(student_dir.glob('*.npy'))
    baseline_files = list(baseline_dir.glob('*.npy'))
    
    # Match files by utterance ID
    student_dict = {f.stem: f for f in student_files}
    baseline_dict = {f.stem: f for f in baseline_files}
    
    common_ids = set(student_dict.keys()) & set(baseline_dict.keys())
    
    print(f"Student predictions: {len(student_files)}")
    print(f"Baseline predictions: {len(baseline_files)}")
    print(f"Common utterances: {len(common_ids)}")
    
    comparison = {
        'common_utterances': len(common_ids),
        'student_only': len(student_dict) - len(common_ids),
        'baseline_only': len(baseline_dict) - len(common_ids),
        'utterance_comparisons': []
    }
    
    # Compare predictions for common utterances
    for utt_id in sorted(common_ids)[:10]:  # Sample first 10 for detailed comparison
        student_probs = np.load(student_dict[utt_id])
        baseline_probs = np.load(baseline_dict[utt_id])
        
        # Align lengths (student may have different frame rate)
        min_len = min(len(student_probs), len(baseline_probs))
        student_probs = student_probs[:min_len]
        baseline_probs = baseline_probs[:min_len]
        
        comparison['utterance_comparisons'].append({
            'utt_id': utt_id,
            'num_frames': min_len,
            'student_mean_prob': float(np.mean(student_probs)),
            'baseline_mean_prob': float(np.mean(baseline_probs)),
            'correlation': float(np.corrcoef(student_probs, baseline_probs)[0, 1]) if min_len > 1 else 0.0,
            'mse': float(np.mean((student_probs - baseline_probs) ** 2))
        })
    
    return comparison


def generate_summary(
    inference_results: Dict,
    comparison_results: Dict,
    checkpoint_metrics: Dict,
    output_dir: Path
):
    """Generate summary report."""
    
    summary = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'inference': inference_results,
        'comparison': comparison_results,
        'checkpoint_metrics': checkpoint_metrics,
        'notes': []
    }
    
    # Add notes
    notes = []
    
    if inference_results['processed'] == 0:
        notes.append("No utterances were processed. Check audio file paths.")
    
    if comparison_results['common_utterances'] == 0:
        notes.append("No common utterances found between student and baseline.")
        notes.append("This is expected if the manifest and baseline outputs don't overlap.")
    
    val_auc = checkpoint_metrics.get('val_auc')
    if isinstance(val_auc, (int, float)) and val_auc > 0.9:
        notes.append(f"Student model achieved excellent Val AUC: {val_auc:.4f}")
    
    summary['notes'] = notes
    
    # Save JSON summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save text summary
    text_path = output_dir / 'summary.txt'
    with open(text_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PERSONAL VAD MILESTONE: MODEL VS BASELINE COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("INFERENCE RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Utterances processed: {inference_results['processed']}\n")
        f.write(f"Utterances skipped: {inference_results['skipped']}\n")
        f.write(f"Utterances failed: {inference_results['failed']}\n\n")
        
        f.write("CHECKPOINT METRICS\n")
        f.write("-" * 40 + "\n")
        for key, value in checkpoint_metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("COMPARISON WITH BASELINE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Common utterances: {comparison_results['common_utterances']}\n")
        f.write(f"Student only: {comparison_results['student_only']}\n")
        f.write(f"Baseline only: {comparison_results['baseline_only']}\n\n")
        
        if comparison_results['utterance_comparisons']:
            f.write("SAMPLE COMPARISONS (first 10):\n")
            f.write("-" * 40 + "\n")
            for comp in comparison_results['utterance_comparisons']:
                f.write(f"\nUtterance: {comp['utt_id']}\n")
                f.write(f"  Frames: {comp['num_frames']}\n")
                f.write(f"  Student mean prob: {comp['student_mean_prob']:.3f}\n")
                f.write(f"  Baseline mean prob: {comp['baseline_mean_prob']:.3f}\n")
                f.write(f"  Correlation: {comp['correlation']:.3f}\n")
                f.write(f"  MSE: {comp['mse']:.6f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("NOTES\n")
        f.write("=" * 60 + "\n")
        for note in notes:
            f.write(f"- {note}\n")
    
    print(f"\nSummary saved to:")
    print(f"  JSON: {summary_path}")
    print(f"  Text: {text_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare TinyVAD student model with energy baseline"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/production_cuda/checkpoints/fold_F01_latest_best.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='manifests/torgo_pilot.csv',
        help='Path to manifest CSV'
    )
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default='outputs/personal_vad/baseline_energy_50/frame_probs',
        help='Directory with baseline predictions'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/personal_vad/comparison',
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/mps/cpu). Auto-detected if not specified.'
    )
    parser.add_argument(
        '--max-utterances',
        type=int,
        default=None,
        help='Maximum utterances to process (for testing)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Recompute predictions even if output files already exist'
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Extract metrics from checkpoint
    checkpoint_metrics = {
        'epoch': checkpoint.get('epoch', 0),
        'val_auc': checkpoint.get('val_auc'),
        'model_size_mb': checkpoint.get('model_size_mb', 0.0),
        'num_parameters': checkpoint.get('num_parameters', 0)
    }
    
    # Load manifest
    utterances = load_manifest(args.manifest, args.max_utterances)
    
    # Run inference
    baseline_dir = Path(args.baseline_dir)
    inference_results = run_inference(
        model, utterances, device, output_dir, baseline_dir, overwrite=args.overwrite
    )
    
    # Compare with baseline
    student_dir = output_dir / 'student_frame_probs'
    comparison_results = compare_with_baseline(student_dir, baseline_dir, output_dir)
    
    # Generate summary
    generate_summary(inference_results, comparison_results, checkpoint_metrics, output_dir)
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
