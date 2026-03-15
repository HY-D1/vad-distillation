#!/usr/bin/env python3
"""
Personal VAD Milestone: Energy VAD Threshold Tuning

This script tunes Energy VAD hyperparameters by:
1. Loading audio files from the manifest
2. Running EnergyVAD with different threshold/hysteresis settings
3. Comparing outputs with student model predictions (proxy reference)
4. Selecting the best setting based on agreement metrics

Scoring Method (Proxy-based):
- Uses student model predictions as reference (not ground truth labels)
- Computes MSE and correlation between energy VAD and student predictions
- Lower MSE = better agreement with student model
- Higher positive correlation = more similar behavior

Limitation Note:
- This is proxy-based tuning using student model as reference
- Not true ground-truth tuning (labels not readily available)
- Appropriate for milestone analysis and relative comparison

Usage:
    python scripts/personal/tune_energy_vad.py \
        --manifest manifests/torgo_pilot.csv \
        --student-preds outputs/personal/comparison/student_frame_probs \
        --output-dir outputs/personal/energy_tuning \
        --max-utterances 20
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torchaudio

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from baselines.energy_vad import EnergyVAD


def load_manifest(manifest_path: str, max_utterances: int = None) -> List[Dict]:
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


def load_student_predictions(student_preds_dir: Path) -> Dict[str, np.ndarray]:
    """Load student model predictions."""
    print(f"Loading student predictions from: {student_preds_dir}")
    predictions = {}
    if not student_preds_dir.exists():
        print(f"Warning: Student predictions directory not found: {student_preds_dir}")
        return predictions
    
    for npy_file in student_preds_dir.glob('*.npy'):
        utt_id = npy_file.stem
        predictions[utt_id] = np.load(npy_file)
    
    print(f"Loaded {len(predictions)} student predictions")
    return predictions


def align_frames(source_probs: np.ndarray, target_len: int) -> np.ndarray:
    """
    Align source probabilities to target length using linear interpolation.
    
    Args:
        source_probs: Source frame probabilities
        target_len: Target number of frames
    
    Returns:
        Aligned probabilities
    """
    if len(source_probs) == target_len:
        return source_probs
    
    # Linear interpolation
    old_indices = np.linspace(0, len(source_probs) - 1, len(source_probs))
    new_indices = np.linspace(0, len(source_probs) - 1, target_len)
    aligned = np.interp(new_indices, old_indices, source_probs)
    
    return aligned


def evaluate_setting(
    setting: Dict,
    utterances: List[Dict],
    student_predictions: Dict[str, np.ndarray]
) -> Dict:
    """
    Evaluate one EnergyVAD setting against student predictions.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create EnergyVAD with this setting
    vad = EnergyVAD(
        frame_hop_ms=setting.get('frame_hop_ms', 10),
        threshold=setting.get('threshold', 0.5),
        hysteresis_high=setting.get('hysteresis_high', 0.6),
        hysteresis_low=setting.get('hysteresis_low', 0.4),
        min_speech_dur=setting.get('min_speech_dur', 0.25),
        min_silence_dur=setting.get('min_silence_dur', 0.25),
        smoothing_window=setting.get('smoothing_window', 3)
    )
    
    all_mses = []
    all_correlations = []
    processed = 0
    failed = 0
    
    for utt in utterances:
        # Generate utterance ID
        utt_id = f"{utt['speaker_id']}_{utt['session']}_{int(utt['utt_id']):04d}"
        
        # Skip if no student prediction for this utterance
        if utt_id not in student_predictions:
            continue
        
        student_probs = student_predictions[utt_id]
        
        try:
            # Load audio
            audio_path = utt['path']
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            audio_np = waveform.squeeze().numpy()
            
            # Get energy VAD predictions
            energy_probs, _ = vad.get_frame_probs(audio_np, sr=16000)
            
            # Align to student frame rate
            energy_probs_aligned = align_frames(energy_probs, len(student_probs))
            
            # Compute metrics
            mse = np.mean((energy_probs_aligned - student_probs) ** 2)
            
            # Correlation (handle constant cases)
            if np.std(energy_probs_aligned) > 0 and np.std(student_probs) > 0:
                correlation = np.corrcoef(energy_probs_aligned, student_probs)[0, 1]
            else:
                correlation = 0.0
            
            all_mses.append(mse)
            all_correlations.append(correlation)
            processed += 1
            
        except Exception as e:
            failed += 1
            continue
    
    # Aggregate metrics
    if len(all_mses) > 0:
        avg_mse = np.mean(all_mses)
        avg_correlation = np.mean(all_correlations)
        median_mse = np.median(all_mses)
        median_correlation = np.median(all_correlations)
    else:
        avg_mse = float('inf')
        avg_correlation = -1.0
        median_mse = float('inf')
        median_correlation = -1.0
    
    return {
        'setting': setting,
        'processed': processed,
        'failed': failed,
        'avg_mse': float(avg_mse),
        'avg_correlation': float(avg_correlation),
        'median_mse': float(median_mse),
        'median_correlation': float(median_correlation),
        'all_mses': [float(x) for x in all_mses],
        'all_correlations': [float(x) for x in all_correlations]
    }


def generate_settings_grid() -> List[Dict]:
    """Generate grid of settings to evaluate."""
    settings = []
    
    # Grid search over key parameters
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    hysteresis_highs = [0.5, 0.6, 0.7]
    hysteresis_lows = [0.2, 0.3, 0.4]
    smoothing_windows = [3, 5]
    
    for threshold in thresholds:
        for hysteresis_high in hysteresis_highs:
            for hysteresis_low in hysteresis_lows:
                for smoothing in smoothing_windows:
                    # Ensure hysteresis makes sense
                    if hysteresis_low >= hysteresis_high:
                        continue
                    
                    setting = {
                        'threshold': threshold,
                        'hysteresis_high': hysteresis_high,
                        'hysteresis_low': hysteresis_low,
                        'smoothing_window': smoothing,
                        'frame_hop_ms': 10,
                        'min_speech_dur': 0.25,
                        'min_silence_dur': 0.25
                    }
                    settings.append(setting)
    
    return settings


def find_best_setting(results: List[Dict]) -> Dict:
    """Find best setting based on MSE (lower is better)."""
    # Filter out failed runs
    valid_results = [r for r in results if r['processed'] > 0 and r['avg_mse'] < float('inf')]
    
    if not valid_results:
        return None
    
    # Sort by average MSE (ascending)
    sorted_results = sorted(valid_results, key=lambda x: x['avg_mse'])
    
    return sorted_results[0]


def save_results(results: List[Dict], best_result: Dict, output_dir: Path):
    """Save tuning results."""
    
    # Save full results as JSON
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'num_settings_evaluated': len(results),
            'all_results': results,
            'best_setting': best_result['setting'] if best_result else None,
            'best_metrics': {
                'avg_mse': best_result['avg_mse'],
                'avg_correlation': best_result['avg_correlation'],
                'processed': best_result['processed']
            } if best_result else None
        }, f, indent=2)
    
    # Save CSV summary
    csv_path = output_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        import csv as csv_module
        writer = csv_module.writer(f)
        writer.writerow([
            'threshold', 'hysteresis_high', 'hysteresis_low', 'smoothing_window',
            'avg_mse', 'avg_correlation', 'median_mse', 'median_correlation', 'processed'
        ])
        
        for r in results:
            s = r['setting']
            writer.writerow([
                s['threshold'], s['hysteresis_high'], s['hysteresis_low'], s['smoothing_window'],
                f"{r['avg_mse']:.6f}", f"{r['avg_correlation']:.4f}",
                f"{r['median_mse']:.6f}", f"{r['median_correlation']:.4f}",
                r['processed']
            ])
    
    # Save text summary
    text_path = output_dir / 'summary.txt'
    with open(text_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ENERGY VAD THRESHOLD TUNING RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 70 + "\n")
        f.write("Scoring Method: Proxy-based (agreement with student model)\n")
        f.write("Note: Student model predictions used as reference, not ground truth\n")
        f.write("Metric: MSE (Mean Squared Error) between Energy VAD and Student\n")
        f.write("Goal: Minimize MSE to find Energy VAD settings most similar to Student\n\n")
        
        f.write(f"Settings Evaluated: {len(results)}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if best_result:
            f.write("=" * 70 + "\n")
            f.write("BEST SETTING\n")
            f.write("=" * 70 + "\n")
            s = best_result['setting']
            f.write(f"Threshold: {s['threshold']}\n")
            f.write(f"Hysteresis High: {s['hysteresis_high']}\n")
            f.write(f"Hysteresis Low: {s['hysteresis_low']}\n")
            f.write(f"Smoothing Window: {s['smoothing_window']}\n")
            f.write(f"Frame Hop: {s['frame_hop_ms']} ms\n")
            f.write(f"Min Speech Dur: {s['min_speech_dur']} s\n\n")
            
            f.write("Performance:\n")
            f.write(f"  Average MSE: {best_result['avg_mse']:.6f}\n")
            f.write(f"  Average Correlation: {best_result['avg_correlation']:.4f}\n")
            f.write(f"  Median MSE: {best_result['median_mse']:.6f}\n")
            f.write(f"  Median Correlation: {best_result['median_correlation']:.4f}\n")
            f.write(f"  Utterances Processed: {best_result['processed']}\n\n")
        
        # Top 5 settings
        valid_results = [r for r in results if r['processed'] > 0 and r['avg_mse'] < float('inf')]
        sorted_results = sorted(valid_results, key=lambda x: x['avg_mse'])[:5]
        
        f.write("=" * 70 + "\n")
        f.write("TOP 5 SETTINGS (by MSE)\n")
        f.write("=" * 70 + "\n\n")
        
        for i, r in enumerate(sorted_results, 1):
            s = r['setting']
            f.write(f"{i}. Threshold={s['threshold']}, H-high={s['hysteresis_high']}, "
                   f"H-low={s['hysteresis_low']}, Smooth={s['smoothing_window']}\n")
            f.write(f"   MSE: {r['avg_mse']:.6f}, Correlation: {r['avg_correlation']:.4f}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("LIMITATIONS\n")
        f.write("=" * 70 + "\n")
        f.write("- This tuning uses student model as proxy reference, not ground truth\n")
        f.write("- Optimal settings may differ when evaluated against true labels\n")
        f.write("- Frame rate mismatch handled by linear interpolation\n")
        f.write("- Results are relative comparison for milestone analysis\n")
    
    print(f"\nResults saved to:")
    print(f"  JSON: {results_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Text: {text_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune Energy VAD thresholds using student model as proxy reference"
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='manifests/torgo_pilot.csv',
        help='Path to manifest CSV'
    )
    parser.add_argument(
        '--student-preds',
        type=str,
        default='outputs/personal/comparison/student_frame_probs',
        help='Directory with student model predictions (.npy files)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/personal/energy_tuning',
        help='Output directory for tuning results'
    )
    parser.add_argument(
        '--max-utterances',
        type=int,
        default=20,
        help='Maximum utterances to evaluate (for faster tuning)'
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    student_preds_dir = Path(args.student_preds)
    
    print("=" * 70)
    print("ENERGY VAD THRESHOLD TUNING")
    print("=" * 70)
    print(f"\nUsing student predictions as proxy reference from: {student_preds_dir}")
    print("NOTE: This is proxy-based tuning, not ground-truth based.\n")
    
    # Load data
    utterances = load_manifest(args.manifest, args.max_utterances)
    student_predictions = load_student_predictions(student_preds_dir)
    
    if not student_predictions:
        print("ERROR: No student predictions found. Run inference first.")
        sys.exit(1)
    
    # Generate settings grid
    settings = generate_settings_grid()
    print(f"\nEvaluating {len(settings)} different settings...")
    print("-" * 70)
    
    # Evaluate each setting
    results = []
    for i, setting in enumerate(settings, 1):
        print(f"\n[{i}/{len(settings)}] Evaluating: threshold={setting['threshold']}, "
              f"h-high={setting['hysteresis_high']}, h-low={setting['hysteresis_low']}, "
              f"smooth={setting['smoothing_window']}")
        
        result = evaluate_setting(setting, utterances, student_predictions)
        results.append(result)
        
        print(f"  -> MSE: {result['avg_mse']:.6f}, Correlation: {result['avg_correlation']:.4f}, "
              f"Processed: {result['processed']}")
    
    # Find best setting
    best_result = find_best_setting(results)
    
    if best_result:
        print("\n" + "=" * 70)
        print("BEST SETTING FOUND")
        print("=" * 70)
        s = best_result['setting']
        print(f"Threshold: {s['threshold']}")
        print(f"Hysteresis High: {s['hysteresis_high']}")
        print(f"Hysteresis Low: {s['hysteresis_low']}")
        print(f"Smoothing Window: {s['smoothing_window']}")
        print(f"\nAverage MSE: {best_result['avg_mse']:.6f}")
        print(f"Average Correlation: {best_result['avg_correlation']:.4f}")
    else:
        print("\nWARNING: No valid results found.")
    
    # Save results
    save_results(results, best_result, output_dir)
    
    print("\n" + "=" * 70)
    print("Tuning complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
