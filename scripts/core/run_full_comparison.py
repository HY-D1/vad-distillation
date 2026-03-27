#!/usr/bin/env python3
"""
Full comparison pipeline for VAD methods.

This script runs the complete comparison workflow:
1. Extract student predictions from outputs/production_cuda/ to outputs/our_model/
2. Run Energy baseline (if not already done)
3. Run Silero baseline (if not already done)  
4. Run SpeechBrain baseline (optional)
5. Run comparison analysis

Usage:
    # Full pipeline (all steps)
    python scripts/core/run_full_comparison.py
    
    # Skip baselines (if already computed)
    python scripts/core/run_full_comparison.py --skip-baselines
    
    # Skip extraction (if student predictions already extracted)
    python scripts/core/run_full_comparison.py --skip-extraction
    
    # Compare only specific methods
    python scripts/core/run_full_comparison.py --methods energy,silero
    
    # Use specific manifest
    python scripts/core/run_full_comparison.py --manifest manifests/torgo_pilot.csv
"""

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# Add project root to path for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.common import get_device

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    'student_results_dir': 'outputs/production_cuda',
    'student_output_dir': 'outputs/our_model',
    'baselines_dir': 'outputs/baselines',
    'analysis_dir': 'analysis/comparison',
    'evaluation_csv': 'outputs/evaluation/comparison_table.csv',
    'timing_file': 'analysis/timing.json',
    'model_size_file': 'analysis/model_sizes.json',
    'manifest': 'manifests/torgo_sentences.csv',
    'teacher_probs_dir': 'teacher_probs',
    'hard_labels_dir': 'teacher_hard_labels/thresh_0.5',
    'device': None,  # Auto-detect
}

AVAILABLE_METHODS = ['energy', 'silero', 'speechbrain', 'our_model']


# =============================================================================
# Utility Functions
# =============================================================================

def check_baseline_exists(baseline_dir: Path, method: str) -> bool:
    """Check if a baseline has already been run."""
    method_dir = baseline_dir / method
    if not method_dir.exists():
        return False
    
    # Check for required subdirectories
    frame_probs_dir = method_dir / 'frame_probs'
    meta_file = method_dir / 'meta.json'
    
    return frame_probs_dir.exists() and meta_file.exists()


def run_command(cmd: List[str], cwd: Optional[str] = None, 
                description: str = "") -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    logger.info(f"Running: {' '.join(cmd)}")
    if description:
        logger.info(f"Description: {description}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False, e.stderr


# =============================================================================
# Step 1: Extract Student Predictions
# =============================================================================

def extract_student_predictions(
    student_results_dir: Path,
    output_dir: Path,
    manifest_path: Path
) -> bool:
    """
    Extract student predictions from training outputs.
    
    This function:
    1. Loads prediction .npz files from each fold
    2. Organizes them into frame_probs directory with proper naming
    3. Creates metadata file
    
    Args:
        student_results_dir: Directory with fold_*_predictions.npz files
        output_dir: Where to save extracted predictions
        manifest_path: Path to manifest CSV
        
    Returns:
        True if successful
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting Student Predictions")
    logger.info("=" * 60)
    
    # Create output directories
    frame_probs_dir = output_dir / 'frame_probs'
    segments_dir = output_dir / 'segments'
    frame_probs_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest to get utterance IDs
    manifest_rows = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)
    
    logger.info(f"Loaded {len(manifest_rows)} utterances from manifest")
    
    # Find all prediction files
    logs_dir = student_results_dir / 'logs'
    if not logs_dir.exists():
        logger.error(f"Logs directory not found: {logs_dir}")
        return False
    
    prediction_files = sorted(logs_dir.glob('*_predictions.npz'))
    logger.info(f"Found {len(prediction_files)} prediction files")
    
    if not prediction_files:
        logger.error("No prediction files found!")
        return False
    
    # Track extraction statistics
    extracted_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each prediction file
    for pred_file in prediction_files:
        # Extract fold name from filename (e.g., fold_F01_predictions.npz)
        fold_name = pred_file.stem.replace('_predictions', '')
        logger.info(f"Processing {fold_name}...")
        
        try:
            # Load predictions
            data = np.load(pred_file, allow_pickle=True)
            
            # The .npz file should contain predictions for test utterances
            # We need to map these back to utterance IDs
            # The predictions file contains arrays: probs, predictions, labels
            
            if 'probs' in data:
                probs = data['probs']
                utt_ids = data['utt_ids'] if 'utt_ids' in data else None

                if utt_ids is not None and len(utt_ids) == len(probs):
                    # Reconstruct per-utterance frame probability files expected by compare_methods.py.
                    by_utt = {}
                    for utt_id, prob in zip(utt_ids, probs):
                        utt_key = utt_id.decode('utf-8') if isinstance(utt_id, bytes) else str(utt_id)
                        by_utt.setdefault(utt_key, []).append(float(prob))

                    for utt_key, utt_probs in by_utt.items():
                        output_file = frame_probs_dir / f"{utt_key}.npy"
                        np.save(output_file, np.asarray(utt_probs, dtype=np.float32))
                        extracted_count += 1

                    logger.info(f"  Saved {len(by_utt)} utterance files for {fold_name}")
                else:
                    # Fallback for legacy prediction format.
                    output_file = frame_probs_dir / f"{fold_name}.npz"
                    np.savez(output_file, **{k: data[k] for k in data.keys()})
                    extracted_count += 1
                    logger.info(f"  Saved fallback predictions to {output_file}")
            else:
                logger.warning(f"  No 'probs' key in {pred_file}")
                skipped_count += 1
                
        except Exception as e:
            logger.error(f"  Error processing {pred_file}: {e}")
            error_count += 1
    
    # Create metadata
    meta = {
        'source': str(student_results_dir),
        'num_folds': len(prediction_files),
        'extracted_count': extracted_count,
        'skipped_count': skipped_count,
        'error_count': error_count,
        'manifest': str(manifest_path),
    }
    
    meta_file = output_dir / 'meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Create config file
    config = {
        'method': 'our_model',
        'description': 'Student model predictions from production training',
        'source_dir': str(student_results_dir),
        'num_folds': len(prediction_files),
    }
    
    config_file = output_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"\nExtraction complete:")
    logger.info(f"  Extracted: {extracted_count}")
    logger.info(f"  Skipped: {skipped_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Output: {output_dir}")
    
    return error_count == 0


# =============================================================================
# Step 2-4: Run Baselines
# =============================================================================

def run_baseline(
    method: str,
    manifest_path: Path,
    output_dir: Path,
    device: str = 'cpu'
) -> bool:
    """
    Run a single baseline method.
    
    Args:
        method: Baseline method name (energy, silero, speechbrain)
        manifest_path: Path to manifest CSV
        output_dir: Output directory for baseline results
        device: Device to use (cpu/cuda)
        
    Returns:
        True if successful
    """
    logger.info(f"\nRunning {method} baseline...")
    
    # Use the existing run_baseline.py script
    cmd = [
        sys.executable,
        'scripts/core/run_baseline.py',
        '--method', method,
        '--manifest', str(manifest_path),
        '--output-dir', str(output_dir),
        '--device', device
    ]
    
    success, output = run_command(cmd, description=f"Run {method} baseline")
    
    if success:
        logger.info(f"  ✓ {method} baseline completed successfully")
    else:
        logger.error(f"  ✗ {method} baseline failed")
        
    return success


def run_all_baselines(
    methods: List[str],
    manifest_path: Path,
    baselines_dir: Path,
    device: str,
    skip_existing: bool = True
) -> Dict[str, bool]:
    """
    Run all specified baseline methods.
    
    Args:
        methods: List of method names to run
        manifest_path: Path to manifest CSV
        baselines_dir: Base directory for baseline outputs
        device: Device to use
        skip_existing: Skip if output already exists
        
    Returns:
        Dictionary mapping method name to success status
    """
    logger.info("=" * 60)
    logger.info("STEPS 2-4: Running Baselines")
    logger.info("=" * 60)
    logger.info(f"Methods: {', '.join(methods)}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Output directory: {baselines_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Skip existing: {skip_existing}")
    
    results = {}
    
    for method in methods:
        if method == 'our_model':
            continue  # Skip our_model, it's handled separately
            
        method_output_dir = baselines_dir / method
        
        # Check if already exists
        if skip_existing and check_baseline_exists(baselines_dir, method):
            logger.info(f"\n{method}: Already exists, skipping (use --no-skip-existing to override)")
            results[method] = True
            continue
        
        # Run the baseline
        success = run_baseline(
            method=method,
            manifest_path=manifest_path,
            output_dir=method_output_dir,
            device=device
        )
        results[method] = success
    
    return results


# =============================================================================
# Step 5: Run Comparison
# =============================================================================

def run_comparison(
    manifest_path: Path,
    methods: List[str],
    method_dirs: List[Path],
    method_names: List[str],
    output_dir: Path,
    teacher_probs_dir: Path,
    hard_labels_dir: Path,
    label_source: str,
    timing_file: Optional[Path],
    model_size_file: Optional[Path],
    threshold: float = 0.5
) -> bool:
    """
    Run comparison analysis using compare_methods.py.
    
    Args:
        manifest_path: Path to manifest CSV
        methods: List of method identifiers
        method_dirs: List of method output directories
        method_names: List of display names for methods
        output_dir: Where to save comparison results
        teacher_probs_dir: Directory with teacher probabilities
        threshold: Classification threshold
        
    Returns:
        True if successful
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Running Comparison Analysis")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command for compare_methods.py
    methods_str = ','.join(str(d) for d in method_dirs)
    method_names_str = ','.join(method_names)
    
    cmd = [
        sys.executable,
        'scripts/analysis/compare_methods.py',
        '--manifest', str(manifest_path),
        '--methods', methods_str,
        '--method-names', method_names_str,
        '--output-dir', str(output_dir),
        '--proxy-labels', label_source,
        '--teacher-dir', str(teacher_probs_dir),
        '--hard-label-dir', str(hard_labels_dir),
        '--threshold', str(threshold)
    ]
    if timing_file is not None and timing_file.exists():
        cmd.extend(['--timing-file', str(timing_file)])
    if model_size_file is not None and model_size_file.exists():
        cmd.extend(['--model-size-file', str(model_size_file)])
    
    success, output = run_command(cmd, description="Run comparison analysis")
    
    if success:
        logger.info("  ✓ Comparison analysis completed successfully")
        # Print summary
        comparison_csv = output_dir / 'comparison_table.csv'
        comparison_md = output_dir / 'comparison_table.md'
        
        if comparison_csv.exists():
            logger.info(f"\n  Results saved:")
            logger.info(f"    CSV: {comparison_csv}")
            logger.info(f"    Markdown: {comparison_md}")
            logger.info(f"    Plots: {output_dir / 'plots'}")
            
            # Print the markdown table
            if comparison_md.exists():
                with open(comparison_md, 'r') as f:
                    content = f.read()
                    # Extract and print the summary table
                    if '## Summary' in content:
                        table_start = content.find('## Summary')
                        table_end = content.find('## Detailed Results')
                        if table_end == -1:
                            table_end = len(content)
                        table_section = content[table_start:table_end].strip()
                        logger.info(f"\n{table_section}")
    else:
        logger.error("  ✗ Comparison analysis failed")
        
    return success


# =============================================================================
# Main Pipeline
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Full comparison pipeline for VAD methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/core/run_full_comparison.py
  
  # Skip baselines (use existing)
  python scripts/core/run_full_comparison.py --skip-baselines
  
  # Skip extraction (student predictions already extracted)
  python scripts/core/run_full_comparison.py --skip-extraction
  
  # Compare only specific methods
  python scripts/core/run_full_comparison.py --methods energy,silero,our_model
  
  # Use specific manifest
  python scripts/core/run_full_comparison.py --manifest manifests/torgo_pilot.csv
  
  # Force re-run baselines even if they exist
  python scripts/core/run_full_comparison.py --no-skip-existing
        """
    )
    
    # Data options
    parser.add_argument(
        '--manifest',
        type=str,
        default=DEFAULT_CONFIG['manifest'],
        help=f'Path to manifest CSV (default: {DEFAULT_CONFIG["manifest"]})'
    )
    parser.add_argument(
        '--student-results-dir',
        type=str,
        default=DEFAULT_CONFIG['student_results_dir'],
        help=f'Directory with student training outputs (default: {DEFAULT_CONFIG["student_results_dir"]})'
    )
    
    # Output options
    parser.add_argument(
        '--student-output-dir',
        type=str,
        default=DEFAULT_CONFIG['student_output_dir'],
        help=f'Where to save extracted student predictions (default: {DEFAULT_CONFIG["student_output_dir"]})'
    )
    parser.add_argument(
        '--baselines-dir',
        type=str,
        default=DEFAULT_CONFIG['baselines_dir'],
        help=f'Where to save baseline outputs (default: {DEFAULT_CONFIG["baselines_dir"]})'
    )
    parser.add_argument(
        '--analysis-dir',
        type=str,
        default=DEFAULT_CONFIG['analysis_dir'],
        help=f'Where to save comparison results (default: {DEFAULT_CONFIG["analysis_dir"]})'
    )
    parser.add_argument(
        '--evaluation-csv',
        type=str,
        default=DEFAULT_CONFIG['evaluation_csv'],
        help=f'Where to write final comparison CSV copy (default: {DEFAULT_CONFIG["evaluation_csv"]})'
    )
    parser.add_argument(
        '--timing-file',
        type=str,
        default=DEFAULT_CONFIG['timing_file'],
        help=f'Optional timing metadata JSON (default: {DEFAULT_CONFIG["timing_file"]})'
    )
    parser.add_argument(
        '--model-size-file',
        type=str,
        default=DEFAULT_CONFIG['model_size_file'],
        help=f'Optional model-size metadata JSON (default: {DEFAULT_CONFIG["model_size_file"]})'
    )
    
    # Method selection
    parser.add_argument(
        '--methods',
        type=str,
        default=None,
        help='Comma-separated list of methods to compare (default: all available). '
             'Options: energy, silero, speechbrain, our_model'
    )
    
    # Control flags
    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip running baselines (use existing results)'
    )
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip extracting student predictions (use existing)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Do not skip baselines even if they already exist'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device for baseline inference (default: auto-detect)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    parser.add_argument(
        '--label-source',
        type=str,
        default='auto',
        choices=['auto', 'hard', 'teacher', 'all_speech', 'none'],
        help='Label source for evaluation (default: auto -> hard if available, else teacher)'
    )
    parser.add_argument(
        '--hard-label-dir',
        type=str,
        default=DEFAULT_CONFIG['hard_labels_dir'],
        help=f'Frame-level hard-label directory (default: {DEFAULT_CONFIG["hard_labels_dir"]})'
    )
    
    # Debug options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the comparison pipeline."""
    args = parse_args()
    
    # Resolve paths
    manifest_path = Path(args.manifest).resolve()
    student_results_dir = Path(args.student_results_dir).resolve()
    student_output_dir = Path(args.student_output_dir).resolve()
    baselines_dir = Path(args.baselines_dir).resolve()
    analysis_dir = Path(args.analysis_dir).resolve()
    evaluation_csv = Path(args.evaluation_csv).resolve()
    timing_file = Path(args.timing_file).resolve() if args.timing_file else None
    model_size_file = Path(args.model_size_file).resolve() if args.model_size_file else None
    teacher_probs_dir = Path(DEFAULT_CONFIG['teacher_probs_dir']).resolve()
    hard_labels_dir = Path(args.hard_label_dir).resolve()
    
    # Validate inputs
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    
    if not student_results_dir.exists():
        logger.error(f"Student results directory not found: {student_results_dir}")
        sys.exit(1)
    
    # Determine device
    device = args.device or get_device()
    logger.info(f"Using device: {device}")
    
    # Determine which methods to compare
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',')]
    else:
        # Default: all methods that exist or can be run
        methods = AVAILABLE_METHODS.copy()
    
    logger.info("=" * 60)
    logger.info("VAD Full Comparison Pipeline")
    logger.info("=" * 60)
    logger.info(f"Methods to compare: {', '.join(methods)}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Device: {device}")

    if args.label_source == 'auto':
        label_source = 'hard' if hard_labels_dir.exists() else 'teacher'
    else:
        label_source = args.label_source
    logger.info(f"Evaluation labels: {label_source}")
    if label_source == 'hard':
        logger.info(f"Hard labels dir: {hard_labels_dir}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN MODE - No actual execution]")
    
    # Track success of each step
    step_results = {}
    
    # ==========================================================================
    # Step 1: Extract Student Predictions
    # ==========================================================================
    if 'our_model' in methods and not args.skip_extraction:
        if args.dry_run:
            logger.info(f"\n[DRY RUN] Would extract student predictions from {student_results_dir}")
            step_results['extraction'] = True
        else:
            success = extract_student_predictions(
                student_results_dir=student_results_dir,
                output_dir=student_output_dir,
                manifest_path=manifest_path
            )
            step_results['extraction'] = success
    else:
        if args.skip_extraction:
            logger.info("\nSkipping extraction (--skip-extraction specified)")
        if 'our_model' not in methods:
            logger.info("\nSkipping extraction (our_model not in --methods)")
        step_results['extraction'] = True
    
    # ==========================================================================
    # Steps 2-4: Run Baselines
    # ==========================================================================
    baseline_methods = [m for m in methods if m != 'our_model']
    
    if baseline_methods and not args.skip_baselines:
        if args.dry_run:
            logger.info(f"\n[DRY RUN] Would run baselines: {', '.join(baseline_methods)}")
            step_results['baselines'] = {m: True for m in baseline_methods}
        else:
            baseline_results = run_all_baselines(
                methods=baseline_methods,
                manifest_path=manifest_path,
                baselines_dir=baselines_dir,
                device=device,
                skip_existing=not args.no_skip_existing
            )
            step_results['baselines'] = baseline_results
    else:
        if args.skip_baselines:
            logger.info("\nSkipping baselines (--skip-baselines specified)")
        if not baseline_methods:
            logger.info("\nNo baselines to run (only comparing our_model)")
        step_results['baselines'] = {}
    
    # ==========================================================================
    # Step 5: Run Comparison
    # ==========================================================================
    if args.dry_run:
        logger.info(f"\n[DRY RUN] Would run comparison analysis")
        step_results['comparison'] = True
    else:
        # Build list of method directories and names
        method_dirs = []
        method_names = []
        
        for method in methods:
            if method == 'our_model':
                if student_output_dir.exists():
                    method_dirs.append(student_output_dir)
                    method_names.append('Our Model')
                else:
                    logger.warning(f"Student output not found: {student_output_dir}")
            else:
                method_dir = baselines_dir / method
                if method_dir.exists():
                    method_dirs.append(method_dir)
                    # Capitalize method name for display
                    method_names.append(method.capitalize())
                else:
                    logger.warning(f"Baseline output not found: {method_dir}")
        
        if len(method_dirs) < 2:
            logger.error("Need at least 2 methods to compare!")
            step_results['comparison'] = False
        else:
            success = run_comparison(
                manifest_path=manifest_path,
                methods=methods,
                method_dirs=method_dirs,
                method_names=method_names,
                output_dir=analysis_dir,
                teacher_probs_dir=teacher_probs_dir,
                hard_labels_dir=hard_labels_dir,
                label_source=label_source,
                timing_file=timing_file,
                model_size_file=model_size_file,
                threshold=args.threshold
            )
            step_results['comparison'] = success
            if success:
                generated_csv = analysis_dir / 'comparison_table.csv'
                generated_md = analysis_dir / 'comparison_table.md'
                if generated_csv.exists():
                    evaluation_csv.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(generated_csv, evaluation_csv)
                    logger.info(f"Copied final CSV to: {evaluation_csv}")
                if generated_md.exists():
                    eval_md = evaluation_csv.parent / 'comparison_table.md'
                    shutil.copy2(generated_md, eval_md)
                    logger.info(f"Copied final Markdown to: {eval_md}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    
    if 'extraction' in step_results:
        status = "✓" if step_results['extraction'] else "✗"
        logger.info(f"{status} Step 1: Student Prediction Extraction")
    
    if 'baselines' in step_results and step_results['baselines']:
        for method, success in step_results['baselines'].items():
            status = "✓" if success else "✗"
            logger.info(f"{status} Step 2-4: {method.capitalize()} Baseline")
    
    if 'comparison' in step_results:
        status = "✓" if step_results['comparison'] else "✗"
        logger.info(f"{status} Step 5: Comparison Analysis")
    
    # Final status
    all_success = all([
        step_results.get('extraction', True),
        all(step_results.get('baselines', {}).values()) if step_results.get('baselines') else True,
        step_results.get('comparison', True)
    ])
    
    if all_success:
        logger.info("\n✓ All steps completed successfully!")
        logger.info(f"\nResults available in:")
        logger.info(f"  Student predictions: {student_output_dir}")
        logger.info(f"  Baseline outputs: {baselines_dir}")
        logger.info(f"  Comparison results: {analysis_dir}")
    else:
        logger.error("\n✗ Some steps failed. Check the logs above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
