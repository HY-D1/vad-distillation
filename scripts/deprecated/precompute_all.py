#!/usr/bin/env python3
"""
Pre-compute all caches for VAD distillation.

One script to cache features and teacher outputs.

Usage:
    python scripts/precompute_all.py \
        --manifest manifests/torgo_sentences.csv \
        --features mel mfcc \
        --teacher \
        --parallel 4
    
    # Quick test mode (process only 5 files)
    python scripts/precompute_all.py --test
    
    # Force rebuild (ignore existing caches)
    python scripts/precompute_all.py --force
"""

import argparse
import csv
import json
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

# Setup path for imports when running from scripts/ directory
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Default configuration
DEFAULTS = {
    'manifest': 'manifests/torgo_sentences.csv',
    'cache_dir': 'cached_features',
    'teacher_dir': 'teacher_probs',
    'n_mels': 40,
    'n_mfcc': 13,
    'sample_rate': 16000,
}


class ProgressTracker:
    """Track progress with ETA calculation."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.completed = 0
        self.start_time = time.time()
        self.errors = []
    
    def update(self, n: int = 1):
        """Update progress."""
        self.completed += n
    
    def add_error(self, item: str, error: str):
        """Record an error."""
        self.errors.append((item, error))
    
    def get_eta(self) -> str:
        """Calculate ETA."""
        if self.completed == 0:
            return "unknown"
        
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed
        remaining = self.total - self.completed
        eta = remaining / rate
        
        # Format ETA
        if eta < 60:
            return f"{eta:.0f}s"
        elif eta < 3600:
            return f"{eta/60:.1f}m"
        else:
            return f"{eta/3600:.1f}h"
    
    def __str__(self) -> str:
        """String representation."""
        pct = self.completed / self.total * 100 if self.total > 0 else 0
        return f"{self.desc}: {self.completed}/{self.total} ({pct:.1f}%) - ETA: {self.get_eta()}"


class CacheManager:
    """Manage cache operations."""
    
    def __init__(self, cache_dir: Path, manifest_path: Path):
        self.cache_dir = cache_dir
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()
        self._interrupted = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print("\n\nInterrupt received. Saving progress...")
        self._interrupted = True
    
    def _load_manifest(self) -> List[Dict]:
        """Load manifest CSV."""
        with open(self.manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def is_interrupted(self) -> bool:
        """Check if interrupted."""
        return self._interrupted
    
    def get_cache_path(self, cache_type: str, row: Dict) -> Path:
        """Get cache file path for an utterance."""
        speaker_id = row['speaker_id']
        utt_id = row['utt_id']
        session = row.get('session', 'unknown')
        
        filename = f"{speaker_id}_{session}_{int(utt_id):04d}.npy"
        
        if cache_type == 'mel':
            return self.cache_dir / 'mel' / filename
        elif cache_type == 'mfcc':
            return self.cache_dir / 'mfcc' / filename
        elif cache_type == 'raw':
            return self.cache_dir / 'raw' / filename
        elif cache_type == 'teacher':
            return Path(DEFAULTS['teacher_dir']) / filename
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def get_cached_files(self, cache_type: str) -> Set[str]:
        """Get set of already cached utterance IDs."""
        if cache_type in ['mel', 'mfcc', 'raw']:
            cache_subdir = self.cache_dir / cache_type
        elif cache_type == 'teacher':
            cache_subdir = Path(DEFAULTS['teacher_dir'])
        else:
            return set()
        
        if not cache_subdir.exists():
            return set()
        
        return {f.stem for f in cache_subdir.glob('*.npy')}
    
    def filter_already_cached(self, rows: List[Dict], cache_type: str) -> List[Dict]:
        """Filter out rows that are already cached."""
        cached = self.get_cached_files(cache_type)
        filtered = []
        
        for row in rows:
            speaker_id = row['speaker_id']
            utt_id = row['utt_id']
            session = row.get('session', 'unknown')
            cache_key = f"{speaker_id}_{session}_{int(utt_id):04d}"
            
            if cache_key not in cached:
                filtered.append(row)
        
        return filtered


def check_disk_space(required_mb: float) -> bool:
    """Check if there's enough disk space."""
    import shutil
    stat = shutil.disk_usage('.')
    available_mb = stat.free / (1024 * 1024)
    
    if available_mb < required_mb:
        print(f"ERROR: Not enough disk space!")
        print(f"  Required: {required_mb:.0f} MB")
        print(f"  Available: {available_mb:.0f} MB")
        return False
    
    return True


def estimate_cache_size(num_files: int, cache_type: str) -> float:
    """Estimate cache size in MB."""
    # Rough estimates based on average file sizes
    estimates = {
        'mel': 0.02,      # ~20 KB per file
        'mfcc': 0.007,    # ~7 KB per file
        'raw': 0.07,      # ~70 KB per file (16kHz, ~4s avg)
        'teacher': 0.005, # ~5 KB per file
    }
    return num_files * estimates.get(cache_type, 0.01)


def run_feature_caching(args, cache_mgr: CacheManager, rows: List[Dict]) -> Dict:
    """Run feature caching."""
    results = {
        'mel': {'success': 0, 'skipped': 0, 'failed': 0},
        'mfcc': {'success': 0, 'skipped': 0, 'failed': 0},
        'raw': {'success': 0, 'skipped': 0, 'failed': 0},
    }
    
    # Import here to avoid loading if not needed
    from scripts.cache_features import process_manifest_parallel, save_metadata
    
    for feature_type in args.features:
        if feature_type not in ['mel', 'mfcc', 'raw']:
            print(f"Unknown feature type: {feature_type}")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Caching {feature_type.upper()} features")
        print('=' * 60)
        
        # Filter already cached
        if not args.force:
            to_process = cache_mgr.filter_already_cached(rows, feature_type)
            skipped = len(rows) - len(to_process)
            results[feature_type]['skipped'] = skipped
            print(f"Already cached: {skipped}, To process: {len(to_process)}")
        else:
            to_process = rows
            print(f"Force mode: Processing all {len(to_process)} files")
        
        if not to_process:
            print(f"All {feature_type} features already cached.")
            continue
        
        # Check disk space
        estimated_size = estimate_cache_size(len(to_process), feature_type)
        print(f"Estimated space needed: {estimated_size:.1f} MB")
        
        if not check_disk_space(estimated_size * 1.5):  # 1.5x safety margin
            print(f"Skipping {feature_type} caching due to insufficient disk space")
            continue
        
        # Process
        tracker = ProgressTracker(len(to_process), f"Caching {feature_type}")
        
        try:
            success, failed = process_manifest_parallel(
                manifest_rows=to_process,
                output_dir=Path(args.cache_dir) / feature_type,
                feature_type=feature_type,
                parallel=args.parallel,
                resume=not args.force,
                verify=args.verify,
                tracker=tracker,
                is_interrupted_func=cache_mgr.is_interrupted,
            )
            
            results[feature_type]['success'] = success
            results[feature_type]['failed'] = failed
            
        except Exception as e:
            print(f"Error during {feature_type} caching: {e}")
            traceback.print_exc()
            results[feature_type]['failed'] = len(to_process)
        
        if cache_mgr.is_interrupted():
            break
    
    return results


def run_teacher_caching(args, cache_mgr: CacheManager, rows: List[Dict]) -> Dict:
    """Run teacher caching."""
    results = {'success': 0, 'skipped': 0, 'failed': 0}
    
    print(f"\n{'=' * 60}")
    print("Caching Teacher Outputs")
    print('=' * 60)
    
    # Import here to avoid loading if not needed
    from scripts.cache_teacher import process_teacher_manifest
    
    # Filter already cached
    if not args.force:
        to_process = cache_mgr.filter_already_cached(rows, 'teacher')
        skipped = len(rows) - len(to_process)
        results['skipped'] = skipped
        print(f"Already cached: {skipped}, To process: {len(to_process)}")
    else:
        to_process = rows
        print(f"Force mode: Processing all {len(to_process)} files")
    
    if not to_process:
        print("All teacher outputs already cached.")
        return results
    
    # Check disk space
    estimated_size = estimate_cache_size(len(to_process), 'teacher')
    print(f"Estimated space needed: {estimated_size:.1f} MB")
    
    if not check_disk_space(estimated_size * 1.5):
        print("Skipping teacher caching due to insufficient disk space")
        return results
    
    # Process
    tracker = ProgressTracker(len(to_process), "Caching teacher")
    
    try:
        success, failed = process_teacher_manifest(
            manifest_rows=to_process,
            output_dir=Path(DEFAULTS['teacher_dir']),
            device=args.device,
            batch_size=args.batch_size,
            resume=not args.force,
            verify=args.verify,
            tracker=tracker,
            is_interrupted_func=cache_mgr.is_interrupted,
        )
        
        results['success'] = success
        results['failed'] = failed
        
    except Exception as e:
        print(f"Error during teacher caching: {e}")
        traceback.print_exc()
        results['failed'] = len(to_process)
    
    return results


def generate_report(feature_results: Dict, teacher_results: Dict, start_time: float) -> Dict:
    """Generate summary report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': time.time() - start_time,
        'features': feature_results,
        'teacher': teacher_results,
    }
    
    # Calculate totals
    total_success = sum(r['success'] for r in feature_results.values())
    total_skipped = sum(r['skipped'] for r in feature_results.values())
    total_failed = sum(r['failed'] for r in feature_results.values())
    
    report['total'] = {
        'success': total_success + teacher_results.get('success', 0),
        'skipped': total_skipped + teacher_results.get('skipped', 0),
        'failed': total_failed + teacher_results.get('failed', 0),
    }
    
    return report


def print_report(report: Dict):
    """Print summary report."""
    print("\n" + "=" * 60)
    print("Pre-computation Report")
    print("=" * 60)
    
    # Feature caching results
    print("\nFeature Caching:")
    for feature_type, results in report['features'].items():
        if results['success'] > 0 or results['failed'] > 0:
            print(f"  {feature_type.upper():<10} "
                  f"Success: {results['success']:>5}, "
                  f"Skipped: {results['skipped']:>5}, "
                  f"Failed: {results['failed']:>5}")
    
    # Teacher caching results
    if report['teacher']:
        t = report['teacher']
        print(f"\nTeacher Caching:")
        print(f"  Success: {t['success']}, Skipped: {t['skipped']}, Failed: {t['failed']}")
    
    # Totals
    total = report['total']
    print(f"\n{'=' * 60}")
    print(f"Total: Success={total['success']}, Skipped={total['skipped']}, Failed={total['failed']}")
    
    # Duration
    duration = report['duration_seconds']
    if duration < 60:
        duration_str = f"{duration:.1f}s"
    elif duration < 3600:
        duration_str = f"{duration/60:.1f}m"
    else:
        duration_str = f"{duration/3600:.2f}h"
    
    print(f"Duration: {duration_str}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute all caches for VAD distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cache only mel spectrograms
  python scripts/precompute_all.py --features mel
  
  # Cache mel and teacher outputs in parallel
  python scripts/precompute_all.py --features mel --teacher --parallel 4
  
  # Test mode (5 files only)
  python scripts/precompute_all.py --test
  
  # Force rebuild
  python scripts/precompute_all.py --force
        """
    )
    
    # Input/output
    parser.add_argument('--manifest', type=str, default=DEFAULTS['manifest'],
                        help='Path to manifest CSV')
    parser.add_argument('--cache-dir', type=str, default=DEFAULTS['cache_dir'],
                        help='Output directory for features')
    parser.add_argument('--teacher-dir', type=str, default=DEFAULTS['teacher_dir'],
                        help='Output directory for teacher probabilities')
    
    # What to cache
    parser.add_argument('--features', nargs='+', choices=['mel', 'mfcc', 'raw', 'all'],
                        default=['mel'],
                        help='Which features to cache (default: mel)')
    parser.add_argument('--teacher', action='store_true',
                        help='Cache teacher outputs')
    
    # Processing options
    parser.add_argument('--parallel', type=int, default=4,
                        help='Number of parallel workers for feature caching')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for teacher inference (cpu/cuda, default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for teacher inference')
    
    # Control options
    parser.add_argument('--force', action='store_true',
                        help='Force rebuild (ignore existing caches)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify cached files')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: process only 5 files')
    
    # Output
    parser.add_argument('--report', type=str,
                        help='Save report to JSON file')
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    import torch
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Validate manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)
    
    # Expand 'all' features
    if 'all' in args.features:
        args.features = ['mel', 'mfcc', 'raw']
    
    print("=" * 60)
    print("Pre-compute All Caches")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Features: {', '.join(args.features)}")
    print(f"Teacher:  {'Yes' if args.teacher else 'No'}")
    print(f"Test mode: {'Yes' if args.test else 'No'}")
    print("=" * 60)
    
    # Initialize cache manager
    cache_mgr = CacheManager(Path(args.cache_dir), manifest_path)
    
    # Get manifest rows
    rows = cache_mgr.manifest
    
    # Test mode: limit to 5 files
    if args.test:
        rows = rows[:5]
        print(f"\nTEST MODE: Processing only {len(rows)} files")
    
    print(f"\nTotal utterances to process: {len(rows)}")
    
    # Track start time
    start_time = time.time()
    
    # Run feature caching
    feature_results = {}
    if args.features:
        feature_results = run_feature_caching(args, cache_mgr, rows)
    
    # Run teacher caching
    teacher_results = {}
    if args.teacher and not cache_mgr.is_interrupted():
        teacher_results = run_teacher_caching(args, cache_mgr, rows)
    
    # Generate report
    report = generate_report(feature_results, teacher_results, start_time)
    
    # Print report
    print_report(report)
    
    # Save report if requested
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")
    
    # Exit code based on results
    total_failed = report['total']['failed']
    if total_failed > 0:
        print(f"\nWarning: {total_failed} files failed to process")
        sys.exit(1)
    
    if cache_mgr.is_interrupted():
        print("\nInterrupted. Run again with --resume to continue.")
        sys.exit(130)
    
    print("\n✓ All caches pre-computed successfully!")


if __name__ == "__main__":
    main()
