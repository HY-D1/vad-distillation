#!/usr/bin/env python3
"""
Cache Manager for VAD Distillation Project.

Unified utility for managing all caches (features, teacher outputs).

Usage:
    # Show cache status
    python scripts/cache_manager.py status
    
    # Show detailed stats
    python scripts/cache_manager.py stats
    
    # Clean old/invalid cache
    python scripts/cache_manager.py clean --older-than 30days
    
    # Verify all cached files
    python scripts/cache_manager.py verify
    
    # Clear specific cache
    python scripts/cache_manager.py clear --type mel
    
    # Rebuild all caches
    python scripts/cache_manager.py rebuild --manifest manifests/torgo_sentences.csv
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Cache configuration
CACHE_CONFIG = {
    'mel': {
        'dir': 'cached_features/mel',
        'ext': '.npy',
        'desc': 'Mel spectrograms',
    },
    'mfcc': {
        'dir': 'cached_features/mfcc',
        'ext': '.npy',
        'desc': 'MFCC features',
    },
    'raw': {
        'dir': 'cached_features/raw',
        'ext': '.npy',
        'desc': 'Raw audio (16kHz)',
    },
    'teacher': {
        'dir': 'teacher_probs',
        'ext': '.npy',
        'desc': 'Teacher probabilities',
    },
}

METADATA_FILES = {
    'mel': 'cached_features/meta.json',
    'mfcc': 'cached_features/meta.json',
    'raw': 'cached_features/meta.json',
    'teacher': 'teacher_probs/meta.json',
}


def get_cache_dir(cache_type: str) -> Path:
    """Get cache directory for a specific type."""
    return Path(CACHE_CONFIG[cache_type]['dir'])


def get_cache_files(cache_type: str) -> List[Path]:
    """Get all cache files for a specific type."""
    cache_dir = get_cache_dir(cache_type)
    if not cache_dir.exists():
        return []
    ext = CACHE_CONFIG[cache_type]['ext']
    return sorted(cache_dir.glob(f'*{ext}'))


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def load_metadata(cache_type: str) -> Optional[Dict]:
    """Load metadata for a cache type."""
    meta_path = Path(METADATA_FILES[cache_type])
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def format_size(size_mb: float) -> str:
    """Format size in human-readable format."""
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb / 1024:.2f} GB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def count_manifest_entries(manifest_path: str) -> int:
    """Count entries in manifest CSV."""
    import csv
    try:
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)
    except FileNotFoundError:
        return 0


def cmd_status(args):
    """Show cache status overview."""
    print("\n" + "=" * 60)
    print("Cache Status")
    print("=" * 60)
    
    total_size_mb = 0
    total_files = 0
    expected_files = None
    
    if args.manifest:
        expected_files = count_manifest_entries(args.manifest)
    
    for cache_type, config in CACHE_CONFIG.items():
        files = get_cache_files(cache_type)
        num_files = len(files)
        total_files += num_files
        
        size_mb = sum(get_file_size_mb(f) for f in files)
        total_size_mb += size_mb
        
        # Calculate percentage
        if expected_files and cache_type != 'teacher':
            pct = (num_files / expected_files * 100) if expected_files > 0 else 0
            status = f"{num_files}/{expected_files} files ({pct:.0f}%)"
        elif cache_type == 'teacher' and expected_files:
            pct = (num_files / expected_files * 100) if expected_files > 0 else 0
            status = f"{num_files}/{expected_files} files ({pct:.0f}%)"
        else:
            status = f"{num_files} files"
        
        print(f"{config['desc']:<25} {status:<20} - {format_size(size_mb)}")
    
    print("-" * 60)
    print(f"{'Total cache size':<25} {total_files} files{'':<9} - {format_size(total_size_mb)}")
    print("=" * 60)
    
    # Show metadata info
    print("\nCache Metadata:")
    for cache_type in CACHE_CONFIG.keys():
        meta = load_metadata(cache_type)
        if meta:
            version = meta.get('version', 'unknown')
            timestamp = meta.get('timestamp', 'unknown')
            print(f"  {cache_type}: v{version}, created {timestamp}")
        else:
            print(f"  {cache_type}: no metadata")


def cmd_stats(args):
    """Show detailed cache statistics."""
    print("\n" + "=" * 60)
    print("Detailed Cache Statistics")
    print("=" * 60)
    
    for cache_type, config in CACHE_CONFIG.items():
        files = get_cache_files(cache_type)
        
        if not files:
            print(f"\n{config['desc']}: No files cached")
            continue
        
        print(f"\n{config['desc']}:")
        print("-" * 40)
        
        # File count and size
        num_files = len(files)
        sizes_mb = [get_file_size_mb(f) for f in files]
        total_size_mb = sum(sizes_mb)
        
        print(f"  Total files:   {num_files}")
        print(f"  Total size:    {format_size(total_size_mb)}")
        print(f"  Average size:  {format_size(total_size_mb / num_files)}")
        print(f"  Min size:      {format_size(min(sizes_mb))}")
        print(f"  Max size:      {format_size(max(sizes_mb))}")
        
        # File age distribution
        now = datetime.now()
        age_buckets = {'<1d': 0, '1-7d': 0, '1-4w': 0, '>1m': 0}
        for f in files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            age = now - mtime
            if age < timedelta(days=1):
                age_buckets['<1d'] += 1
            elif age < timedelta(days=7):
                age_buckets['1-7d'] += 1
            elif age < timedelta(days=30):
                age_buckets['1-4w'] += 1
            else:
                age_buckets['>1m'] += 1
        
        print(f"\n  File age distribution:")
        for bucket, count in age_buckets.items():
            pct = count / num_files * 100
            bar = '█' * int(pct / 5)
            print(f"    {bucket:>6}: {count:>5} ({pct:>5.1f}%) {bar}")
        
        # Metadata
        meta = load_metadata(cache_type)
        if meta:
            print(f"\n  Metadata:")
            for key, value in meta.items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"    {key}: {value}")


def verify_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Verify a cache file can be loaded."""
    try:
        data = np.load(file_path)
        if data.size == 0:
            return False, "empty array"
        if not np.isfinite(data).all():
            return False, "contains NaN or Inf"
        return True, None
    except Exception as e:
        return False, str(e)


def cmd_verify(args):
    """Verify all cached files are valid."""
    print("\n" + "=" * 60)
    print("Verifying Cache Files")
    print("=" * 60)
    
    all_valid = True
    
    for cache_type, config in CACHE_CONFIG.items():
        files = get_cache_files(cache_type)
        
        if not files:
            continue
        
        print(f"\nVerifying {config['desc']}...")
        
        invalid_files = []
        for i, f in enumerate(files):
            if args.verbose and (i + 1) % 100 == 0:
                print(f"  Checked {i + 1}/{len(files)} files...")
            
            valid, error = verify_file(f)
            if not valid:
                invalid_files.append((f, error))
        
        if invalid_files:
            all_valid = False
            print(f"  ✗ Found {len(invalid_files)} invalid files:")
            for f, error in invalid_files[:10]:  # Show first 10
                print(f"    - {f.name}: {error}")
            if len(invalid_files) > 10:
                print(f"    ... and {len(invalid_files) - 10} more")
        else:
            print(f"  ✓ All {len(files)} files valid")
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ All cache files are valid")
    else:
        print("✗ Some cache files are invalid. Run 'clean' to remove them.")
    print("=" * 60)


def cmd_clean(args):
    """Clean old or invalid cache files."""
    print("\n" + "=" * 60)
    print("Cleaning Cache")
    print("=" * 60)
    
    removed_count = 0
    removed_size_mb = 0
    
    # Parse age threshold
    if args.older_than:
        if args.older_than.endswith('d'):
            days = int(args.older_than[:-1])
        elif args.older_than.endswith('w'):
            days = int(args.older_than[:-1]) * 7
        elif args.older_than.endswith('m'):
            days = int(args.older_than[:-1]) * 30
        else:
            days = int(args.older_than)
        
        cutoff = datetime.now() - timedelta(days=days)
        print(f"Removing files older than {args.older_than} ({cutoff.strftime('%Y-%m-%d')})")
    else:
        cutoff = None
    
    # Filter by type if specified
    types_to_clean = [args.type] if args.type else list(CACHE_CONFIG.keys())
    
    for cache_type in types_to_clean:
        if cache_type not in CACHE_CONFIG:
            print(f"Unknown cache type: {cache_type}")
            continue
        
        config = CACHE_CONFIG[cache_type]
        files = get_cache_files(cache_type)
        
        print(f"\nCleaning {config['desc']}...")
        
        for f in files:
            should_remove = False
            
            # Check age
            if cutoff:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    should_remove = True
            
            # Check validity if requested
            if args.invalid:
                valid, _ = verify_file(f)
                if not valid:
                    should_remove = True
            
            if should_remove:
                size_mb = get_file_size_mb(f)
                try:
                    f.unlink()
                    removed_count += 1
                    removed_size_mb += size_mb
                    if args.verbose:
                        print(f"  Removed: {f.name}")
                except OSError as e:
                    print(f"  Error removing {f.name}: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Removed {removed_count} files ({format_size(removed_size_mb)})")
    print("=" * 60)


def cmd_clear(args):
    """Clear specific or all caches."""
    print("\n" + "=" * 60)
    print("Clearing Cache")
    print("=" * 60)
    
    if not args.force:
        print("\nWARNING: This will delete all cached files!")
        response = input("Type 'yes' to continue: ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    types_to_clear = [args.type] if args.type else list(CACHE_CONFIG.keys())
    
    for cache_type in types_to_clear:
        if cache_type not in CACHE_CONFIG:
            print(f"Unknown cache type: {cache_type}")
            continue
        
        config = CACHE_CONFIG[cache_type]
        cache_dir = get_cache_dir(cache_type)
        
        print(f"\nClearing {config['desc']}...")
        
        if cache_dir.exists():
            files = list(cache_dir.glob(f'*{config["ext"]}'))
            total_size_mb = sum(get_file_size_mb(f) for f in files)
            
            for f in files:
                f.unlink()
            
            print(f"  Removed {len(files)} files ({format_size(total_size_mb)})")
        else:
            print(f"  Directory does not exist")
    
    print("\n" + "=" * 60)
    print("Cache cleared")
    print("=" * 60)


def cmd_rebuild(args):
    """Rebuild all caches."""
    print("\n" + "=" * 60)
    print("Rebuilding Caches")
    print("=" * 60)
    
    if not args.manifest:
        print("Error: --manifest is required for rebuild")
        sys.exit(1)
    
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        sys.exit(1)
    
    # First clear existing caches
    print("\nStep 1: Clearing existing caches...")
    for cache_type in ['mel', 'mfcc', 'raw', 'teacher']:
        cache_dir = get_cache_dir(cache_type)
        if cache_dir.exists():
            files = list(cache_dir.glob('*'))
            for f in files:
                f.unlink()
            print(f"  Cleared {cache_type} cache")
    
    # Rebuild features
    print("\nStep 2: Rebuilding feature caches...")
    import subprocess
    
    cmd = [
        sys.executable, "-m", "scripts.cache_features",
        "--manifest", str(manifest_path),
        "--output_dir", "cached_features",
    ]
    
    if args.parallel:
        cmd.extend(["--parallel", str(args.parallel)])
    
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("  ✗ Feature caching failed")
        return
    print("  ✓ Feature caching complete")
    
    # Rebuild teacher
    print("\nStep 3: Rebuilding teacher cache...")
    cmd = [
        sys.executable, "-m", "scripts.cache_teacher",
        "--manifest", str(manifest_path),
        "--output_dir", "teacher_probs",
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("  ✗ Teacher caching failed")
        return
    print("  ✓ Teacher caching complete")
    
    print("\n" + "=" * 60)
    print("Rebuild complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Cache Manager for VAD Distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show cache status
  python scripts/cache_manager.py status
  
  # Show detailed statistics
  python scripts/cache_manager.py stats
  
  # Verify all cached files
  python scripts/cache_manager.py verify
  
  # Remove files older than 30 days
  python scripts/cache_manager.py clean --older-than 30d
  
  # Remove invalid files
  python scripts/cache_manager.py clean --invalid
  
  # Clear all mel spectrograms
  python scripts/cache_manager.py clear --type mel
  
  # Rebuild all caches
  python scripts/cache_manager.py rebuild --manifest manifests/torgo_sentences.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show cache status')
    status_parser.add_argument('--manifest', type=str, 
                               default='manifests/torgo_sentences.csv',
                               help='Manifest for counting expected files')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show detailed statistics')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify cache files')
    verify_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Show progress')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean old/invalid cache')
    clean_parser.add_argument('--older-than', type=str,
                              help='Remove files older than (e.g., 30d, 4w, 1m)')
    clean_parser.add_argument('--invalid', action='store_true',
                              help='Remove invalid/corrupted files')
    clean_parser.add_argument('--type', choices=list(CACHE_CONFIG.keys()),
                              help='Only clean specific cache type')
    clean_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Show deleted files')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache')
    clear_parser.add_argument('--type', choices=list(CACHE_CONFIG.keys()),
                              help='Only clear specific cache type')
    clear_parser.add_argument('--force', action='store_true',
                              help='Skip confirmation')
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild all caches')
    rebuild_parser.add_argument('--manifest', type=str, required=True,
                                help='Path to manifest CSV')
    rebuild_parser.add_argument('--parallel', type=int,
                                help='Number of parallel workers')
    rebuild_parser.add_argument('--device', type=str, default='cuda',
                                help='Device for teacher inference')
    rebuild_parser.add_argument('--batch-size', type=int,
                                help='Batch size for teacher inference')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to command handler
    commands = {
        'status': cmd_status,
        'stats': cmd_stats,
        'verify': cmd_verify,
        'clean': cmd_clean,
        'clear': cmd_clear,
        'rebuild': cmd_rebuild,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
