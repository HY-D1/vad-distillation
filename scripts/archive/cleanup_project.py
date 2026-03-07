#!/usr/bin/env python3
"""Clean up project by removing junk files.

This script removes:
- .DS_Store files (macOS system files)
- __pycache__/ directories (Python cache)
- .ipynb_checkpoints/ directories (Jupyter checkpoints)
- *.pyc, *.pyo files (compiled Python)
- Backup files (*~, *.bak, *.tmp)

Usage:
    python scripts/cleanup_project.py         # Dry run (show what would be deleted)
    python scripts/cleanup_project.py --exec  # Actually delete files
"""

import os
import argparse
from pathlib import Path


def find_ds_store_files(project_root: Path):
    """Find all .DS_Store files."""
    return list(project_root.rglob(".DS_Store"))


def find_pycache_dirs(project_root: Path):
    """Find all __pycache__ directories."""
    return [d for d in project_root.rglob("__pycache__") if d.is_dir()]


def find_ipynb_checkpoints(project_root: Path):
    """Find all .ipynb_checkpoints directories."""
    return [d for d in project_root.rglob(".ipynb_checkpoints") if d.is_dir()]


def find_pyc_files(project_root: Path):
    """Find all compiled Python files."""
    return list(project_root.rglob("*.pyc")) + list(project_root.rglob("*.pyo"))


def find_backup_files(project_root: Path):
    """Find all backup/temp files."""
    patterns = ["*~", "*.bak", "*.tmp", "*.temp"]
    files = []
    for pattern in patterns:
        files.extend(project_root.rglob(pattern))
    return files


def find_egg_info_dirs(project_root: Path):
    """Find all *.egg-info directories."""
    return [d for d in project_root.rglob("*.egg-info") if d.is_dir()]


def remove_file(filepath: Path, dry_run: bool = True) -> bool:
    """Remove a file, or simulate if dry_run."""
    try:
        if dry_run:
            print(f"  [DRY RUN] Would delete file: {filepath}")
            return True
        else:
            filepath.unlink()
            print(f"  Deleted file: {filepath}")
            return True
    except Exception as e:
        print(f"  Error deleting {filepath}: {e}")
        return False


def remove_directory(dirpath: Path, dry_run: bool = True) -> bool:
    """Remove a directory recursively, or simulate if dry_run."""
    try:
        if dry_run:
            count = sum(1 for _ in dirpath.rglob("*") if _.is_file())
            print(f"  [DRY RUN] Would delete directory: {dirpath} ({count} files)")
            return True
        else:
            import shutil
            shutil.rmtree(dirpath)
            print(f"  Deleted directory: {dirpath}")
            return True
    except Exception as e:
        print(f"  Error deleting {dirpath}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean up project by removing junk files."
    )
    parser.add_argument(
        "--exec",
        action="store_true",
        help="Actually delete files (default is dry run)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )
    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    dry_run = not args.exec

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No files will actually be deleted")
        print("Run with --exec to actually delete files")
        print("=" * 60)
    else:
        print("=" * 60)
        print("EXECUTION MODE - Files will be deleted!")
        print("=" * 60)

    print(f"\nProject root: {project_root}\n")

    all_items = []

    # Find .DS_Store files
    print("Finding .DS_Store files...")
    ds_store_files = find_ds_store_files(project_root)
    print(f"  Found {len(ds_store_files)} .DS_Store files")
    all_items.extend(ds_store_files)

    # Find __pycache__ directories
    print("Finding __pycache__ directories...")
    pycache_dirs = find_pycache_dirs(project_root)
    print(f"  Found {len(pycache_dirs)} __pycache__ directories")
    all_items.extend(pycache_dirs)

    # Find .ipynb_checkpoints directories
    print("Finding .ipynb_checkpoints directories...")
    checkpoint_dirs = find_ipynb_checkpoints(project_root)
    print(f"  Found {len(checkpoint_dirs)} .ipynb_checkpoints directories")
    all_items.extend(checkpoint_dirs)

    # Find .pyc files
    print("Finding compiled Python files...")
    pyc_files = find_pyc_files(project_root)
    print(f"  Found {len(pyc_files)} .pyc/.pyo files")
    all_items.extend(pyc_files)

    # Find backup files
    print("Finding backup/temp files...")
    backup_files = find_backup_files(project_root)
    print(f"  Found {len(backup_files)} backup/temp files")
    all_items.extend(backup_files)

    # Find egg-info directories
    print("Finding .egg-info directories...")
    egg_info_dirs = find_egg_info_dirs(project_root)
    print(f"  Found {len(egg_info_dirs)} .egg-info directories")
    all_items.extend(egg_info_dirs)

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: Found {len(all_items)} items to clean up")
    print("=" * 60)

    if not all_items:
        print("\nNothing to clean up! Project is already clean.")
        return

    # Process items
    print("\nProcessing items...")
    deleted_count = 0
    failed_count = 0

    for item in all_items:
        if item.is_dir():
            if remove_directory(item, dry_run):
                deleted_count += 1
            else:
                failed_count += 1
        else:
            if remove_file(item, dry_run):
                deleted_count += 1
            else:
                failed_count += 1

    print("\n" + "=" * 60)
    if dry_run:
        print(f"DRY RUN COMPLETE: Would delete {deleted_count} items")
        print("Run with --exec to actually delete")
    else:
        print(f"CLEANUP COMPLETE: Deleted {deleted_count} items")
        if failed_count > 0:
            print(f"Failed to delete {failed_count} items")
    print("=" * 60)


if __name__ == "__main__":
    main()
