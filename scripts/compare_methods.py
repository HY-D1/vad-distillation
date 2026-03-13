#!/usr/bin/env python3
"""
Wrapper for scripts/analysis/compare_methods.py

This file provides backward compatibility for scripts that call
the old path: scripts/compare_methods.py

The actual implementation has been moved to scripts/analysis/compare_methods.py

Usage:
    python scripts/compare_methods.py --manifest MANIFEST --methods METHODS --output-dir DIR
    (equivalent to)
    python scripts/analysis/compare_methods.py --manifest MANIFEST --methods METHODS --output-dir DIR
"""

import sys
import subprocess
import os


def main():
    """Forward all arguments to the actual implementation."""
    # Get the project root (parent of scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Path to the actual implementation
    actual_script = os.path.join(script_dir, 'analysis', 'compare_methods.py')
    
    # Forward all command line arguments
    cmd = [sys.executable, actual_script] + sys.argv[1:]
    
    # Set up environment with project root in PYTHONPATH for imports
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = project_root + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = project_root
    
    # Run the actual script with proper environment
    result = subprocess.run(cmd, cwd=project_root, env=env)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
