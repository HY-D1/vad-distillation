#!/usr/bin/env python3
"""
Wrapper for scripts/core/run_baseline.py

This file provides backward compatibility for scripts that call
the old path: scripts/run_baseline.py

The actual implementation has been moved to scripts/core/run_baseline.py

Usage:
    python scripts/run_baseline.py --method energy --manifest MANIFEST --output-dir DIR
    (equivalent to)
    python scripts/core/run_baseline.py --method energy --manifest MANIFEST --output-dir DIR
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
    actual_script = os.path.join(script_dir, 'core', 'run_baseline.py')
    
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
