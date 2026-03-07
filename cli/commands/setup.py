"""
Setup command for VAD distillation.

Environment setup and dependency installation.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from cli.utils import (
    print_error,
    print_info,
    print_success,
    print_warning,
    with_project_root
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add setup-specific arguments."""
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Check only, do not install'
    )
    parser.add_argument(
        '--extras',
        type=str,
        default=None,
        help='Comma-separated extras to install (dev,cuda)'
    )
    parser.add_argument(
        '--upgrade',
        action='store_true',
        help='Upgrade existing packages'
    )


@with_project_root
def main(args: argparse.Namespace) -> int:
    """
    Execute setup command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("="*60)
    print("VAD DISTILLATION - SETUP")
    print("="*60)
    print()
    
    # Check Python version
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print_error("Python 3.8+ required")
        return 3
    print_success(f"Python {sys.version.split()[0]} OK")
    
    # Check dependencies
    print("\nChecking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print_warning(f"Missing packages: {', '.join(missing)}")
        
        if args.check_only:
            print_info("Use 'vad.py setup' to install missing packages")
            return 3
        
        # Install missing packages
        if not install_packages(missing, args.upgrade):
            return 3
    else:
        print_success("All required packages installed")
    
    # Check optional dependencies
    print("\nChecking optional dependencies...")
    check_optional_deps()
    
    # Check directory structure
    print("\nChecking directory structure...")
    check_directories()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print_success("Environment is ready!")
    print()
    print("Next steps:")
    print("  1. Place TORGO data in data/torgo_raw/")
    print("  2. Run 'python vad.py validate'")
    print("  3. Run 'python vad.py train --quick'")
    
    return 0


def check_dependencies() -> list:
    """
    Check required dependencies.
    
    Returns:
        List of missing package names
    """
    required = [
        'torch',
        'torchaudio',
        'numpy',
        'yaml',
        'tqdm',
        'sklearn',
        'librosa',
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print_success(f"  {package}")
        except ImportError:
            print_error(f"  {package} (missing)")
            missing.append(package)
    
    return missing


def install_packages(packages: list, upgrade: bool) -> bool:
    """
    Install missing packages.
    
    Args:
        packages: List of package names
        upgrade: Whether to upgrade existing
        
    Returns:
        True if successful
    """
    print(f"\nInstalling packages: {', '.join(packages)}")
    
    # Map to pip package names
    pip_packages = []
    for pkg in packages:
        if pkg == 'yaml':
            pip_packages.append('pyyaml')
        elif pkg == 'sklearn':
            pip_packages.append('scikit-learn')
        else:
            pip_packages.append(pkg)
    
    cmd = [sys.executable, '-m', 'pip', 'install']
    if upgrade:
        cmd.append('--upgrade')
    cmd.extend(pip_packages)
    
    try:
        subprocess.run(cmd, check=True)
        print_success("Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Installation failed: {e}")
        return False


def check_optional_deps() -> None:
    """Check optional dependencies."""
    optional = {
        'matplotlib': 'For plotting and visualization',
        'seaborn': 'For enhanced plots',
        'speechbrain': 'For SpeechBrain baseline',
        'onnx': 'For ONNX export',
    }
    
    for package, description in optional.items():
        try:
            __import__(package)
            print_success(f"  {package} ({description})")
        except ImportError:
            print_info(f"  {package} not installed (optional: {description})")


def check_directories() -> None:
    """Check and create required directories."""
    required_dirs = [
        'data',
        'outputs',
        'splits',
        'teacher_probs',
        'manifests',
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print_info(f"  Created: {dir_name}/")
        else:
            print_success(f"  {dir_name}/ exists")
