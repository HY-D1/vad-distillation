#!/usr/bin/env python3
"""
Deep environment validation for VAD distillation project.

This script performs comprehensive checks on the environment and dependencies.
Used by start.sh for the setup mode.

Usage:
    python scripts/utils/check_environment.py [--json]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.resolve()


def check_python_version() -> Tuple[bool, str, Dict]:
    """Check Python version >= 3.8."""
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    is_valid = version_info.major >= 3 and version_info.minor >= 8
    
    details = {
        "version": version_str,
        "major": version_info.major,
        "minor": version_info.minor,
        "micro": version_info.micro,
        "required": ">= 3.8"
    }
    
    return is_valid, version_str, details


def check_pip() -> Tuple[bool, str, Dict]:
    """Check pip availability."""
    try:
        result = subprocess.run(
            ["pip", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.split()[1]
        return True, version, {"version": version}
    except Exception as e:
        return False, str(e), {"error": str(e)}


def check_pytorch() -> Tuple[bool, str, Dict]:
    """Check PyTorch installation and capabilities."""
    try:
        import torch
        
        details = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            details["cuda_version"] = torch.version.cuda
            details["cuda_device_count"] = torch.cuda.device_count()
            details["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        
        if hasattr(torch.backends, 'mps'):
            details["mps_available"] = torch.backends.mps.is_available()
        else:
            details["mps_available"] = False
        
        info_str = f"{torch.__version__}"
        if details["cuda_available"]:
            info_str += f" (CUDA {details.get('cuda_version', 'unknown')})"
        elif details["mps_available"]:
            info_str += " (MPS)"
        else:
            info_str += " (CPU)"
        
        return True, info_str, details
    except ImportError as e:
        return False, str(e), {"error": str(e)}


def check_audio_deps() -> Tuple[bool, str, Dict]:
    """Check audio processing dependencies."""
    deps = {}
    all_ok = True
    
    try:
        import torchaudio
        deps["torchaudio"] = torchaudio.__version__
    except ImportError:
        deps["torchaudio"] = "missing"
        all_ok = False
    
    try:
        import librosa
        deps["librosa"] = librosa.__version__
    except ImportError:
        deps["librosa"] = "missing"
        all_ok = False
    
    try:
        import soundfile
        deps["soundfile"] = soundfile.__version__
    except ImportError:
        deps["soundfile"] = "missing"
        all_ok = False
    
    info_str = ", ".join([f"{k}={v}" for k, v in deps.items()])
    return all_ok, info_str, deps


def check_ml_deps() -> Tuple[bool, str, Dict]:
    """Check ML and data dependencies."""
    deps = {}
    all_ok = True
    
    try:
        import numpy
        deps["numpy"] = numpy.__version__
    except ImportError:
        deps["numpy"] = "missing"
        all_ok = False
    
    try:
        import pandas
        deps["pandas"] = pandas.__version__
    except ImportError:
        deps["pandas"] = "missing"
        all_ok = False
    
    try:
        import sklearn
        deps["scikit-learn"] = sklearn.__version__
    except ImportError:
        deps["scikit-learn"] = "missing"
        all_ok = False
    
    try:
        import yaml
        deps["pyyaml"] = "installed"
    except ImportError:
        deps["pyyaml"] = "missing"
        all_ok = False
    
    info_str = ", ".join([f"{k}={v}" for k, v in deps.items()])
    return all_ok, info_str, deps


def check_data_structure(data_dir: Path) -> Tuple[bool, str, Dict]:
    """Check TORGO data structure."""
    results = {
        "exists": False,
        "speakers": [],
        "total_wav_files": 0,
        "issues": []
    }
    
    if not data_dir.exists():
        results["issues"].append(f"Data directory not found: {data_dir}")
        return False, "not found", results
    
    results["exists"] = True
    
    # Check for speaker directories
    speaker_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    results["speakers"] = [d.name for d in speaker_dirs]
    
    if not speaker_dirs:
        results["issues"].append("No speaker directories found")
        return False, "empty", results
    
    # Count WAV files
    total_wavs = 0
    for speaker_dir in speaker_dirs:
        wav_files = list(speaker_dir.rglob("*.wav"))
        total_wavs += len(wav_files)
    
    results["total_wav_files"] = total_wavs
    
    info_str = f"{len(speaker_dirs)} speakers, {total_wavs} WAV files"
    is_valid = len(speaker_dirs) >= 15 and total_wavs > 0
    
    return is_valid, info_str, results


def check_manifests(manifests_dir: Path) -> Tuple[bool, str, Dict]:
    """Check manifest files."""
    results = {
        "exists": False,
        "files": [],
        "issues": []
    }
    
    if not manifests_dir.exists():
        results["issues"].append(f"Manifests directory not found: {manifests_dir}")
        return False, "not found", results
    
    results["exists"] = True
    
    csv_files = list(manifests_dir.glob("*.csv"))
    results["files"] = [f.name for f in csv_files]
    
    info_str = f"{len(csv_files)} CSV files"
    is_valid = len(csv_files) >= 1
    
    return is_valid, info_str, results


def check_splits(splits_dir: Path) -> Tuple[bool, str, Dict]:
    """Check split files."""
    results = {
        "exists": False,
        "folds": [],
        "issues": []
    }
    
    if not splits_dir.exists():
        results["issues"].append(f"Splits directory not found: {splits_dir}")
        return False, "not found", results
    
    results["exists"] = True
    
    fold_files = list(splits_dir.glob("fold_*.json"))
    results["folds"] = [f.stem.replace("fold_", "") for f in fold_files]
    
    info_str = f"{len(fold_files)} fold files"
    is_valid = len(fold_files) >= 15
    
    return is_valid, info_str, results


def check_configs(configs_dir: Path) -> Tuple[bool, str, Dict]:
    """Check config files."""
    results = {
        "exists": False,
        "files": [],
        "issues": []
    }
    
    if not configs_dir.exists():
        results["issues"].append(f"Configs directory not found: {configs_dir}")
        return False, "not found", results
    
    results["exists"] = True
    
    yaml_files = list(configs_dir.glob("*.yaml"))
    results["files"] = [f.name for f in yaml_files]
    
    info_str = f"{len(yaml_files)} YAML files"
    is_valid = len(yaml_files) >= 1
    
    return is_valid, info_str, results


def check_imports(project_root: Path) -> Tuple[bool, str, Dict]:
    """Check if project modules can be imported."""
    results = {
        "project_root": str(project_root),
        "imports": {}
    }
    
    sys.path.insert(0, str(project_root))
    
    modules = [
        ("data", "TORGODataset"),
        ("models", "create_student_model"),
        ("utils", "load_config"),
        ("models.losses", "DistillationLoss"),
    ]
    
    all_ok = True
    for module_name, attr in modules:
        try:
            module = __import__(module_name, fromlist=[attr])
            getattr(module, attr)
            results["imports"][f"{module_name}.{attr}"] = "OK"
        except Exception as e:
            results["imports"][f"{module_name}.{attr}"] = f"FAILED: {str(e)}"
            all_ok = False
    
    info_str = f"{sum(1 for v in results['imports'].values() if v == 'OK')}/{len(modules)} OK"
    return all_ok, info_str, results


def run_all_checks(project_root: Path, data_dir: Path) -> Dict[str, Any]:
    """Run all environment checks."""
    results = {
        "timestamp": "",
        "platform": sys.platform,
        "checks": {}
    }
    
    checks = [
        ("python", check_python_version, ()),
        ("pip", check_pip, ()),
        ("pytorch", check_pytorch, ()),
        ("audio_deps", check_audio_deps, ()),
        ("ml_deps", check_ml_deps, ()),
        ("data", check_data_structure, (data_dir,)),
        ("manifests", check_manifests, (project_root / "manifests",)),
        ("splits", check_splits, (project_root / "splits",)),
        ("configs", check_configs, (project_root / "configs",)),
        ("imports", check_imports, (project_root,)),
    ]
    
    all_passed = True
    for name, check_func, args in checks:
        try:
            is_valid, info, details = check_func(*args)
            results["checks"][name] = {
                "status": "PASS" if is_valid else "FAIL",
                "info": info,
                "details": details
            }
            if not is_valid:
                all_passed = False
        except Exception as e:
            results["checks"][name] = {
                "status": "ERROR",
                "info": str(e),
                "details": {}
            }
            all_passed = False
    
    results["overall_status"] = "PASS" if all_passed else "FAIL"
    
    # Add timestamp
    from datetime import datetime
    results["timestamp"] = datetime.now().isoformat()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Check environment for VAD distillation project"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to TORGO data directory"
    )
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "torgo_raw"
    
    results = run_all_checks(project_root, data_dir)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("=" * 60)
        print("Environment Check Results")
        print("=" * 60)
        print(f"\nPlatform: {results['platform']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Overall Status: {results['overall_status']}")
        print("")
        
        for name, check_result in results["checks"].items():
            status_icon = "✓" if check_result["status"] == "PASS" else "✗"
            print(f"{status_icon} {name:20s}: {check_result['info']}")
        
        print("")
        print("=" * 60)
    
    return 0 if results["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
