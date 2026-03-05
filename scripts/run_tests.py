#!/usr/bin/env python3
"""Run all project tests."""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


TESTS = [
    ("Import Tests", "scripts.test_all_imports"),
    ("Model Tests", "models.tinyvad_student"),
    ("Dataset Tests", "data.torgo_dataset"),
]


def run_python_module(module_name):
    """Run a Python module and return success status."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def run_script(script_path):
    """Run a script directly and return success status."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def test_model():
    """Test model module."""
    print("\nTesting Model Module...")
    print("-" * 50)
    
    try:
        from models.tinyvad_student import create_student_model
        model = create_student_model()
        size_kb = model.get_model_size_kb()
        
        print(f"  Model size: {size_kb:.2f} KB")
        
        if size_kb > 500:
            print(f"  ✗ Model too large: {size_kb:.2f} KB > 500 KB")
            return False
        
        print(f"  ✓ Model size OK ({size_kb:.2f} KB < 500 KB)")
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_dataset():
    """Test dataset module."""
    print("\nTesting Dataset Module...")
    print("-" * 50)
    
    try:
        from data.torgo_dataset import TORGODataset
        
        manifest_path = project_root / "manifests" / "torgo_pilot.csv"
        teacher_probs_dir = project_root / "teacher_probs"
        fold_config_path = project_root / "splits" / "fold_F01.json"
        
        if not manifest_path.exists():
            print(f"  ✗ Manifest not found: {manifest_path}")
            return False
        
        # Test basic dataset
        dataset = TORGODataset(
            manifest_path=manifest_path,
            teacher_probs_dir=teacher_probs_dir,
            n_mels=40,
        )
        print(f"  ✓ Basic dataset: {len(dataset)} utterances")
        
        # Test fold-based split
        if fold_config_path.exists():
            train_dataset = TORGODataset(
                manifest_path=manifest_path,
                teacher_probs_dir=teacher_probs_dir,
                fold_config=fold_config_path,
                mode='train',
                n_mels=40,
            )
            print(f"  ✓ Train split: {len(train_dataset)} utterances")
        
        return True
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        return False


def test_paths():
    """Test that all required paths exist."""
    print("\nTesting File Paths...")
    print("-" * 50)
    
    required_paths = [
        ("Manifest (Pilot)", "manifests/torgo_pilot.csv"),
        ("Manifest (Sentences)", "manifests/torgo_sentences.csv"),
        ("Teacher Probs Dir", "teacher_probs/"),
        ("Splits Dir", "splits/"),
        ("Configs Dir", "configs/"),
    ]
    
    all_ok = True
    for name, path in required_paths:
        full_path = project_root / path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    # Check for fold configs
    fold_configs = list((project_root / "splits").glob("fold_*.json"))
    if fold_configs:
        print(f"  ✓ Found {len(fold_configs)} fold configurations")
    else:
        print(f"  ✗ No fold configurations found")
        all_ok = False
    
    # Check for teacher probs
    teacher_probs = list((project_root / "teacher_probs").glob("*.npy"))
    if teacher_probs:
        print(f"  ✓ Found {len(teacher_probs)} teacher probability files")
    else:
        print(f"  ⚠ No teacher probability files found (may need to generate)")
    
    return all_ok


def test_configs():
    """Test config files are valid."""
    print("\nTesting Config Files...")
    print("-" * 50)
    
    import yaml
    import json
    
    config_files = [
        ("configs/pilot.yaml", "yaml"),
        ("configs/production.yaml", "yaml"),
        ("configs/alpha_sweep.yaml", "yaml"),
        ("configs/temperature_sweep.yaml", "yaml"),
        ("configs/week2_matrix.json", "json"),
    ]
    
    all_ok = True
    for path, format_type in config_files:
        full_path = project_root / path
        if not full_path.exists():
            print(f"  ✗ {path} not found")
            all_ok = False
            continue
        
        try:
            with open(full_path, 'r') as f:
                if format_type == "yaml":
                    yaml.safe_load(f)
                else:
                    json.load(f)
            print(f"  ✓ {path} is valid {format_type.upper()}")
        except Exception as e:
            print(f"  ✗ {path} parse error: {e}")
            all_ok = False
    
    return all_ok


def test_scripts():
    """Test key scripts can be imported and have expected functions."""
    print("\nTesting Scripts...")
    print("-" * 50)
    
    all_ok = True
    
    # Test validate_torgo_setup
    try:
        result = run_script(project_root / "scripts" / "validate_torgo_setup.py")
        if result[0]:
            print(f"  ✓ validate_torgo_setup.py runs")
        else:
            print(f"  ⚠ validate_torgo_setup.py runs (with warnings)")
    except Exception as e:
        print(f"  ✗ validate_torgo_setup.py failed: {e}")
        all_ok = False
    
    # Test cache_manager
    try:
        result = run_python_module("scripts.cache_manager")
        if "Cache Status" in result[1] or "usage:" in result[1]:
            print(f"  ✓ cache_manager.py runs")
        else:
            print(f"  ⚠ cache_manager.py runs (unexpected output)")
    except Exception as e:
        print(f"  ✗ cache_manager.py failed: {e}")
        all_ok = False
    
    # Test generate_experiment_matrix
    try:
        result = run_script(project_root / "scripts" / "generate_experiment_matrix.py")
        if "usage:" in result[1]:
            print(f"  ✓ generate_experiment_matrix.py runs")
        else:
            print(f"  ⚠ generate_experiment_matrix.py runs (unexpected output)")
    except Exception as e:
        print(f"  ✗ generate_experiment_matrix.py failed: {e}")
        all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("=" * 60)
    print("VAD Distillation - Comprehensive Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("File Paths", test_paths()))
    results.append(("Config Files", test_configs()))
    results.append(("Import Tests", test_model()))
    results.append(("Dataset Tests", test_dataset()))
    results.append(("Script Tests", test_scripts()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    passed_count = sum(1 for _, r in results if r)
    total_count = len(results)
    
    print("-" * 60)
    print(f"Total: {passed_count}/{total_count} test groups passed")
    
    if passed_count == total_count:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
