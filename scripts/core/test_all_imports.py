#!/usr/bin/env python3
"""Test that all imports work correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_model_imports():
    """Test model imports."""
    try:
        from models.tinyvad_student import TinyVAD, create_student_model
        print("✓ Model imports")
        return True
    except Exception as e:
        print(f"✗ Model imports: {e}")
        return False


def test_data_imports():
    """Test data imports."""
    try:
        from data.torgo_dataset import TORGODataset, collate_fn, load_mel_spectrogram
        print("✓ Data imports")
        return True
    except Exception as e:
        print(f"✗ Data imports: {e}")
        return False


def test_script_imports():
    """Test key script functions."""
    results = []
    
    # Test validate_torgo_setup
    try:
        import scripts.validate_torgo_setup as validate
        print("✓ validate_torgo_setup import")
        results.append(True)
    except Exception as e:
        print(f"✗ validate_torgo_setup import: {e}")
        results.append(False)
    
    # Test cache_manager
    try:
        import scripts.cache_manager as cache_mgr
        print("✓ cache_manager import")
        results.append(True)
    except Exception as e:
        print(f"✗ cache_manager import: {e}")
        results.append(False)
    
    # Test generate_experiment_matrix
    try:
        import scripts.generate_experiment_matrix as gen_matrix
        print("✓ generate_experiment_matrix import")
        results.append(True)
    except Exception as e:
        print(f"✗ generate_experiment_matrix import: {e}")
        results.append(False)
    
    # Test run_sweep
    try:
        import scripts.run_sweep as run_sweep
        print("✓ run_sweep import")
        results.append(True)
    except Exception as e:
        print(f"✗ run_sweep import: {e}")
        results.append(False)
    
    return all(results)


def test_torch_imports():
    """Test PyTorch imports."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        print("✓ PyTorch imports")
        return True
    except Exception as e:
        print(f"✗ PyTorch imports: {e}")
        return False


def test_other_imports():
    """Test other required imports."""
    results = []
    
    imports_to_test = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('yaml', 'PyYAML'),
        ('sklearn', 'scikit-learn'),
        ('librosa', 'Librosa'),
    ]
    
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✓ {display_name} import")
            results.append(True)
        except ImportError:
            print(f"⚠ {display_name} not available (optional)")
            results.append(True)  # Optional, don't fail
        except Exception as e:
            print(f"✗ {display_name} import: {e}")
            results.append(False)
    
    return all(results)


def test_model_creation():
    """Test creating model instances."""
    try:
        from models.tinyvad_student import create_student_model
        model = create_student_model()
        size_kb = model.get_model_size_kb()
        print(f"✓ Model creation (size: {size_kb:.2f} KB)")
        return True
    except Exception as e:
        print(f"✗ Model creation: {e}")
        return False


def test_dataset_creation():
    """Test creating dataset."""
    try:
        from data.torgo_dataset import TORGODataset
        from pathlib import Path
        
        manifest_path = project_root / "manifests" / "torgo_pilot.csv"
        teacher_probs_dir = project_root / "teacher_probs"
        
        if not manifest_path.exists():
            print("⚠ Dataset creation skipped (manifest not found)")
            return True
            
        dataset = TORGODataset(
            manifest_path=manifest_path,
            teacher_probs_dir=teacher_probs_dir,
            n_mels=40,
        )
        print(f"✓ Dataset creation ({len(dataset)} utterances)")
        return True
    except Exception as e:
        print(f"✗ Dataset creation: {e}")
        return False


def main():
    """Run all import tests."""
    print("=" * 60)
    print("Import Validation Tests")
    print("=" * 60)
    
    results = []
    
    print("\n1. Testing model imports...")
    results.append(("Model Imports", test_model_imports()))
    
    print("\n2. Testing data imports...")
    results.append(("Data Imports", test_data_imports()))
    
    print("\n3. Testing script imports...")
    results.append(("Script Imports", test_script_imports()))
    
    print("\n4. Testing PyTorch imports...")
    results.append(("PyTorch Imports", test_torch_imports()))
    
    print("\n5. Testing other imports...")
    results.append(("Other Imports", test_other_imports()))
    
    print("\n6. Testing model creation...")
    results.append(("Model Creation", test_model_creation()))
    
    print("\n7. Testing dataset creation...")
    results.append(("Dataset Creation", test_dataset_creation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    passed_count = sum(1 for _, r in results if r)
    total_count = len(results)
    
    print("-" * 60)
    print(f"Total: {passed_count}/{total_count} passed")
    
    if passed_count == total_count:
        print("\n✅ All import tests passed!")
        return True
    else:
        print("\n❌ Some import tests failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
