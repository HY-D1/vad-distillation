#!/usr/bin/env python3
"""
Generate experiment matrix JSON from parameter sweeps.

This script creates a JSON file that defines a grid of experiments to run,
sweeping over hyperparameters like alpha (distillation weight) and temperature.

Usage:
    # Basic parameter sweep
    python scripts/generate_experiment_matrix.py \
        --alphas 0.3 0.5 0.7 \
        --temperatures 1 2 3 5 \
        --folds F01 M01 FC01 \
        --base_config configs/pilot.yaml \
        --output configs/week2_matrix.json

    # Single parameter sweep with custom name
    python scripts/generate_experiment_matrix.py \
        --alphas 0.1 0.3 0.5 0.7 0.9 \
        --temperatures 3.0 \
        --folds F01 \
        --base_config configs/pilot.yaml \
        --output configs/alpha_sweep.json \
        --name-prefix "alpha_sweep"

    # Fine-grained temperature sweep
    python scripts/generate_experiment_matrix.py \
        --alphas 0.5 \
        --temperatures 1 1.5 2 2.5 3 4 5 7 10 \
        --folds F01 M01 \
        --base_config configs/pilot.yaml \
        --output configs/temp_sweep.json \
        --name-prefix "temp_sweep"

    # Add other hyperparameters
    python scripts/generate_experiment_matrix.py \
        --alphas 0.3 0.5 0.7 \
        --temperatures 2 3 \
        --folds F01 \
        --base_config configs/pilot.yaml \
        --output configs/full_sweep.json \
        --override learning_rate=0.001,0.0001 \
        --override batch_size=16,32

    # Generate from existing template
    python scripts/generate_experiment_matrix.py \
        --from-template configs/templates/week2_template.yaml \
        --output configs/week2_matrix.json
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def parse_override(override_str: str) -> tuple:
    """
    Parse an override string like 'learning_rate=0.001,0.0001'.
    
    Returns:
        Tuple of (key, values)
    """
    if '=' not in override_str:
        raise ValueError(f"Override must contain '=': {override_str}")
    
    key, values_str = override_str.split('=', 1)
    values = values_str.split(',')
    
    # Try to convert to appropriate types
    typed_values = []
    for v in values:
        v = v.strip()
        try:
            # Try int
            typed_values.append(int(v))
        except ValueError:
            try:
                # Try float
                typed_values.append(float(v))
            except ValueError:
                # Keep as string
                typed_values.append(v)
    
    return key, typed_values


def generate_experiment_name(
    alpha: float,
    temperature: float,
    fold: str,
    prefix: str = "",
    extra_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a descriptive experiment name.
    
    Args:
        alpha: Distillation alpha value
        temperature: Distillation temperature
        fold: LOSO fold
        prefix: Optional name prefix
        extra_params: Additional parameters to include in name
    
    Returns:
        Experiment name string
    """
    # Format numbers nicely
    alpha_str = f"{alpha:.1f}" if alpha == int(alpha) else f"{alpha:.2f}"
    temp_str = f"{temperature:.1f}" if temperature == int(temperature) else f"{temperature:.2f}"
    
    parts = []
    if prefix:
        parts.append(prefix)
    
    parts.append(f"alpha_{alpha_str}")
    parts.append(f"T_{temp_str}")
    
    if extra_params:
        for key, value in extra_params.items():
            if isinstance(value, float):
                value_str = f"{value:.1f}" if value == int(value) else f"{value:.2f}"
            else:
                value_str = str(value)
            parts.append(f"{key}_{value_str}")
    
    parts.append(fold)
    
    return "_".join(parts)


def generate_matrix(
    alphas: List[float],
    temperatures: List[float],
    folds: List[str],
    base_config: str,
    name_prefix: str = "",
    extra_overrides: Optional[Dict[str, List[Any]]] = None
) -> Dict[str, List[Dict]]:
    """
    Generate experiment matrix from parameter combinations.
    
    Args:
        alphas: List of alpha values
        temperatures: List of temperature values
        folds: List of LOSO folds
        base_config: Path to base config file
        name_prefix: Optional prefix for experiment names
        extra_overrides: Additional parameter overrides to sweep
    
    Returns:
        Dictionary with 'experiments' key containing list of experiment configs
    """
    experiments = []
    
    # Build list of parameter keys and their values
    param_names = ['alpha', 'temperature', 'fold']
    param_lists = [alphas, temperatures, folds]
    
    if extra_overrides:
        for key, values in extra_overrides.items():
            param_names.append(key)
            param_lists.append(values)
    
    # Generate all combinations
    for combo in itertools.product(*param_lists):
        params = dict(zip(param_names, combo))
        
        alpha = params['alpha']
        temperature = params['temperature']
        fold = params['fold']
        
        # Build overrides dict
        overrides = {
            'alpha': alpha,
            'temperature': temperature
        }
        
        # Add extra overrides
        extra_params = {}
        if extra_overrides:
            for key in extra_overrides.keys():
                overrides[key] = params[key]
                extra_params[key] = params[key]
        
        # Generate experiment name
        exp_name = generate_experiment_name(
            alpha, temperature, fold,
            prefix=name_prefix,
            extra_params=extra_params if extra_params else None
        )
        
        experiment = {
            'name': exp_name,
            'config': base_config,
            'fold': fold,
            'overrides': overrides
        }
        
        experiments.append(experiment)
    
    return {'experiments': experiments}


def load_template(template_path: str) -> Dict[str, Any]:
    """
    Load experiment template from YAML file.
    
    Template format:
        name_prefix: "week2"
        base_config: "configs/pilot.yaml"
        parameters:
          alpha: [0.3, 0.5, 0.7]
          temperature: [1, 2, 3, 5]
          fold: [F01, M01, FC01]
        # Optional: include additional parameter combinations
        include:
          - name: "baseline_no_distill"
            config: "configs/pilot.yaml"
            fold: "F01"
            overrides:
              alpha: 0.0
              temperature: 1.0
    """
    with open(template_path, 'r') as f:
        template = yaml.safe_load(f)
    
    return template


def generate_from_template(template_path: str) -> Dict[str, List[Dict]]:
    """Generate experiment matrix from a template file."""
    template = load_template(template_path)
    
    # Extract parameters
    params = template.get('parameters', {})
    alphas = params.get('alpha', [0.5])
    temperatures = params.get('temperature', [3.0])
    folds = params.get('fold', ['F01'])
    
    # Remove standard parameters from extras
    extra_params = {k: v for k, v in params.items() 
                    if k not in ['alpha', 'temperature', 'fold']}
    
    base_config = template.get('base_config', 'configs/pilot.yaml')
    name_prefix = template.get('name_prefix', '')
    
    # Generate base matrix
    matrix = generate_matrix(
        alphas=alphas,
        temperatures=temperatures,
        folds=folds,
        base_config=base_config,
        name_prefix=name_prefix,
        extra_overrides=extra_params if extra_params else None
    )
    
    # Add any custom experiments
    if 'include' in template:
        for custom_exp in template['include']:
            matrix['experiments'].append(custom_exp)
    
    # Remove any excluded combinations
    if 'exclude' in template:
        exclude_patterns = template['exclude']
        filtered = []
        for exp in matrix['experiments']:
            should_exclude = False
            for pattern in exclude_patterns:
                match = True
                for key, value in pattern.items():
                    if key == 'name':
                        if value not in exp['name']:
                            match = False
                            break
                    elif exp['overrides'].get(key) != value:
                        match = False
                        break
                if match:
                    should_exclude = True
                    break
            if not should_exclude:
                filtered.append(exp)
        matrix['experiments'] = filtered
    
    return matrix


def validate_matrix(matrix: Dict[str, List[Dict]]) -> bool:
    """
    Validate the experiment matrix structure.
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if 'experiments' not in matrix:
        raise ValueError("Matrix must contain 'experiments' key")
    
    experiments = matrix['experiments']
    if not isinstance(experiments, list):
        raise ValueError("'experiments' must be a list")
    
    required_keys = {'name', 'config', 'fold'}
    
    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise ValueError(f"Experiment {i} must be a dictionary")
        
        missing = required_keys - set(exp.keys())
        if missing:
            raise ValueError(f"Experiment {i} missing required keys: {missing}")
        
        if 'overrides' not in exp:
            exp['overrides'] = {}
        elif not isinstance(exp['overrides'], dict):
            raise ValueError(f"Experiment {i}: 'overrides' must be a dictionary")
    
    # Check for duplicate names
    names = [exp['name'] for exp in experiments]
    if len(names) != len(set(names)):
        from collections import Counter
        counts = Counter(names)
        duplicates = [name for name, count in counts.items() if count > 1]
        raise ValueError(f"Duplicate experiment names found: {duplicates}")
    
    return True


def print_matrix_info(matrix: Dict[str, List[Dict]]):
    """Print summary information about the matrix."""
    experiments = matrix['experiments']
    
    print(f"\n{'='*60}")
    print("Experiment Matrix Summary")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    
    # Count unique values
    alphas = set()
    temps = set()
    folds = set()
    
    for exp in experiments:
        overrides = exp.get('overrides', {})
        alphas.add(overrides.get('alpha'))
        temps.add(overrides.get('temperature'))
        folds.add(exp['fold'])
    
    print(f"\nParameter space:")
    print(f"  Alpha values: {len(alphas)} - {sorted(alphas)}")
    print(f"  Temperature values: {len(temps)} - {sorted(temps)}")
    print(f"  Folds: {len(folds)} - {sorted(folds)}")
    
    print(f"\nFirst 3 experiments:")
    for i, exp in enumerate(experiments[:3], 1):
        print(f"  {i}. {exp['name']}")
        print(f"     Config: {exp['config']}")
        print(f"     Fold: {exp['fold']}")
        print(f"     Overrides: {exp['overrides']}")
    
    if len(experiments) > 3:
        print(f"\n  ... and {len(experiments) - 3} more")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate experiment matrix JSON from parameter sweeps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sweep
  python scripts/generate_experiment_matrix.py \\
      --alphas 0.3 0.5 0.7 --temperatures 1 2 3 --folds F01 M01 \\
      --base_config configs/pilot.yaml --output configs/sweep.json

  # From template
  python scripts/generate_experiment_matrix.py \\
      --from-template configs/template.yaml --output configs/sweep.json
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--from-template', type=str,
                             help='Generate from template YAML file')
    input_group.add_argument('--alphas', type=float, nargs='+',
                             help='List of alpha values to sweep')
    
    # Sweep parameters (used with --alphas)
    parser.add_argument('--temperatures', type=float, nargs='+',
                        help='List of temperature values to sweep')
    parser.add_argument('--folds', type=str, nargs='+',
                        help='List of LOSO folds to sweep (e.g., F01 M01 FC01)')
    parser.add_argument('--base_config', type=str, default='configs/pilot.yaml',
                        help='Base config file path (default: configs/pilot.yaml)')
    parser.add_argument('--name-prefix', type=str, default='',
                        help='Prefix for experiment names')
    
    # Additional overrides
    parser.add_argument('--override', type=str, action='append',
                        help='Additional parameter to sweep (format: key=value1,value2)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite existing output file')
    parser.add_argument('--indent', type=int, default=2,
                        help='JSON indentation level (default: 2)')
    
    args = parser.parse_args()
    
    # Check output file
    if os.path.exists(args.output) and not args.force:
        print(f"ERROR: Output file already exists: {args.output}")
        print("Use --force to overwrite")
        sys.exit(1)
    
    # Generate matrix
    if args.from_template:
        print(f"Loading template: {args.from_template}")
        if not os.path.exists(args.from_template):
            print(f"ERROR: Template file not found: {args.from_template}")
            sys.exit(1)
        
        matrix = generate_from_template(args.from_template)
    
    else:
        # Validate required arguments for manual mode
        if not args.temperatures:
            parser.error('--temperatures is required when not using --from-template')
        if not args.folds:
            parser.error('--folds is required when not using --from-template')
        
        # Parse extra overrides
        extra_overrides = {}
        if args.override:
            for override_str in args.override:
                key, values = parse_override(override_str)
                extra_overrides[key] = values
        
        print(f"Generating matrix:")
        print(f"  Alphas: {args.alphas}")
        print(f"  Temperatures: {args.temperatures}")
        print(f"  Folds: {args.folds}")
        print(f"  Base config: {args.base_config}")
        if extra_overrides:
            print(f"  Extra overrides: {extra_overrides}")
        
        matrix = generate_matrix(
            alphas=args.alphas,
            temperatures=args.temperatures,
            folds=args.folds,
            base_config=args.base_config,
            name_prefix=args.name_prefix,
            extra_overrides=extra_overrides if extra_overrides else None
        )
    
    # Validate matrix
    try:
        validate_matrix(matrix)
    except ValueError as e:
        print(f"ERROR: Invalid matrix: {e}")
        sys.exit(1)
    
    # Print summary
    print_matrix_info(matrix)
    
    # Save matrix
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(matrix, f, indent=args.indent)
    
    print(f"Matrix saved to: {args.output}")
    
    # Print command to run
    print(f"\nTo run these experiments:")
    print(f"  python scripts/run_experiment.py \\")
    print(f"      --matrix {args.output} \\")
    print(f"      --output_dir outputs/{Path(args.output).stem}/")


# =============================================================================
# Test Code
# =============================================================================

def test_generate_matrix():
    """Test the matrix generation functionality."""
    import tempfile
    import shutil
    
    print("="*60)
    print("Testing Matrix Generator")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp(prefix="vad_matrix_test_")
    print(f"Test directory: {temp_dir}\n")
    
    try:
        # Test 1: Basic generation
        print("--- Test 1: Basic matrix generation ---")
        matrix = generate_matrix(
            alphas=[0.3, 0.5],
            temperatures=[2.0, 3.0],
            folds=['F01'],
            base_config='configs/pilot.yaml',
            name_prefix='test'
        )
        
        assert len(matrix['experiments']) == 4, f"Expected 4 experiments, got {len(matrix['experiments'])}"
        print(f"  Generated {len(matrix['experiments'])} experiments")
        
        # Check structure
        exp = matrix['experiments'][0]
        assert 'name' in exp
        assert 'config' in exp
        assert 'fold' in exp
        assert 'overrides' in exp
        print("  Structure validated")
        
        # Test 2: Name generation
        print("\n--- Test 2: Experiment name generation ---")
        name = generate_experiment_name(0.5, 3.0, 'F01')
        print(f"  Basic name: {name}")
        assert 'alpha_0.5' in name
        assert 'T_3.0' in name
        assert 'F01' in name
        
        name = generate_experiment_name(0.5, 3.0, 'F01', prefix='sweep')
        print(f"  With prefix: {name}")
        assert name.startswith('sweep')
        
        # Test 3: Validation
        print("\n--- Test 3: Matrix validation ---")
        validate_matrix(matrix)
        print("  Validation passed")
        
        # Test 4: Duplicate detection
        print("\n--- Test 4: Duplicate name detection ---")
        bad_matrix = {
            'experiments': [
                {'name': 'dup', 'config': 'a.yaml', 'fold': 'F01', 'overrides': {}},
                {'name': 'dup', 'config': 'b.yaml', 'fold': 'F01', 'overrides': {}}
            ]
        }
        try:
            validate_matrix(bad_matrix)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"  Correctly detected duplicates: {e}")
        
        # Test 5: Extra overrides
        print("\n--- Test 5: Extra parameter overrides ---")
        matrix = generate_matrix(
            alphas=[0.5],
            temperatures=[3.0],
            folds=['F01'],
            base_config='configs/pilot.yaml',
            extra_overrides={'learning_rate': [0.001, 0.0001]}
        )
        
        assert len(matrix['experiments']) == 2
        assert 'learning_rate' in matrix['experiments'][0]['overrides']
        print(f"  Generated {len(matrix['experiments'])} experiments with extra params")
        
        # Test 6: Template loading
        print("\n--- Test 6: Template loading ---")
        template = {
            'name_prefix': 'week2',
            'base_config': 'configs/pilot.yaml',
            'parameters': {
                'alpha': [0.3, 0.5],
                'temperature': [2, 3],
                'fold': ['F01']
            }
        }
        
        template_path = os.path.join(temp_dir, 'test_template.yaml')
        with open(template_path, 'w') as f:
            yaml.dump(template, f)
        
        matrix = generate_from_template(template_path)
        assert len(matrix['experiments']) == 4
        print(f"  Generated {len(matrix['experiments'])} experiments from template")
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up: {temp_dir}")


if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.argv.remove('--test')
        test_generate_matrix()
    else:
        main()
