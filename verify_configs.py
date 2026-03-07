#!/usr/bin/env python3
"""
Configuration and Manifest Verification Script for VAD Distillation Project

This script validates:
1. YAML configuration files (syntax, paths, consistency)
2. CSV manifest files (format, duplicates, speaker IDs)
3. JSON split files (syntax, speaker coverage, overlaps)
4. Path references (existence of referenced files)
"""

import json
import csv
import yaml
from pathlib import Path
from collections import defaultdict
import sys

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_section(title):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def print_error(msg):
    print(f"{RED}✗ ERROR: {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠ WARNING: {msg}{RESET}")

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_info(msg):
    print(f"  {msg}")

# ============== CONFIG VALIDATION ==============

def validate_yaml_syntax(filepath):
    """Validate YAML file syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        data = yaml.safe_load(content)
        return True, data, None
    except yaml.YAMLError as e:
        return False, None, f"YAML syntax error: {e}"
    except Exception as e:
        return False, None, f"Error reading file: {e}"

def validate_config_paths(config, config_path, base_dir):
    """Validate path references in config."""
    issues = []
    warnings = []
    
    # Check manifest path
    if 'manifest' in config:
        manifest_path = base_dir / config['manifest']
        if not manifest_path.exists():
            issues.append(f"Manifest not found: {config['manifest']}")
        else:
            print_success(f"Manifest exists: {config['manifest']}")
    
    # Check teacher_probs_dir
    if 'teacher_probs_dir' in config:
        probs_dir = base_dir / config['teacher_probs_dir']
        if not probs_dir.exists():
            warnings.append(f"Teacher probs dir not found: {config['teacher_probs_dir']}")
        else:
            print_success(f"Teacher probs dir exists: {config['teacher_probs_dir']}")
    
    # Check output_dir (may not exist yet)
    if 'output_dir' in config:
        print_info(f"Output dir: {config['output_dir']}")
    
    return issues, warnings

def validate_config_consistency(configs):
    """Check consistency between different config files."""
    issues = []
    
    # Check that all configs use the same seed
    seeds = {name: cfg.get('seed') for name, cfg in configs.items()}
    unique_seeds = set(seeds.values())
    if len(unique_seeds) > 1:
        issues.append(f"Inconsistent seeds across configs: {seeds}")
    else:
        print_success(f"All configs use consistent seed: {list(unique_seeds)[0]}")
    
    # Check model architecture consistency in production configs
    prod_configs = {k: v for k, v in configs.items() if 'production' in k}
    if prod_configs:
        model_archs = {}
        for name, cfg in prod_configs.items():
            model = cfg.get('model', {})
            arch_key = (tuple(model.get('cnn_channels', [])), 
                       model.get('gru_hidden'), 
                       model.get('gru_layers'))
            model_archs[name] = arch_key
        
        unique_archs = set(model_archs.values())
        if len(unique_archs) > 1:
            issues.append(f"Inconsistent model architectures in production configs: {model_archs}")
        else:
            print_success("All production configs use consistent model architecture")
    
    return issues

# ============== MANIFEST VALIDATION ==============

def validate_csv_format(filepath):
    """Validate CSV file format."""
    issues = []
    warnings = []
    rows = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return False, [], ["Empty CSV file"], []
            
            expected_cols = ['speaker_id', 'session', 'utt_id', 'path', 'duration', 'text']
            if header != expected_cols:
                issues.append(f"Unexpected header: {header}")
            
            row_num = 1
            for row in reader:
                row_num += 1
                if len(row) != len(expected_cols):
                    issues.append(f"Row {row_num}: Expected {len(expected_cols)} columns, got {len(row)}")
                    continue
                rows.append(row)
                
                # Check for empty critical fields
                if not row[0]:  # speaker_id
                    issues.append(f"Row {row_num}: Empty speaker_id")
                if not row[3]:  # path
                    issues.append(f"Row {row_num}: Empty path")
                
                # Check path format
                if row[3] and not row[3].endswith('.wav'):
                    warnings.append(f"Row {row_num}: Path doesn't end with .wav: {row[3][:50]}")
        
        return len(issues) == 0, rows, issues, warnings
    except Exception as e:
        return False, [], [f"Error reading CSV: {e}"], []

def validate_manifest_against_splits(manifest_rows, splits_dir):
    """Validate that manifest speaker IDs match splits."""
    issues = []
    
    # Get all speaker IDs from manifest
    manifest_speakers = set(row[0] for row in manifest_rows if row[0])
    
    # Get all speaker IDs from splits summary
    summary_path = splits_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        split_speakers = set(summary.get('fold_ids', []))
        
        # Check for mismatches
        missing_in_splits = manifest_speakers - split_speakers
        missing_in_manifest = split_speakers - manifest_speakers
        
        if missing_in_splits:
            issues.append(f"Speakers in manifest but not in splits: {missing_in_splits}")
        if missing_in_manifest:
            issues.append(f"Speakers in splits but not in manifest: {missing_in_manifest}")
        
        if not missing_in_splits and not missing_in_manifest:
            print_success(f"All {len(manifest_speakers)} speakers match between manifest and splits")
    
    return issues

def check_manifest_duplicates(manifest_rows):
    """Check for duplicate entries in manifest."""
    issues = []
    
    # Check for duplicate paths
    paths = [row[3] for row in manifest_rows if row[3]]
    path_counts = defaultdict(int)
    for path in paths:
        path_counts[path] += 1
    
    duplicates = {p: c for p, c in path_counts.items() if c > 1}
    if duplicates:
        issues.append(f"Found {len(duplicates)} duplicate paths")
        for path, count in list(duplicates.items())[:3]:
            print_warning(f"  '{path[:60]}...' appears {count} times")
    else:
        print_success("No duplicate paths found")
    
    return issues

# ============== SPLIT VALIDATION ==============

def validate_split_file(filepath, all_speakers):
    """Validate a single split JSON file."""
    issues = []
    
    try:
        with open(filepath, 'r') as f:
            split = json.load(f)
    except json.JSONDecodeError as e:
        return [f"JSON decode error: {e}"]
    except Exception as e:
        return [f"Error reading file: {e}"]
    
    fold_id = split.get('fold_id')
    test_spk = split.get('test_speaker')
    val_spk = split.get('val_speaker')
    train_spks = split.get('train_speakers', [])
    
    # Check that speakers are valid
    for spk in [test_spk, val_spk] + train_spks:
        if spk and spk not in all_speakers:
            issues.append(f"Invalid speaker ID: {spk}")
    
    # Check for overlaps
    test_set = {test_spk} if test_spk else set()
    val_set = {val_spk} if val_spk else set()
    train_set = set(train_spks)
    
    if test_set & val_set:
        issues.append(f"Overlap between test and validation: {test_set & val_set}")
    if test_set & train_set:
        issues.append(f"Overlap between test and train: {test_set & train_set}")
    if val_set & train_set:
        issues.append(f"Overlap between validation and train: {val_set & train_set}")
    
    # Check that all speakers are covered
    all_in_split = test_set | val_set | train_set
    missing = all_speakers - all_in_split
    if missing:
        issues.append(f"Missing speakers in split: {missing}")
    
    extra = all_in_split - all_speakers
    if extra:
        issues.append(f"Extra speakers in split: {extra}")
    
    return issues

def validate_all_splits(splits_dir):
    """Validate all split files."""
    issues = []
    
    # Get expected speakers from summary
    summary_path = splits_dir / 'summary.json'
    if not summary_path.exists():
        return ["summary.json not found"]
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    expected_folds = summary.get('fold_ids', [])
    all_speakers = set(expected_folds)
    
    print_info(f"Expected {len(expected_folds)} folds: {expected_folds}")
    
    # Check each fold file
    found_folds = []
    for fold_id in expected_folds:
        fold_path = splits_dir / f'fold_{fold_id}.json'
        if not fold_path.exists():
            issues.append(f"Missing fold file: fold_{fold_id}.json")
        else:
            found_folds.append(fold_id)
            fold_issues = validate_split_file(fold_path, all_speakers)
            if fold_issues:
                issues.append(f"fold_{fold_id}.json: {fold_issues}")
    
    if len(found_folds) == len(expected_folds):
        print_success(f"All {len(expected_folds)} fold files present")
    
    return issues

# ============== MAIN VALIDATION ==============

def main():
    base_dir = Path('e:/harry_dai_code/2026 project/vad-distillation')
    configs_dir = base_dir / 'configs'
    manifests_dir = base_dir / 'manifests'
    splits_dir = base_dir / 'splits'
    
    all_issues = []
    all_warnings = []
    
    # ============== VALIDATE CONFIGS ==============
    print_section("CONFIG FILE VALIDATION")
    
    config_files = list(configs_dir.glob('*.yaml'))
    print_info(f"Found {len(config_files)} YAML config files")
    
    configs = {}
    for cfg_path in config_files:
        print(f"\n  Checking {cfg_path.name}...")
        valid, data, error = validate_yaml_syntax(cfg_path)
        
        if not valid:
            print_error(f"{cfg_path.name}: {error}")
            all_issues.append(f"{cfg_path.name}: {error}")
        else:
            print_success(f"Valid YAML syntax")
            configs[cfg_path.name] = data
            
            # Validate paths
            issues, warnings = validate_config_paths(data, cfg_path, base_dir)
            all_issues.extend([f"{cfg_path.name}: {i}" for i in issues])
            all_warnings.extend([f"{cfg_path.name}: {w}" for w in warnings])
            for w in warnings:
                print_warning(w)
    
    # Check consistency between configs
    if len(configs) > 1:
        print("\n  Checking consistency between configs...")
        consistency_issues = validate_config_consistency(configs)
        all_issues.extend(consistency_issues)
        for issue in consistency_issues:
            print_error(issue)
    
    # ============== VALIDATE MANIFESTS ==============
    print_section("MANIFEST VALIDATION")
    
    manifest_files = list(manifests_dir.glob('*.csv'))
    print_info(f"Found {len(manifest_files)} manifest files")
    
    for mft_path in manifest_files:
        print(f"\n  Checking {mft_path.name}...")
        valid, rows, issues, warnings = validate_csv_format(mft_path)
        
        if not valid:
            for issue in issues:
                print_error(f"{mft_path.name}: {issue}")
            all_issues.extend([f"{mft_path.name}: {i}" for i in issues])
        else:
            print_success(f"Valid CSV format ({len(rows)} rows)")
            
            # Check duplicates
            dup_issues = check_manifest_duplicates(rows)
            for issue in dup_issues:
                print_warning(f"{mft_path.name}: {issue}")
            all_warnings.extend([f"{mft_path.name}: {i}" for i in dup_issues])
            
            # Validate against splits
            if mft_path.name == 'torgo_sentences.csv':
                print("  Checking against splits...")
                split_issues = validate_manifest_against_splits(rows, splits_dir)
                for issue in split_issues:
                    print_error(issue)
                all_issues.extend([f"{mft_path.name}: {i}" for i in split_issues])
    
    # ============== VALIDATE SPLITS ==============
    print_section("SPLIT FILE VALIDATION")
    
    split_issues = validate_all_splits(splits_dir)
    for issue in split_issues:
        print_error(issue)
    all_issues.extend(split_issues)
    
    if not split_issues:
        print_success("All split files are valid")
    
    # ============== SUMMARY ==============
    print_section("VALIDATION SUMMARY")
    
    if all_issues:
        print_error(f"Found {len(all_issues)} issue(s):")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print_success("No critical issues found!")
    
    if all_warnings:
        print_warning(f"Found {len(all_warnings)} warning(s):")
        for warning in all_warnings[:10]:  # Show first 10
            print(f"  - {warning}")
        if len(all_warnings) > 10:
            print(f"  ... and {len(all_warnings) - 10} more")
    
    # Return exit code
    return 1 if all_issues else 0

if __name__ == '__main__':
    sys.exit(main())
