# Deprecated Scripts

These scripts are kept for reference but superseded by newer implementations.

## Moved Scripts

| Script | Original Location | Reason | Replacement |
|--------|-------------------|--------|-------------|
| `precompute_all.py` | `scripts/core/` | Superseded by modular caching system | Use `scripts/data/cache_teacher.py` and `scripts/data/cache_features.py` |
| `cache_manager.py` | `scripts/data/` | Cache management utility, no longer needed | New caching system handles this automatically |
| `generate_hard_labels.py` | `scripts/data/` | Specialized threshold-based label generation | Use teacher soft labels directly or `cache_teacher.py` |
| `compare_verification.py` | `scripts/analysis/` | One-time platform verification | Use `scripts/analysis/compare_platforms.py` instead |
| `extract_predictions.py` | `scripts/analysis/` | NPZ extraction utility | Use analysis utilities in `compare_methods.py` or `analyze_week2.py` |
| `generate_experiment_matrix.py` | `scripts/analysis/` | Week 2 specific batch experiment tool | Use `scripts/core/run_sweep.py` for parameter sweeps |

## Notes

- These scripts are **preserved intact** for historical reference
- No active code imports from these scripts
- For new work, use the replacements listed above
- See `AGENTS.md` for current recommended workflow
