#!/usr/bin/env python3
"""
Week 2 Experiment Analysis Script

Analyzes hyperparameter sweep results for alpha (α) and temperature (T).

Usage:
    python scripts/analyze_week2.py --results-dir outputs/week2/ --output-dir analysis/week2/
    python scripts/analyze_week2.py --results-dir outputs/week2/ --quick-summary

Outputs:
    - analysis/week2/summary.csv
    - analysis/week2/best_config.json
    - analysis/week2/plots/*.png
    - analysis/week2/report.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will not be generated.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Week 2 hyperparameter sweep results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/week2",
        help="Directory for analysis outputs"
    )
    parser.add_argument(
        "--quick-summary",
        action="store_true",
        help="Print quick summary to stdout only"
    )
    parser.add_argument(
        "--check-failures",
        action="store_true",
        help="Check for failed experiments and list them"
    )
    parser.add_argument(
        "--teacher-auc",
        type=float,
        default=0.95,
        help="Teacher AUC for comparison (default: 0.95)"
    )
    return parser.parse_args()


def load_experiment_results(results_dir: str) -> List[Dict]:
    """Load results from all experiment subdirectories."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return []
    
    # Look for result files in subdirectories
    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue
            
        # Try to load metrics.json
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                metrics["experiment_id"] = exp_dir.name
                results.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to load {metrics_file}: {e}")
        else:
            # Try alternative: config.yaml + results.txt
            config_file = exp_dir / "config.yaml"
            results_txt = exp_dir / "results.txt"
            if config_file.exists() and results_txt.exists():
                # Parse basic info from directory name
                exp_name = exp_dir.name
                result = parse_experiment_name(exp_name)
                if result:
                    result["experiment_id"] = exp_name
                    results.append(result)
    
    return results


def parse_experiment_name(name: str) -> Optional[Dict]:
    """Parse experiment name to extract alpha, temperature, fold."""
    # Expected format: alpha_{alpha}_T_{temp}_{fold}
    # e.g., alpha_0.5_T_3_F01
    try:
        parts = name.split("_")
        if len(parts) >= 5 and parts[0] == "alpha" and parts[2] == "T":
            return {
                "alpha": float(parts[1]),
                "temperature": float(parts[3]),
                "fold": parts[4] if len(parts) > 4 else "unknown"
            }
    except (ValueError, IndexError):
        pass
    return None


def parse_config_for_params(exp_dir: Path) -> Dict:
    """Try to extract alpha, temperature, fold from config files."""
    params = {"alpha": None, "temperature": None, "fold": None}
    
    # Try config.yaml
    config_file = exp_dir / "config.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            params["alpha"] = config.get("alpha")
            params["temperature"] = config.get("temperature")
            params["fold"] = config.get("fold", config.get("test_speaker"))
        except Exception:
            pass
    
    # Try args.json
    args_file = exp_dir / "args.json"
    if args_file.exists():
        try:
            with open(args_file) as f:
                args = json.load(f)
            if params["alpha"] is None:
                params["alpha"] = args.get("alpha")
            if params["temperature"] is None:
                params["temperature"] = args.get("temperature")
            if params["fold"] is None:
                params["fold"] = args.get("fold", args.get("test_speaker"))
        except Exception:
            pass
    
    return params


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results by (alpha, temperature) combination."""
    aggregated = {}
    
    for r in results:
        alpha = r.get("alpha")
        temp = r.get("temperature")
        
        if alpha is None or temp is None:
            # Try to parse from experiment_id
            parsed = parse_experiment_name(r.get("experiment_id", ""))
            if parsed:
                alpha = parsed["alpha"]
                temp = parsed["temperature"]
            else:
                continue
        
        key = (alpha, temp)
        if key not in aggregated:
            aggregated[key] = {
                "alpha": alpha,
                "temperature": temp,
                "aucs": [],
                "miss_rates": [],
                "losses": [],
                "folds": []
            }
        
        # Extract metrics
        val_metrics = r.get("validation", r.get("val", r.get("metrics", {})))
        if isinstance(val_metrics, dict):
            auc = val_metrics.get("auc", val_metrics.get("AUC"))
            miss_rate = val_metrics.get("miss_rate", val_metrics.get("missRate", val_metrics.get("miss_rate_0.5")))
            loss = val_metrics.get("loss", val_metrics.get("val_loss"))
            
            if auc is not None:
                aggregated[key]["aucs"].append(auc)
            if miss_rate is not None:
                aggregated[key]["miss_rates"].append(miss_rate)
            if loss is not None:
                aggregated[key]["losses"].append(loss)
            
            fold = r.get("fold", r.get("experiment_id", "unknown").split("_")[-1])
            aggregated[key]["folds"].append(fold)
    
    return aggregated


def compute_statistics(aggregated: Dict) -> List[Dict]:
    """Compute mean and std for each configuration."""
    stats = []
    
    for key, data in aggregated.items():
        alpha, temp = key
        
        row = {
            "alpha": alpha,
            "temperature": temp,
            "n_folds": len(data["aucs"]),
        }
        
        if data["aucs"]:
            row["auc_mean"] = np.mean(data["aucs"])
            row["auc_std"] = np.std(data["aucs"])
            row["auc_min"] = np.min(data["aucs"])
            row["auc_max"] = np.max(data["aucs"])
        
        if data["miss_rates"]:
            row["miss_rate_mean"] = np.mean(data["miss_rates"])
            row["miss_rate_std"] = np.std(data["miss_rates"])
        
        if data["losses"]:
            row["loss_mean"] = np.mean(data["losses"])
            row["loss_std"] = np.std(data["losses"])
        
        stats.append(row)
    
    # Sort by AUC mean descending
    stats.sort(key=lambda x: x.get("auc_mean", 0), reverse=True)
    
    return stats


def find_best_config(stats: List[Dict]) -> Optional[Dict]:
    """Find the best configuration based on mean AUC."""
    if not stats:
        return None
    
    best = stats[0]
    
    # Add confidence interval
    if "auc_mean" in best and "auc_std" in best:
        best["auc_ci_lower"] = best["auc_mean"] - 1.96 * best["auc_std"] / np.sqrt(best["n_folds"])
        best["auc_ci_upper"] = best["auc_mean"] + 1.96 * best["auc_std"] / np.sqrt(best["n_folds"])
    
    return best


def save_summary_csv(stats: List[Dict], output_path: Path):
    """Save summary statistics to CSV."""
    import csv
    
    if not stats:
        print("Warning: No statistics to save")
        return
    
    # Determine fieldnames from first row
    fieldnames = list(stats[0].keys())
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    
    print(f"Saved summary CSV: {output_path}")


def save_best_config(best: Dict, output_path: Path):
    """Save best configuration to JSON."""
    with open(output_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved best config: {output_path}")


def create_plots(stats: List[Dict], output_dir: Path, teacher_auc: float = 0.95):
    """Create visualization plots."""
    if not MATPLOTLIB_AVAILABLE or not stats:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract unique values
    alphas = sorted(set(s["alpha"] for s in stats))
    temps = sorted(set(s["temperature"] for s in stats))
    
    # Create pivot table for heatmap
    auc_matrix = np.full((len(alphas), len(temps)), np.nan)
    for s in stats:
        i = alphas.index(s["alpha"])
        j = temps.index(s["temperature"])
        auc_matrix[i, j] = s.get("auc_mean", np.nan)
    
    # 1. Heatmap of AUC
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        auc_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=[f"T={t}" for t in temps],
        yticklabels=[f"α={a}" for a in alphas],
        cmap="RdYlGn",
        vmin=0.7,
        vmax=teacher_auc,
        cbar_kws={"label": "Mean AUC"}
    )
    plt.title("Week 2: AUC Heatmap (α vs Temperature)")
    plt.tight_layout()
    plt.savefig(output_dir / "auc_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved heatmap: {output_dir / 'auc_heatmap.png'}")
    
    # 2. Line plot: AUC vs Temperature for each alpha
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        alpha_stats = [s for s in stats if s["alpha"] == alpha]
        alpha_stats.sort(key=lambda x: x["temperature"])
        temps_a = [s["temperature"] for s in alpha_stats]
        aucs = [s.get("auc_mean", np.nan) for s in alpha_stats]
        stds = [s.get("auc_std", 0) for s in alpha_stats]
        plt.errorbar(temps_a, aucs, yerr=stds, marker="o", label=f"α={alpha}", capsize=5)
    
    plt.axhline(y=teacher_auc, color="r", linestyle="--", label=f"Teacher (AUC={teacher_auc})")
    plt.axhline(y=teacher_auc * 0.9, color="orange", linestyle="--", label="90% of Teacher")
    plt.xlabel("Temperature")
    plt.ylabel("Mean AUC")
    plt.title("AUC vs Temperature for Different Alpha Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "auc_vs_temperature.png", dpi=150)
    plt.close()
    print(f"Saved line plot: {output_dir / 'auc_vs_temperature.png'}")
    
    # 3. Line plot: AUC vs Alpha for each temperature
    plt.figure(figsize=(10, 6))
    for temp in temps:
        temp_stats = [s for s in stats if s["temperature"] == temp]
        temp_stats.sort(key=lambda x: x["alpha"])
        alphas_t = [s["alpha"] for s in temp_stats]
        aucs = [s.get("auc_mean", np.nan) for s in temp_stats]
        stds = [s.get("auc_std", 0) for s in temp_stats]
        plt.errorbar(alphas_t, aucs, yerr=stds, marker="s", label=f"T={temp}", capsize=5)
    
    plt.axhline(y=teacher_auc, color="r", linestyle="--", label=f"Teacher (AUC={teacher_auc})")
    plt.axhline(y=teacher_auc * 0.9, color="orange", linestyle="--", label="90% of Teacher")
    plt.xlabel("Alpha (α)")
    plt.ylabel("Mean AUC")
    plt.title("AUC vs Alpha for Different Temperature Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "auc_vs_alpha.png", dpi=150)
    plt.close()
    print(f"Saved line plot: {output_dir / 'auc_vs_alpha.png'}")
    
    # 4. Bar plot of best configurations
    plt.figure(figsize=(12, 6))
    top_5 = stats[:5]
    labels = [f"α={s['alpha']}, T={s['temperature']}" for s in top_5]
    aucs = [s.get("auc_mean", 0) for s in top_5]
    errors = [s.get("auc_std", 0) for s in top_5]
    
    plt.bar(range(len(top_5)), aucs, yerr=errors, capsize=5, color="steelblue", alpha=0.7)
    plt.axhline(y=teacher_auc, color="r", linestyle="--", label=f"Teacher (AUC={teacher_auc})")
    plt.axhline(y=teacher_auc * 0.9, color="orange", linestyle="--", label="90% of Teacher")
    plt.xticks(range(len(top_5)), labels, rotation=45, ha="right")
    plt.ylabel("Mean AUC")
    plt.title("Top 5 Configurations by Mean AUC")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "top5_configs.png", dpi=150)
    plt.close()
    print(f"Saved bar plot: {output_dir / 'top5_configs.png'}")


def generate_report(stats: List[Dict], best: Dict, output_path: Path, teacher_auc: float = 0.95):
    """Generate markdown report."""
    lines = [
        "# Week 2 Analysis Report",
        "",
        f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Configurations Tested:** {len(stats)}",
        "",
        "## Summary Statistics",
        "",
    ]
    
    if stats:
        aucs = [s.get("auc_mean", 0) for s in stats]
        lines.extend([
            f"- **Best AUC:** {max(aucs):.4f}",
            f"- **Worst AUC:** {min(aucs):.4f}",
            f"- **Mean AUC:** {np.mean(aucs):.4f}",
            f"- **Std AUC:** {np.std(aucs):.4f}",
            "",
        ])
    
    if best:
        lines.extend([
            "## Best Configuration",
            "",
            f"- **Alpha (α):** {best['alpha']}",
            f"- **Temperature (T):** {best['temperature']}",
            f"- **Mean AUC:** {best.get('auc_mean', 'N/A'):.4f}",
            f"- **AUC Std:** {best.get('auc_std', 'N/A'):.4f}",
            f"- **Number of Folds:** {best['n_folds']}",
            "",
        ])
        
        # Check if within 10% of teacher
        if "auc_mean" in best:
            ratio = best["auc_mean"] / teacher_auc
            lines.append(f"- **vs Teacher:** {ratio*100:.1f}% of teacher AUC ({teacher_auc})")
            if ratio >= 0.9:
                lines.append("  - ✅ **Within 10% of teacher (SUCCESS)**")
            else:
                lines.append(f"  - ⚠️ {(1-ratio)*100:.1f}% gap to teacher")
            lines.append("")
    
    lines.extend([
        "## All Configurations (Ranked by Mean AUC)",
        "",
        "| Rank | Alpha | Temperature | Mean AUC | Std AUC | Miss Rate | N Folds |",
        "|------|-------|-------------|----------|---------|-----------|---------|",
    ])
    
    for i, s in enumerate(stats, 1):
        lines.append(
            f"| {i} | {s['alpha']:.1f} | {s['temperature']:.1f} | "
            f"{s.get('auc_mean', 'N/A'):.4f} | {s.get('auc_std', 'N/A'):.4f} | "
            f"{s.get('miss_rate_mean', 'N/A'):.4f} | {s['n_folds']} |"
        )
    
    lines.extend([
        "",
        "## Key Findings",
        "",
    ])
    
    if stats:
        # Find best alpha
        alpha_perf = {}
        for s in stats:
            a = s["alpha"]
            if a not in alpha_perf:
                alpha_perf[a] = []
            alpha_perf[a].append(s.get("auc_mean", 0))
        
        best_alpha = max(alpha_perf.keys(), key=lambda a: np.mean(alpha_perf[a]))
        lines.append(f"1. **Best Alpha:** α={best_alpha} achieves highest average AUC")
        
        # Find best temperature
        temp_perf = {}
        for s in stats:
            t = s["temperature"]
            if t not in temp_perf:
                temp_perf[t] = []
            temp_perf[t].append(s.get("auc_mean", 0))
        
        best_temp = max(temp_perf.keys(), key=lambda t: np.mean(temp_perf[t]))
        lines.append(f"2. **Best Temperature:** T={best_temp} achieves highest average AUC")
        
        # Check sensitivity
        alpha_range = max(alpha_perf.keys()) - min(alpha_perf.keys())
        temp_range = max(temp_perf.keys()) - min(temp_perf.keys())
        
        auc_ranges = [max(v) - min(v) for v in alpha_perf.values()]
        avg_alpha_sensitivity = np.mean(auc_ranges)
        
        lines.append(f"3. **Alpha Sensitivity:** Average AUC range of {avg_alpha_sensitivity:.4f} across alphas")
        
        auc_ranges_t = [max(v) - min(v) for v in temp_perf.values()]
        avg_temp_sensitivity = np.mean(auc_ranges_t)
        lines.append(f"4. **Temperature Sensitivity:** Average AUC range of {avg_temp_sensitivity:.4f} across temperatures")
    
    lines.extend([
        "",
        "## Recommendations for Week 3",
        "",
    ])
    
    if best:
        lines.extend([
            f"1. **Use configuration:** α={best['alpha']}, T={best['temperature']}",
            "2. Verify this configuration on full LOSO (all folds)",
            "3. Compare with hard-label baseline (α=0, T=1)",
            "",
        ])
    
    lines.extend([
        "## Generated Files",
        "",
        "- `summary.csv` - All configurations with statistics",
        "- `best_config.json` - Best configuration details",
        "- `plots/auc_heatmap.png` - AUC heatmap (α vs T)",
        "- `plots/auc_vs_temperature.png` - AUC vs Temperature",
        "- `plots/auc_vs_alpha.png` - AUC vs Alpha",
        "- `plots/top5_configs.png` - Top 5 configurations",
        "",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved report: {output_path}")


def check_failures(results_dir: str) -> List[str]:
    """Check for failed or missing experiments."""
    failed = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return [f"Results directory not found: {results_dir}"]
    
    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check for metrics file
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            failed.append(f"Missing metrics: {exp_dir.name}")
            continue
        
        # Try to load and validate
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            # Check for NaN or invalid values
            val_metrics = metrics.get("validation", metrics.get("val", {}))
            if isinstance(val_metrics, dict):
                auc = val_metrics.get("auc", val_metrics.get("AUC"))
                if auc is None or (isinstance(auc, float) and (auc != auc)):  # NaN check
                    failed.append(f"Invalid AUC: {exp_dir.name}")
        except Exception as e:
            failed.append(f"Error loading {exp_dir.name}: {e}")
    
    return failed


def print_quick_summary(stats: List[Dict], best: Dict, teacher_auc: float = 0.95):
    """Print quick summary to stdout."""
    print("\n" + "=" * 60)
    print("WEEK 2 ANALYSIS - QUICK SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal configurations tested: {len(stats)}")
    
    if stats:
        aucs = [s.get("auc_mean", 0) for s in stats]
        print(f"AUC Range: {min(aucs):.4f} - {max(aucs):.4f}")
        print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    if best:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Alpha (α): {best['alpha']}")
        print(f"   Temperature (T): {best['temperature']}")
        print(f"   Mean AUC: {best.get('auc_mean', 'N/A'):.4f}")
        print(f"   Folds: {best['n_folds']}")
        
        if "auc_mean" in best:
            ratio = best["auc_mean"] / teacher_auc
            print(f"   vs Teacher: {ratio*100:.1f}%")
            if ratio >= 0.9:
                print("   ✅ Within 10% of teacher - SUCCESS!")
    
    print("\n" + "=" * 60)


def main():
    args = parse_args()
    
    # Check for failures if requested
    if args.check_failures:
        failed = check_failures(args.results_dir)
        if failed:
            print(f"\nFound {len(failed)} failed/missing experiments:")
            for f in failed:
                print(f"  - {f}")
        else:
            print("\n✅ All experiments completed successfully")
        return
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_experiment_results(args.results_dir)
    print(f"Loaded {len(results)} experiment results")
    
    if not results:
        print("Error: No results found")
        sys.exit(1)
    
    # Aggregate and compute statistics
    aggregated = aggregate_results(results)
    stats = compute_statistics(aggregated)
    best = find_best_config(stats)
    
    # Quick summary mode
    if args.quick_summary:
        print_quick_summary(stats, best, args.teacher_auc)
        return
    
    # Full analysis mode
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary CSV
    save_summary_csv(stats, output_dir / "summary.csv")
    
    # Save best config
    if best:
        save_best_config(best, output_dir / "best_config.json")
    
    # Create plots
    plots_dir = output_dir / "plots"
    create_plots(stats, plots_dir, args.teacher_auc)
    
    # Generate report
    generate_report(stats, best, output_dir / "report.md", args.teacher_auc)
    
    # Print summary
    print_quick_summary(stats, best, args.teacher_auc)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
