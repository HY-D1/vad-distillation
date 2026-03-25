#!/usr/bin/env python3
"""Validation-based model selection over candidate VAD methods."""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(v: str, default: float = 0.0) -> float:
    try:
        if v is None or v == "" or str(v).lower() == "none":
            return default
        return float(v)
    except ValueError:
        return default


def aggregate(rows: List[Dict]) -> Dict[str, Optional[float]]:
    tp = sum(int(to_float(r.get("tp", "0"))) for r in rows)
    tn = sum(int(to_float(r.get("tn", "0"))) for r in rows)
    fp = sum(int(to_float(r.get("fp", "0"))) for r in rows)
    fn = sum(int(to_float(r.get("fn", "0"))) for r in rows)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    auc_values = []
    for r in rows:
        val = r.get("auc")
        if val is None or val == "" or str(val).lower() == "none":
            continue
        try:
            auc_values.append(float(val))
        except ValueError:
            continue

    return {
        "count": len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "auc_mean": float(np.mean(auc_values)) if auc_values else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "segment_iou_mean": float(np.mean([to_float(r.get("segment_iou", "0")) for r in rows])) if rows else 0.0,
    }


def fmt(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "N/A"
    return f"{v:.6f}"


def maybe_plot(candidate_rows: List[Dict], output_path: Path) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    methods = [r["method"] for r in candidate_rows]
    val_f1 = [to_float(r["val_f1"]) for r in candidate_rows]
    test_f1 = [to_float(r["test_f1"]) for r in candidate_rows]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, val_f1, w, label="Validation F1", color="#2a9d8f")
    ax.bar(x + w / 2, test_f1, w, label="Test F1", color="#457b9d")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Model Selection Candidates (Validation vs Test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-based model selection")
    parser.add_argument("--input", type=str, default="outputs/personal_vad/ground_truth_eval/results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/personal_vad/model_selection")
    parser.add_argument("--selection-metric", type=str, default="f1", choices=["f1", "accuracy", "precision", "recall"])  # no AUC by default
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input results not found: {input_path}")

    rows = load_rows(input_path)
    methods = sorted({r.get("method", "") for r in rows if r.get("method")})

    by_method_split: Dict[str, Dict[str, List[Dict]]] = {m: {"val": [], "test": []} for m in methods}
    for r in rows:
        method = r.get("method")
        split = r.get("split")
        if method in by_method_split and split in by_method_split[method]:
            by_method_split[method][split].append(r)

    candidate_rows: List[Dict] = []
    for method in methods:
        val_stats = aggregate(by_method_split[method]["val"]) if by_method_split[method]["val"] else None
        test_stats = aggregate(by_method_split[method]["test"]) if by_method_split[method]["test"] else None

        candidate_rows.append(
            {
                "method": method,
                "val_count": val_stats["count"] if val_stats else 0,
                "val_f1": val_stats["f1"] if val_stats else "",
                "val_accuracy": val_stats["accuracy"] if val_stats else "",
                "val_precision": val_stats["precision"] if val_stats else "",
                "val_recall": val_stats["recall"] if val_stats else "",
                "test_count": test_stats["count"] if test_stats else 0,
                "test_f1": test_stats["f1"] if test_stats else "",
                "test_accuracy": test_stats["accuracy"] if test_stats else "",
                "test_precision": test_stats["precision"] if test_stats else "",
                "test_recall": test_stats["recall"] if test_stats else "",
            }
        )

    csv_path = output_dir / "candidate_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()) if candidate_rows else ["method"])
        writer.writeheader()
        writer.writerows(candidate_rows)

    eligible = [r for r in candidate_rows if to_float(str(r["val_count"])) > 0]

    summary: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selection_metric": args.selection_metric,
        "candidates": candidate_rows,
        "selection": None,
        "status": "blocked" if not eligible else "ok",
    }

    txt_lines = [
        "=" * 70,
        "MODEL SELECTION SUMMARY",
        "=" * 70,
        "",
        f"Selection metric: {args.selection_metric}",
        "Validation split is used for selection; test split is used for final reporting.",
        "",
    ]

    if not eligible:
        txt_lines.extend(
            [
                "SELECTION BLOCKED",
                "-" * 70,
                "No candidates have validation rows in the input results.",
                "Run ground-truth evaluation with validation split evidence before selecting a winner.",
            ]
        )
    else:
        winner = max(eligible, key=lambda r: to_float(str(r[f"val_{args.selection_metric}"])))
        summary["selection"] = {
            "winner": winner["method"],
            "metric": args.selection_metric,
            "validation_score": to_float(str(winner[f"val_{args.selection_metric}"])),
            "test_f1": to_float(str(winner["test_f1"])),
            "test_accuracy": to_float(str(winner["test_accuracy"])),
        }

        txt_lines.extend(
            [
                "WINNER",
                "-" * 70,
                f"Selected model: {winner['method']}",
                f"Validation {args.selection_metric}: {fmt(to_float(str(winner[f'val_{args.selection_metric}'])))}",
                f"Test F1: {fmt(to_float(str(winner['test_f1'])))}",
                f"Test Accuracy: {fmt(to_float(str(winner['test_accuracy'])))}",
                "",
                "CANDIDATES",
                "-" * 70,
            ]
        )
        for r in candidate_rows:
            txt_lines.append(
                f"{r['method']}: val_f1={fmt(to_float(str(r['val_f1'])))} | test_f1={fmt(to_float(str(r['test_f1'])))}"
            )

    plot_path = maybe_plot(candidate_rows, output_dir / "candidate_comparison.png") if candidate_rows else None
    if plot_path:
        summary["plot"] = plot_path

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    txt_path = output_dir / "summary.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")
    if plot_path:
        print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
