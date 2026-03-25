#!/usr/bin/env python3
"""Grouped error analysis for personal VAD evaluation outputs."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(v: str, default: float = 0.0) -> float:
    try:
        if v is None or v == "" or str(v).lower() == "none":
            return default
        return float(v)
    except ValueError:
        return default


def duration_bin(duration_sec: float) -> str:
    if duration_sec < 2.0:
        return "short(<2s)"
    if duration_sec < 4.0:
        return "medium(2-4s)"
    return "long(>=4s)"


def speech_ratio_bin(ratio: float) -> str:
    if ratio < 0.3:
        return "silence-heavy(<0.3)"
    if ratio <= 0.7:
        return "mixed(0.3-0.7)"
    return "speech-heavy(>0.7)"


def aggregate_group(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {
            "count": 0,
            "mean_f1": 0.0,
            "mean_accuracy": 0.0,
            "mean_segment_iou": 0.0,
            "mean_miss_rate": 0.0,
        }

    miss_rates = []
    for r in rows:
        tp = to_float(r.get("tp", "0"))
        fn = to_float(r.get("fn", "0"))
        miss = fn / (tp + fn) if (tp + fn) else 0.0
        miss_rates.append(miss)

    return {
        "count": len(rows),
        "mean_f1": float(np.mean([to_float(r.get("f1", "0")) for r in rows])),
        "mean_accuracy": float(np.mean([to_float(r.get("accuracy", "0")) for r in rows])),
        "mean_segment_iou": float(np.mean([to_float(r.get("segment_iou", "0")) for r in rows])),
        "mean_miss_rate": float(np.mean(miss_rates)),
    }


def grouped_metrics(rows: List[Dict], factor_name: str, key_fn) -> List[Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        method = r.get("method", "unknown")
        grp = key_fn(r)
        groups.setdefault((method, grp), []).append(r)

    out: List[Dict] = []
    for (method, grp), grouped in sorted(groups.items()):
        agg = aggregate_group(grouped)
        out.append(
            {
                "factor": factor_name,
                "group": grp,
                "method": method,
                "count": agg["count"],
                "mean_f1": agg["mean_f1"],
                "mean_accuracy": agg["mean_accuracy"],
                "mean_segment_iou": agg["mean_segment_iou"],
                "mean_miss_rate": agg["mean_miss_rate"],
            }
        )
    return out


def representative_examples(rows: List[Dict]) -> Dict[str, Dict]:
    by_method: Dict[str, List[Dict]] = {}
    for r in rows:
        by_method.setdefault(r.get("method", "unknown"), []).append(r)

    result: Dict[str, Dict] = {}
    for method, items in by_method.items():
        items_sorted = sorted(items, key=lambda x: to_float(x.get("f1", "0")))
        worst = items_sorted[0]
        best = items_sorted[-1]
        typical = items_sorted[len(items_sorted) // 2]
        result[method] = {
            "best": {"utt_id": best.get("utt_id"), "split": best.get("split"), "f1": to_float(best.get("f1", "0"))},
            "worst": {"utt_id": worst.get("utt_id"), "split": worst.get("split"), "f1": to_float(worst.get("f1", "0"))},
            "typical": {
                "utt_id": typical.get("utt_id"),
                "split": typical.get("split"),
                "f1": to_float(typical.get("f1", "0")),
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped error analysis for VAD methods")
    parser.add_argument("--input", type=str, default="outputs/personal_vad/ground_truth_eval/results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/personal_vad/error_analysis")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)

    grouped: List[Dict] = []
    grouped.extend(grouped_metrics(rows, "speaker", lambda r: r.get("speaker_id", "unknown")))
    grouped.extend(grouped_metrics(rows, "duration", lambda r: duration_bin(to_float(r.get("duration_sec", "0")))))
    grouped.extend(
        grouped_metrics(rows, "speech_ratio", lambda r: speech_ratio_bin(to_float(r.get("speech_ratio_true", "0"))))
    )

    grouped_csv = output_dir / "grouped_metrics.csv"
    with grouped_csv.open("w", newline="") as f:
        fieldnames = ["factor", "group", "method", "count", "mean_f1", "mean_accuracy", "mean_segment_iou", "mean_miss_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grouped)

    reps = representative_examples(rows)

    hardest: Dict[str, Dict[str, Dict]] = {}
    for factor in ["speaker", "duration", "speech_ratio"]:
        factor_rows = [r for r in grouped if r["factor"] == factor]
        by_method: Dict[str, List[Dict]] = {}
        for r in factor_rows:
            by_method.setdefault(r["method"], []).append(r)
        hardest[factor] = {}
        for method, items in by_method.items():
            hardest_group = min(items, key=lambda x: x["mean_f1"]) if items else None
            hardest[factor][method] = hardest_group if hardest_group else {}

    summary_json = output_dir / "error_summary.json"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": str(input_path),
        "num_rows": len(rows),
        "grouped_metrics": str(grouped_csv),
        "representative_examples": reps,
        "hardest_groups": hardest,
        "limitations": [
            "Analysis quality depends on PHN coverage and label fidelity.",
            "If only one split is present, cross-split conclusions are limited.",
            "Per-utterance summary metrics can hide fine-grained boundary errors.",
        ],
    }
    summary_json.write_text(json.dumps(payload, indent=2))

    summary_txt = output_dir / "error_summary.txt"
    lines = [
        "=" * 70,
        "ERROR ANALYSIS SUMMARY",
        "=" * 70,
        "",
        f"Input rows: {len(rows)}",
        f"Grouped metrics: {grouped_csv}",
        "",
        "HARDEST GROUPS BY FACTOR",
        "-" * 70,
    ]

    for factor in ["speaker", "duration", "speech_ratio"]:
        lines.append(f"{factor}:")
        for method, info in hardest.get(factor, {}).items():
            if not info:
                lines.append(f"  {method}: N/A")
                continue
            lines.append(
                f"  {method}: {info['group']} | mean_f1={info['mean_f1']:.4f} | count={int(info['count'])}"
            )

    lines.extend(["", "REPRESENTATIVE EXAMPLES", "-" * 70])
    for method, info in reps.items():
        lines.append(
            f"{method}: best={info['best']['utt_id']} ({info['best']['f1']:.4f}), "
            f"worst={info['worst']['utt_id']} ({info['worst']['f1']:.4f}), "
            f"typical={info['typical']['utt_id']} ({info['typical']['f1']:.4f})"
        )

    lines.extend(
        [
            "",
            "LIMITATIONS / CONFOUNDERS",
            "-" * 70,
            "- PHN-derived labels can disagree with perceived VAD boundaries.",
            "- Missing labels for some speakers can skew grouped comparisons.",
            "- Metrics are aggregated and may hide timing-level boundary misses.",
        ]
    )

    summary_txt.write_text("\n".join(lines) + "\n")

    print(f"Wrote: {grouped_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_txt}")


if __name__ == "__main__":
    main()
