#!/usr/bin/env python3
"""
Evaluate TinyVAD and Energy VAD against PHN-derived reference labels.

Design:
1) Coverage/split audit always runs.
2) Ground-truth metric claims are enabled only when a coverage gate passes.

Outputs:
- coverage_report.csv
- split_coverage_summary.json
- ground_truth_claims_blocked.txt (when gate fails)
- results.csv, summary.json, summary.txt (when gate passes)
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from baselines.energy_vad import EnergyVAD
from models.tinyvad_student import TinyVAD
from audio_loader import load_audio_mono


SILENCE_PHONES_DEFAULT = ["sil", "noi", "noio", "sp", "spn", "pau"]


def resolve_repo_path(path_str: str) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def build_utt_id(row: Dict[str, str]) -> str:
    raw_utt = str(row["utt_id"])
    try:
        utt = f"{int(raw_utt):04d}"
    except ValueError:
        utt = raw_utt
    return f"{row['speaker_id']}_{row['session']}_{utt}"


def load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[build_utt_id(row)] = row
    return rows


def load_fold(path: Path) -> Dict:
    return json.loads(path.read_text())


def find_phn_for_row(row: Dict[str, str]) -> Tuple[Optional[Path], Optional[str]]:
    audio_path = resolve_repo_path(row["path"])
    try:
        session_dir = audio_path.parents[1]
    except IndexError:
        return None, None

    raw_utt = str(row["utt_id"])
    try:
        utt = f"{int(raw_utt):04d}"
    except ValueError:
        utt = raw_utt

    for phn_dir in ["phn_headMic", "phn_arrayMic"]:
        for ext in [".PHN", ".phn"]:
            candidate = session_dir / phn_dir / f"{utt}{ext}"
            if candidate.exists():
                return candidate, phn_dir
    return None, None


def parse_phn(phn_path: Path) -> List[Tuple[int, int, str]]:
    entries: List[Tuple[int, int, str]] = []
    for line in phn_path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            start = int(float(parts[0]))
            end = int(float(parts[1]))
        except ValueError:
            continue
        if end <= start:
            continue
        phone = parts[2].lower()
        entries.append((start, end, phone))
    return entries


def phn_entries_to_speech_intervals(
    entries: Sequence[Tuple[int, int, str]],
    phn_sr: int,
    silence_phones: Sequence[str],
) -> List[Tuple[float, float]]:
    silence = set(s.lower() for s in silence_phones)
    intervals: List[Tuple[float, float]] = []
    for start, end, phone in entries:
        if phone in silence:
            continue
        intervals.append((start / phn_sr, end / phn_sr))
    return intervals


def labels_for_times(times_sec: np.ndarray, speech_intervals: Sequence[Tuple[float, float]]) -> np.ndarray:
    labels = np.zeros(len(times_sec), dtype=np.int32)
    for start, end in speech_intervals:
        mask = (times_sec >= start) & (times_sec < end)
        labels[mask] = 1
    return labels


def align_to_len(source: np.ndarray, target_len: int) -> np.ndarray:
    if len(source) == target_len:
        return source.astype(np.float32, copy=False)
    if len(source) == 0:
        return np.zeros(target_len, dtype=np.float32)
    old_idx = np.linspace(0.0, 1.0, num=len(source))
    new_idx = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(new_idx, old_idx, source).astype(np.float32)


def frame_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)

    auc: Optional[float] = None
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, probs))
        except ValueError:
            auc = None

    frame_step = 0.01  # 10ms evaluation grid
    intersection = int(np.sum((y_true == 1) & (y_pred == 1))) * frame_step
    union = int(np.sum((y_true == 1) | (y_pred == 1))) * frame_step
    segment_iou = 1.0 if union == 0 else float(intersection / union)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "segment_iou": segment_iou,
        "speech_ratio_pred": float(np.mean(y_pred)),
    }


def aggregate_metrics(rows: List[Dict]) -> Dict[str, Optional[float]]:
    tp = sum(int(r["tp"]) for r in rows)
    tn = sum(int(r["tn"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    auc_values = [r["auc"] for r in rows if r["auc"] is not None]
    mean_auc: Optional[float] = float(np.mean(auc_values)) if auc_values else None
    segment_iou = float(np.mean([r["segment_iou"] for r in rows])) if rows else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "auc_mean_utterance": mean_auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "segment_iou_mean": segment_iou,
        "utterance_count": len(rows),
    }


def load_student(checkpoint_path: Path, device: torch.device) -> TinyVAD:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_cfg = config.get(
        "model",
        {
            "n_mels": 40,
            "cnn_channels": [14, 28],
            "gru_hidden": 32,
            "gru_layers": 2,
            "dropout": 0.1,
        },
    )
    model = TinyVAD(**model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def safe_float(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "N/A"
    return f"{v:.6f}"


def write_coverage_report(path: Path, rows: List[Dict]) -> None:
    fieldnames = [
        "split",
        "utt_id",
        "speaker_id",
        "has_manifest",
        "has_phn",
        "phn_source",
        "phn_path",
        "audio_path",
        "reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_csv(path: Path, rows: List[Dict]) -> None:
    fieldnames = [
        "split",
        "utt_id",
        "speaker_id",
        "method",
        "num_frames",
        "duration_sec",
        "speech_ratio_true",
        "speech_ratio_pred",
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "auc",
        "tp",
        "tn",
        "fp",
        "fn",
        "segment_iou",
        "phn_source",
        "phn_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coverage-gated ground-truth evaluation for TinyVAD personal milestone"
    )
    parser.add_argument("--manifest", type=str, default="manifests/torgo_sentences.csv")
    parser.add_argument("--fold", type=str, default="F01")
    parser.add_argument("--fold-config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="outputs/production_cuda/checkpoints/fold_F01_latest_best.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/personal_vad/ground_truth_eval")
    parser.add_argument("--eval-split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--max-utterances", type=int, default=50)
    parser.add_argument("--min-labeled-utterances", type=int, default=30)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    parser.add_argument("--phn-sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--student-threshold", type=float, default=0.5)
    parser.add_argument("--energy-threshold", type=float, default=0.5)
    parser.add_argument("--energy-hysteresis-high", type=float, default=0.6)
    parser.add_argument("--energy-hysteresis-low", type=float, default=0.4)
    parser.add_argument("--energy-smoothing-window", type=int, default=3)
    parser.add_argument("--silence-phones", nargs="*", default=SILENCE_PHONES_DEFAULT)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_config_path = (
        resolve_repo_path(args.fold_config)
        if args.fold_config
        else resolve_repo_path(f"splits/fold_{args.fold}.json")
    )
    if not fold_config_path.exists():
        raise FileNotFoundError(f"Fold config not found: {fold_config_path}")

    manifest_path = resolve_repo_path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    fold = load_fold(fold_config_path)

    split_to_ids = {
        "val": fold.get("val_utterances", []),
        "test": fold.get("test_utterances", []),
    }

    coverage_rows: List[Dict] = []
    split_stats: Dict[str, Dict[str, float]] = {}

    for split_name in ["val", "test"]:
        ids = split_to_ids.get(split_name, [])
        total = len(ids)
        manifest_hits = 0
        label_hits = 0

        for utt_id in ids:
            row = manifest.get(utt_id)
            if row is None:
                coverage_rows.append(
                    {
                        "split": split_name,
                        "utt_id": utt_id,
                        "speaker_id": utt_id.split("_")[0] if "_" in utt_id else "",
                        "has_manifest": False,
                        "has_phn": False,
                        "phn_source": "",
                        "phn_path": "",
                        "audio_path": "",
                        "reason": "missing_manifest_row",
                    }
                )
                continue

            manifest_hits += 1
            phn_path, phn_source = find_phn_for_row(row)
            has_phn = phn_path is not None
            if has_phn:
                label_hits += 1

            coverage_rows.append(
                {
                    "split": split_name,
                    "utt_id": utt_id,
                    "speaker_id": row["speaker_id"],
                    "has_manifest": True,
                    "has_phn": has_phn,
                    "phn_source": phn_source or "",
                    "phn_path": str(phn_path) if phn_path else "",
                    "audio_path": row["path"],
                    "reason": "" if has_phn else "missing_phn",
                }
            )

        coverage = (label_hits / total) if total else 0.0
        split_stats[split_name] = {
            "total_utterances": total,
            "manifest_utterances": manifest_hits,
            "labeled_utterances": label_hits,
            "coverage": coverage,
        }

    coverage_csv = output_dir / "coverage_report.csv"
    write_coverage_report(coverage_csv, coverage_rows)

    eval_stats = split_stats[args.eval_split]
    gate_reasons: List[str] = []
    if eval_stats["labeled_utterances"] < args.min_labeled_utterances:
        gate_reasons.append(
            f"labeled_utterances={eval_stats['labeled_utterances']} < min_labeled_utterances={args.min_labeled_utterances}"
        )
    if eval_stats["coverage"] < args.min_coverage:
        gate_reasons.append(
            f"coverage={eval_stats['coverage']:.3f} < min_coverage={args.min_coverage:.3f}"
        )

    gate_passed = len(gate_reasons) == 0

    split_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fold_id": fold.get("fold_id", args.fold),
        "eval_split": args.eval_split,
        "gate": {
            "min_labeled_utterances": args.min_labeled_utterances,
            "min_coverage": args.min_coverage,
            "passed": gate_passed,
            "reasons": gate_reasons,
        },
        "split_stats": split_stats,
        "label_source": {
            "type": "PHN files under session-level phn_* directories",
            "silence_phones": args.silence_phones,
            "notes": "PHN labels are used as best-available reference and may not perfectly match VAD boundaries.",
        },
    }

    split_summary_path = output_dir / "split_coverage_summary.json"
    split_summary_path.write_text(json.dumps(split_summary, indent=2))

    if not gate_passed:
        blocked_path = output_dir / "ground_truth_claims_blocked.txt"
        lines = [
            "GROUND-TRUTH METRIC CLAIMS BLOCKED",
            "=" * 60,
            "",
            "Coverage audit completed, but metric claims are disabled by gate.",
            f"Evaluated split: {args.eval_split}",
            f"Labeled utterances: {int(eval_stats['labeled_utterances'])}",
            f"Coverage: {eval_stats['coverage']:.2%}",
            f"Gate requirements: min_labeled={args.min_labeled_utterances}, min_coverage={args.min_coverage:.0%}",
            "",
            "Blocking reasons:",
        ]
        lines.extend([f"- {r}" for r in gate_reasons])
        lines.extend(
            [
                "",
                "No ground-truth performance claim is made in this run.",
                "Use split_coverage_summary.json and coverage_report.csv as evidence artifacts.",
            ]
        )
        blocked_path.write_text("\n".join(lines) + "\n")

        print("Coverage audit complete.")
        print(f"Gate passed: {gate_passed}")
        print(f"Wrote: {coverage_csv}")
        print(f"Wrote: {split_summary_path}")
        print(f"Wrote: {blocked_path}")
        return

    # Gate passed: evaluate methods on labeled utterances of eval split.
    device = torch.device(args.device)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    student = load_student(checkpoint_path, device)
    energy = EnergyVAD(
        frame_hop_ms=10,
        threshold=args.energy_threshold,
        hysteresis_high=args.energy_hysteresis_high,
        hysteresis_low=args.energy_hysteresis_low,
        min_speech_dur=0.25,
        min_silence_dur=0.25,
        smoothing_window=args.energy_smoothing_window,
    )

    candidate_ids = [
        r["utt_id"]
        for r in coverage_rows
        if r["split"] == args.eval_split and r["has_manifest"] and r["has_phn"]
    ]
    candidate_ids = candidate_ids[: args.max_utterances]

    results: List[Dict] = []

    for i, utt_id in enumerate(candidate_ids, 1):
        row = manifest[utt_id]
        phn_path, phn_source = find_phn_for_row(row)
        if phn_path is None:
            continue

        audio, sr = load_audio_mono(str(resolve_repo_path(row["path"])), target_sr=16000)
        duration = len(audio) / float(sr)

        student_probs = student.predict(audio, device=device, return_numpy=True)
        energy_probs, _ = energy.get_frame_probs(audio, sr=sr)

        eval_len = len(energy_probs)
        if eval_len == 0:
            continue

        eval_times = np.linspace(0.0, duration, num=eval_len, endpoint=False)
        phn_entries = parse_phn(phn_path)
        speech_intervals = phn_entries_to_speech_intervals(phn_entries, args.phn_sr, args.silence_phones)
        y_true = labels_for_times(eval_times, speech_intervals)

        student_aligned = align_to_len(np.asarray(student_probs), eval_len)
        energy_aligned = np.asarray(energy_probs, dtype=np.float32)

        method_payloads = [
            ("student", student_aligned, args.student_threshold),
            ("energy_default", energy_aligned, args.energy_threshold),
        ]

        for method_name, probs, threshold in method_payloads:
            m = frame_metrics(y_true, probs, threshold)
            results.append(
                {
                    "split": args.eval_split,
                    "utt_id": utt_id,
                    "speaker_id": row["speaker_id"],
                    "method": method_name,
                    "num_frames": int(eval_len),
                    "duration_sec": float(duration),
                    "speech_ratio_true": float(np.mean(y_true)),
                    "speech_ratio_pred": m["speech_ratio_pred"],
                    "threshold": float(threshold),
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "accuracy": m["accuracy"],
                    "auc": m["auc"],
                    "tp": m["tp"],
                    "tn": m["tn"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "segment_iou": m["segment_iou"],
                    "phn_source": phn_source or "",
                    "phn_path": str(phn_path),
                }
            )

        if i % 10 == 0:
            print(f"Processed {i}/{len(candidate_ids)} utterances")

    results_csv = output_dir / "results.csv"
    write_results_csv(results_csv, results)

    by_method: Dict[str, List[Dict]] = {}
    for r in results:
        by_method.setdefault(r["method"], []).append(r)

    method_summary = {name: aggregate_metrics(rows) for name, rows in by_method.items()}

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fold_id": fold.get("fold_id", args.fold),
        "eval_split": args.eval_split,
        "gate": split_summary["gate"],
        "evaluation": {
            "utterances_evaluated": len(candidate_ids),
            "methods": method_summary,
            "notes": [
                "Ground-truth claims are PHN-derived and should be interpreted with label-source uncertainty.",
                "Validation split should be used for selection; test split for final reporting only.",
            ],
        },
        "coverage": split_summary["split_stats"],
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    summary_txt = output_dir / "summary.txt"
    lines = [
        "=" * 70,
        "GROUND-TRUTH EVALUATION (COVERAGE-GATED)",
        "=" * 70,
        "",
        f"Fold: {summary['fold_id']}",
        f"Evaluated split: {args.eval_split}",
        f"Gate passed: {gate_passed}",
        f"Utterances evaluated: {len(candidate_ids)}",
        "",
        "SPLIT COVERAGE",
        "-" * 70,
    ]
    for split_name in ["val", "test"]:
        st = split_stats[split_name]
        lines.append(
            f"{split_name}: total={int(st['total_utterances'])}, labeled={int(st['labeled_utterances'])}, coverage={st['coverage']:.2%}"
        )

    lines.extend(["", "METHOD METRICS", "-" * 70])
    for method_name, metrics in method_summary.items():
        lines.extend(
            [
                f"{method_name}",
                f"  precision: {metrics['precision']:.4f}",
                f"  recall: {metrics['recall']:.4f}",
                f"  f1: {metrics['f1']:.4f}",
                f"  accuracy: {metrics['accuracy']:.4f}",
                f"  auc (mean utterance): {safe_float(metrics['auc_mean_utterance'])}",
                f"  segment_iou (mean): {metrics['segment_iou_mean']:.4f}",
                f"  confusion: tp={metrics['tp']}, tn={metrics['tn']}, fp={metrics['fp']}, fn={metrics['fn']}",
                "",
            ]
        )

    lines.extend(
        [
            "LIMITATIONS / NEGATIVE FINDINGS",
            "-" * 70,
            "- PHN-derived labels are best-available proxy for VAD boundaries, not perfect oracle labels.",
            "- Coverage is incomplete across TORGO speakers; claims are gated and split-specific.",
            "- Proxy-based tuning from earlier milestone remains non-definitive for true generalization.",
        ]
    )

    summary_txt.write_text("\n".join(lines) + "\n")

    print("Coverage audit and evaluation complete.")
    print(f"Wrote: {coverage_csv}")
    print(f"Wrote: {split_summary_path}")
    print(f"Wrote: {results_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_txt}")


if __name__ == "__main__":
    main()
