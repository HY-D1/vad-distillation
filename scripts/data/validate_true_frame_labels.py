#!/usr/bin/env python3
"""
Validate independent frame-level true labels for strict final metrics.

This script audits per-utterance .npy label files against a manifest and writes:
1) summary.json with overall coverage and validity
2) coverage_by_speaker.csv
3) missing_labels.csv
4) invalid_labels.csv

Expected label naming:
  - SPEAKER_SESSION_UTT.npy
  - SPEAKER_UTT.npy
where UTT can be zero-padded or raw.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def hard_labels_are_teacher_derived(labels_dir: Path) -> Tuple[bool, str]:
    """
    Detect whether a label directory appears to be teacher-derived.
    """
    labels_dir = labels_dir.resolve()
    labels_dir_lower = str(labels_dir).lower()
    if "teacher_hard_labels" in labels_dir_lower:
        return True, f"path suggests teacher-derived labels: {labels_dir}"

    meta_candidates = [
        labels_dir / "meta.json",
        labels_dir.parent / "meta_all_thresholds.json",
    ]

    for meta_path in meta_candidates:
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r") as f:
                payload = json.load(f)
        except Exception:
            continue

        if isinstance(payload, dict) and payload.get("teacher_probs_dir"):
            return True, f"{meta_path} contains teacher_probs_dir"

        summaries = payload.get("summaries")
        if isinstance(summaries, list):
            for summary in summaries:
                if isinstance(summary, dict) and summary.get("teacher_probs_dir"):
                    return True, f"{meta_path} summary contains teacher_probs_dir"

    return False, "no teacher-derived metadata marker found"


def build_label_candidates(speaker_id: str, session: str, utt_id: str) -> List[str]:
    """
    Build candidate label file names following dataset resolver conventions.
    """
    utt_id_raw = str(utt_id)
    try:
        utt_id_zfill = str(int(utt_id)).zfill(4)
    except ValueError:
        utt_id_zfill = utt_id_raw

    candidates = [
        f"{speaker_id}_{session}_{utt_id_zfill}.npy",
        f"{speaker_id}_{session}_{utt_id_raw}.npy",
        f"{speaker_id}_{utt_id_zfill}.npy",
        f"{speaker_id}_{utt_id_raw}.npy",
    ]
    return candidates


def classify_label_array(array: np.ndarray) -> Tuple[bool, bool, str]:
    """
    Validate label array and classify whether it's binary or thresholdable.

    Returns:
        (is_valid, is_binary, reason)
    """
    if array.ndim != 1:
        return False, False, f"expected 1D array, got shape {array.shape}"

    if array.size == 0:
        return False, False, "empty array"

    if not np.all(np.isfinite(array)):
        return False, False, "contains non-finite values"

    uniques = np.unique(array)
    if np.all(np.isin(uniques, [0, 1])):
        return True, True, "binary"

    min_val = float(np.min(array))
    max_val = float(np.max(array))
    if min_val >= 0.0 and max_val <= 1.0:
        return True, False, "thresholdable"

    return False, False, f"values out of [0,1] range (min={min_val:.6f}, max={max_val:.6f})"


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    """
    Write rows to CSV.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate independent frame-level true labels against a manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/torgo_sentences.csv"),
        help="Path to manifest CSV (default: manifests/torgo_sentences.csv)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/true_frame_labels"),
        help="Directory containing per-utterance true-label .npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/true_label_audit"),
        help="Directory for audit outputs",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if coverage is not 100%%, invalid labels exist, or teacher-derived labels are detected",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}")
        return 1

    manifest = pd.read_csv(args.manifest)
    required_cols = ["speaker_id", "session", "utt_id"]
    missing_cols = [col for col in required_cols if col not in manifest.columns]
    if missing_cols:
        print(f"Error: manifest missing required columns: {missing_cols}")
        return 1

    labels_dir_exists = args.labels_dir.exists()
    teacher_derived = False
    teacher_derived_evidence = "labels directory missing"
    if labels_dir_exists:
        teacher_derived, teacher_derived_evidence = hard_labels_are_teacher_derived(args.labels_dir)

    total_rows = len(manifest)
    covered_rows = 0
    invalid_rows = 0
    binary_rows = 0
    thresholdable_rows = 0

    missing_rows: List[Dict] = []
    invalid_label_rows: List[Dict] = []
    speaker_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "covered": 0,
            "missing": 0,
            "invalid": 0,
            "binary": 0,
            "thresholdable": 0,
        }
    )

    for _, row in manifest.iterrows():
        speaker_id = str(row["speaker_id"])
        session = str(row["session"])
        utt_id = str(row["utt_id"])

        speaker_stats[speaker_id]["total"] += 1
        candidates = build_label_candidates(speaker_id, session, utt_id)
        resolved_path: Optional[Path] = None

        if labels_dir_exists:
            for candidate in candidates:
                candidate_path = args.labels_dir / candidate
                if candidate_path.exists():
                    resolved_path = candidate_path
                    break

        if resolved_path is None:
            speaker_stats[speaker_id]["missing"] += 1
            missing_rows.append(
                {
                    "speaker_id": speaker_id,
                    "session": session,
                    "utt_id": utt_id,
                    "primary_expected_file": candidates[0],
                    "all_candidates": "|".join(candidates),
                }
            )
            continue

        try:
            labels = np.load(resolved_path)
        except Exception as exc:
            invalid_rows += 1
            speaker_stats[speaker_id]["invalid"] += 1
            invalid_label_rows.append(
                {
                    "speaker_id": speaker_id,
                    "session": session,
                    "utt_id": utt_id,
                    "label_file": str(resolved_path),
                    "reason": f"failed to load npy: {exc}",
                }
            )
            continue

        is_valid, is_binary, reason = classify_label_array(labels.astype(np.float32))
        if not is_valid:
            invalid_rows += 1
            speaker_stats[speaker_id]["invalid"] += 1
            invalid_label_rows.append(
                {
                    "speaker_id": speaker_id,
                    "session": session,
                    "utt_id": utt_id,
                    "label_file": str(resolved_path),
                    "reason": reason,
                }
            )
            continue

        covered_rows += 1
        speaker_stats[speaker_id]["covered"] += 1
        if is_binary:
            binary_rows += 1
            speaker_stats[speaker_id]["binary"] += 1
        else:
            thresholdable_rows += 1
            speaker_stats[speaker_id]["thresholdable"] += 1

    coverage_pct = (covered_rows / total_rows * 100.0) if total_rows > 0 else 0.0
    missing_rows_count = total_rows - covered_rows - invalid_rows

    speaker_coverage_rows: List[Dict] = []
    for speaker_id in sorted(speaker_stats.keys()):
        stats = speaker_stats[speaker_id]
        total = stats["total"]
        covered = stats["covered"]
        speaker_coverage = (covered / total * 100.0) if total > 0 else 0.0
        speaker_coverage_rows.append(
            {
                "speaker_id": speaker_id,
                "total": total,
                "covered": covered,
                "missing": stats["missing"],
                "invalid": stats["invalid"],
                "binary": stats["binary"],
                "thresholdable": stats["thresholdable"],
                "coverage_pct": round(speaker_coverage, 4),
            }
        )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "labels_dir": str(args.labels_dir),
        "labels_dir_exists": labels_dir_exists,
        "strict": args.strict,
        "teacher_derived_detected": teacher_derived,
        "teacher_derived_evidence": teacher_derived_evidence,
        "totals": {
            "rows": total_rows,
            "covered_rows": covered_rows,
            "missing_rows": missing_rows_count,
            "invalid_rows": invalid_rows,
            "binary_rows": binary_rows,
            "thresholdable_rows": thresholdable_rows,
            "coverage_pct": round(coverage_pct, 4),
        },
        "output_files": {
            "coverage_by_speaker_csv": str(args.output_dir / "coverage_by_speaker.csv"),
            "missing_labels_csv": str(args.output_dir / "missing_labels.csv"),
            "invalid_labels_csv": str(args.output_dir / "invalid_labels.csv"),
            "summary_json": str(args.output_dir / "summary.json"),
        },
    }

    write_csv(
        args.output_dir / "coverage_by_speaker.csv",
        ["speaker_id", "total", "covered", "missing", "invalid", "binary", "thresholdable", "coverage_pct"],
        speaker_coverage_rows,
    )
    write_csv(
        args.output_dir / "missing_labels.csv",
        ["speaker_id", "session", "utt_id", "primary_expected_file", "all_candidates"],
        missing_rows,
    )
    write_csv(
        args.output_dir / "invalid_labels.csv",
        ["speaker_id", "session", "utt_id", "label_file", "reason"],
        invalid_label_rows,
    )

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("TRUE FRAME LABEL AUDIT")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Labels dir: {args.labels_dir}")
    print(f"Rows: {total_rows}")
    print(f"Covered: {covered_rows}")
    print(f"Missing: {missing_rows_count}")
    print(f"Invalid: {invalid_rows}")
    print(f"Coverage: {coverage_pct:.2f}%")
    print(f"Binary labels: {binary_rows}")
    print(f"Thresholdable labels: {thresholdable_rows}")
    if teacher_derived:
        print(f"Teacher-derived warning: {teacher_derived_evidence}")
    print(f"Audit output: {args.output_dir}")

    if args.strict:
        if teacher_derived:
            print("STRICT CHECK FAILED: label directory appears teacher-derived.")
            return 1
        if coverage_pct < 100.0 or invalid_rows > 0:
            print("STRICT CHECK FAILED: coverage must be 100% and invalid count must be 0.")
            return 1
        print("STRICT CHECK PASSED: independent labels are complete and valid.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
