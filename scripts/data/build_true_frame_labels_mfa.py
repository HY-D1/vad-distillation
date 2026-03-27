#!/usr/bin/env python3
"""
Build independent frame-level true labels from MFA TextGrid alignments.

This script maps manifest rows to MFA alignment files, converts aligned speech
intervals to binary frame labels, and writes one .npy file per utterance to:
  data/true_frame_labels/

Audit artifacts are written to:
  outputs/evaluation/true_label_build/
    - summary.json
    - missing_alignments.csv
    - build_errors.csv

The output naming convention follows strict validator expectations:
  - SPEAKER_SESSION_UTT.npy (primary)
"""

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SILENCE_TOKENS = {
    "",
    "sil",
    "silence",
    "sp",
    "spn",
    "pau",
    "<eps>",
    "<unk>",
    "unk",
    "noise",
    "[noise]",
    "nsn",
}


def _is_speech_label(label: str) -> bool:
    value = label.strip().lower()
    return value not in SILENCE_TOKENS


def _parse_textgrid_long_format(text: str) -> List[Tuple[float, float, str]]:
    """
    Parse long-format TextGrid intervals.
    """
    pattern = re.compile(
        r"xmin\s*=\s*([0-9eE+\-.]+)\s*"
        r"\n\s*xmax\s*=\s*([0-9eE+\-.]+)\s*"
        r"\n\s*text\s*=\s*\"(.*?)\"",
        re.DOTALL,
    )
    intervals: List[Tuple[float, float, str]] = []
    for match in pattern.finditer(text):
        xmin = float(match.group(1))
        xmax = float(match.group(2))
        label = match.group(3).replace('""', '"')
        if xmax > xmin:
            intervals.append((xmin, xmax, label))
    return intervals


def _parse_textgrid_short_format(text: str) -> List[Tuple[float, float, str]]:
    """
    Parse short-format TextGrid intervals (time time "label").
    """
    pattern = re.compile(
        r"^\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+\"(.*)\"\s*$",
        re.MULTILINE,
    )
    intervals: List[Tuple[float, float, str]] = []
    for match in pattern.finditer(text):
        xmin = float(match.group(1))
        xmax = float(match.group(2))
        label = match.group(3).replace('""', '"')
        if xmax > xmin:
            intervals.append((xmin, xmax, label))
    return intervals


def parse_textgrid_intervals(path: Path) -> List[Tuple[float, float, str]]:
    """
    Parse intervals from a TextGrid file.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    intervals = _parse_textgrid_long_format(raw)
    if intervals:
        return intervals

    intervals = _parse_textgrid_short_format(raw)
    if intervals:
        return intervals

    raise ValueError("no intervals parsed from TextGrid")


def _zfill_utt_id(utt_id: str) -> str:
    try:
        return str(int(utt_id)).zfill(4)
    except ValueError:
        return str(utt_id)


def _candidate_alignment_paths(
    alignments_root: Path,
    row: pd.Series,
) -> Sequence[Path]:
    """
    Build candidate TextGrid paths for one manifest row.
    """
    speaker = str(row["speaker_id"])
    session = str(row["session"])
    utt_raw = str(row["utt_id"])
    utt_z = _zfill_utt_id(utt_raw)
    audio_path = Path(str(row["path"]))

    candidates: List[Path] = []

    # Common MFA outputs that mirror corpus structure.
    candidates.append(alignments_root / speaker / session / "wav_headMic" / f"{utt_z}.TextGrid")
    candidates.append(alignments_root / speaker / session / f"{utt_z}.TextGrid")
    candidates.append(alignments_root / speaker / session / "wav_headMic" / f"{utt_raw}.TextGrid")
    candidates.append(alignments_root / speaker / session / f"{utt_raw}.TextGrid")

    # Try preserving relative audio path shape under alignment root.
    audio_stem = audio_path.with_suffix(".TextGrid")
    if audio_stem.is_absolute():
        audio_rel = Path(*audio_stem.parts[1:])
    else:
        audio_rel = audio_stem
    candidates.append(alignments_root / audio_rel)

    # If manifest path includes data/torgo_raw/, strip that prefix.
    parts = list(audio_rel.parts)
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "torgo_raw":
        candidates.append(alignments_root / Path(*parts[2:]))

    # De-duplicate while preserving order.
    seen = set()
    unique: List[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _resolve_alignment_path(
    alignments_root: Path,
    row: pd.Series,
) -> Optional[Path]:
    for candidate in _candidate_alignment_paths(alignments_root, row):
        if candidate.exists():
            return candidate
    return None


def intervals_to_binary_labels(
    intervals: Iterable[Tuple[float, float, str]],
    frame_hz: float,
    total_duration: Optional[float] = None,
) -> np.ndarray:
    """
    Convert aligned intervals into frame-level binary labels.
    """
    interval_list = list(intervals)
    if not interval_list:
        raise ValueError("empty interval list")

    max_xmax = max(xmax for _, xmax, _ in interval_list)
    duration = max_xmax
    if total_duration is not None and not math.isnan(total_duration):
        duration = max(duration, float(total_duration))

    num_frames = max(1, int(math.ceil(duration * frame_hz)))
    labels = np.zeros(num_frames, dtype=np.float32)

    for xmin, xmax, text in interval_list:
        if not _is_speech_label(text):
            continue
        start = max(0, int(math.floor(xmin * frame_hz)))
        end = min(num_frames, int(math.ceil(xmax * frame_hz)))
        if end > start:
            labels[start:end] = 1.0

    return labels


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build independent frame-level labels from MFA TextGrid alignments."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/torgo_sentences.csv"),
        help="Manifest CSV path (default: manifests/torgo_sentences.csv)",
    )
    parser.add_argument(
        "--alignments-root",
        type=Path,
        default=Path("data/forced_alignments"),
        help="Root directory containing MFA TextGrid outputs",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/true_frame_labels"),
        help="Output directory for per-utterance .npy labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/true_label_build"),
        help="Directory for build audit artifacts",
    )
    parser.add_argument(
        "--frame-hz",
        type=float,
        default=125.0,
        help="Target frame rate for labels (default: 125.0)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless all manifest rows are built with zero errors",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.labels_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}")
        return 1

    manifest = pd.read_csv(args.manifest)
    required_cols = ["speaker_id", "session", "utt_id", "path"]
    missing_cols = [col for col in required_cols if col not in manifest.columns]
    if missing_cols:
        print(f"Error: manifest missing required columns: {missing_cols}")
        return 1

    missing_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    built_rows = 0
    total_rows = len(manifest)
    alignments_root_exists = args.alignments_root.exists()

    for _, row in manifest.iterrows():
        speaker = str(row["speaker_id"])
        session = str(row["session"])
        utt_raw = str(row["utt_id"])
        utt_z = _zfill_utt_id(utt_raw)
        output_name = f"{speaker}_{session}_{utt_z}.npy"
        output_path = args.labels_dir / output_name

        alignment_path = _resolve_alignment_path(args.alignments_root, row)
        if alignment_path is None:
            missing_rows.append(
                {
                    "speaker_id": speaker,
                    "session": session,
                    "utt_id": utt_raw,
                    "expected_output": output_name,
                    "candidate_alignment_files": "|".join(
                        str(path) for path in _candidate_alignment_paths(args.alignments_root, row)
                    ),
                }
            )
            continue

        try:
            intervals = parse_textgrid_intervals(alignment_path)
            duration = row.get("duration", np.nan)
            labels = intervals_to_binary_labels(
                intervals=intervals,
                frame_hz=args.frame_hz,
                total_duration=float(duration) if pd.notna(duration) else None,
            )
            np.save(output_path, labels.astype(np.float32))
            built_rows += 1
        except Exception as exc:
            error_rows.append(
                {
                    "speaker_id": speaker,
                    "session": session,
                    "utt_id": utt_raw,
                    "alignment_file": str(alignment_path),
                    "error": str(exc),
                }
            )

    missing_count = len(missing_rows)
    error_count = len(error_rows)
    coverage_pct = (built_rows / total_rows * 100.0) if total_rows else 0.0

    write_csv(
        args.output_dir / "missing_alignments.csv",
        ["speaker_id", "session", "utt_id", "expected_output", "candidate_alignment_files"],
        missing_rows,
    )
    write_csv(
        args.output_dir / "build_errors.csv",
        ["speaker_id", "session", "utt_id", "alignment_file", "error"],
        error_rows,
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "alignments_root": str(args.alignments_root),
        "alignments_root_exists": alignments_root_exists,
        "labels_dir": str(args.labels_dir),
        "frame_hz": args.frame_hz,
        "strict": args.strict,
        "totals": {
            "rows": total_rows,
            "built_rows": built_rows,
            "missing_rows": missing_count,
            "error_rows": error_count,
            "coverage_pct": round(coverage_pct, 4),
        },
        "output_files": {
            "summary_json": str(args.output_dir / "summary.json"),
            "missing_alignments_csv": str(args.output_dir / "missing_alignments.csv"),
            "build_errors_csv": str(args.output_dir / "build_errors.csv"),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 60)
    print("TRUE FRAME LABEL BUILD (MFA)")
    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"Alignments root: {args.alignments_root}")
    print(f"Rows: {total_rows}")
    print(f"Built: {built_rows}")
    print(f"Missing alignments: {missing_count}")
    print(f"Build errors: {error_count}")
    print(f"Coverage: {coverage_pct:.2f}%")
    print(f"Output labels dir: {args.labels_dir}")
    print(f"Audit output dir: {args.output_dir}")

    if args.strict and (built_rows != total_rows or error_count > 0):
        print("STRICT CHECK FAILED: full coverage with zero build errors is required.")
        return 1

    if args.strict:
        print("STRICT CHECK PASSED: all labels built successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
