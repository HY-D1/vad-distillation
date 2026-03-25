# Personal VAD Milestone - Demo Guide (Notebook-First)

This milestone demo is **notebook-first** and **terminal-driven**.

Primary demo file:
- `milestones/personal_vad/notebook/TinyVAD_Video_Milestone.ipynb`

Equivalent terminal runner:
- `milestones/personal_vad/demo/run_personal_milestone.sh`

## Core Milestone Logic

1. Coverage audit always runs.
2. Ground-truth metric claims are made **only if coverage gate passes**.
3. Split logic is explicit:
   - validation split is for selection work,
   - test split is for final reporting.
4. Limitations and negative findings are shown directly in outputs.

## Quick Run

```bash
cd /Users/harrydai/Desktop/HD/CS\ 6140/project/group/vad-distillation
bash milestones/personal_vad/demo/run_personal_milestone.sh
```

## Primary Evidence Files

- `outputs/personal_vad/ground_truth_eval/split_coverage_summary.json`
- `outputs/personal_vad/ground_truth_eval/coverage_report.csv`
- `outputs/personal_vad/ground_truth_eval/summary.txt` (gate passed path)
- `outputs/personal_vad/ground_truth_eval/ground_truth_claims_blocked.txt` (gate blocked path)
- `outputs/personal_vad/comparison_live_ready/summary.txt`
- `outputs/personal_vad/energy_tuning_live_ready/summary.txt`
- `outputs/personal_vad/qualitative_plots_live_ready/F01_Session1_0002.png`

## Secondary (If Time Allows)

- Model selection:
  - `outputs/personal_vad/model_selection/summary.txt`
- Error analysis:
  - `outputs/personal_vad/error_analysis/error_summary.txt`

## Course Outcomes Mapping

See:
- `demo/course_outcomes_map.md`

This maps outcomes to exact scripts, commands, and output artifacts.
