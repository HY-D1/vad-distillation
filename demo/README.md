# Personal VAD Milestone - Demo Guide

Quick start guide for recording the 5-minute milestone demo video.

## Files

- `commands.txt` - Exact commands to run during demo
- `README.md` - This file (quick reference)

## Main Documentation

See `../docs/day8_demo_workflow.md` for complete video plan with:
- Timeline (4-6 minutes)
- Talking points
- Primary and backup demo paths
- Recording checklist

## Quick Setup

```bash
# 1. Navigate to project root
cd /Users/harrydai/Desktop/HD/CS\ 6140/project/group/vad-distillation

# 2. Verify everything is ready
python scripts/personal/run_my_experiment.py --max-utterances 5

# 3. Check outputs exist
ls outputs/personal/comparison/summary.txt
ls outputs/personal/qualitative_plots/*.png
```

## Demo Sequence (5 minutes)

1. **Intro** (0:00-0:30) - Show scripts
2. **Code Overview** (0:30-1:00) - Show script contents
3. **Live Demo** (1:00-1:30) - Run comparison
4. **Outputs** (1:30-2:30) - Show results
5. **Challenge** (2:30-3:30) - Frame rate mismatch
6. **Tuning** (3:30-4:00) - Show tuning results
7. **Plots** (4:00-4:45) - Open visualizations
8. **Closing** (4:45-5:00) - Summarize

## Recording Tips

- Increase terminal font size (Cmd/Ctrl + +)
- Have file explorer open to `outputs/personal/`
- Have image viewer ready for plots
- Test audio levels before starting

## If Live Command Fails

Use backup path: Show pre-generated outputs instead of running live.

See `commands.txt` for backup commands.
