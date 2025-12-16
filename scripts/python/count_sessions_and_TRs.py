#!/usr/bin/env python3
"""Describe practice phase metadata across subjects."""

import json
from pathlib import Path
import numpy as np

metadata_dir = Path("inputs/practice_phase_metadata")
files = sorted(metadata_dir.glob("*.json"))

# Collect data per subject
subjects = {}
for f in files:
    data = json.loads(f.read_text())
    sub = data["subject"]
    if sub not in subjects:
        subjects[sub] = {
            "sessions": [], "runs_per_session": [], "trs_per_session": [],
            "trs_per_run": [], "levels_per_run": []
        }
    
    subjects[sub]["sessions"].append(data["session"])
    subjects[sub]["runs_per_session"].append(data["n_runs"])
    subjects[sub]["trs_per_session"].append(data["n_samples"])
    subjects[sub]["trs_per_run"].extend(data["n_trs_per_run"])
    
    # Calculate levels per run
    run_starts = data["run_onsets"]
    run_ends = [run_starts[i+1] for i in range(len(run_starts)-1)] + [data["n_samples"]]
    level_onsets = data["level_onsets"]
    for i in range(len(run_starts)):
        levels_in_run = sum(1 for onset in level_onsets if run_starts[i] <= onset < run_ends[i])
        subjects[sub]["levels_per_run"].append(levels_in_run)

# Summary stats
print("=== PER SUBJECT ===")
for sub in sorted(subjects.keys()):
    n_sess = len(subjects[sub]["sessions"])
    runs = subjects[sub]["runs_per_session"]
    trs_sess = subjects[sub]["trs_per_session"]
    trs_run = subjects[sub]["trs_per_run"]
    levels_run = subjects[sub]["levels_per_run"]
    total_trs = sum(trs_sess)
    
    print(f"\nSub-{sub:02d}:")
    print(f"  Sessions: {n_sess}")
    print(f"  Runs/session: {np.mean(runs):.1f} ± {np.std(runs):.1f} (range: {min(runs)}-{max(runs)})")
    print(f"  TRs/session: {np.mean(trs_sess):.0f} ± {np.std(trs_sess):.0f} (range: {min(trs_sess)}-{max(trs_sess)})")
    print(f"  TRs/run: {np.mean(trs_run):.0f} ± {np.std(trs_run):.0f} (range: {min(trs_run)}-{max(trs_run)})")
    print(f"  Levels/run: {np.mean(levels_run):.1f} ± {np.std(levels_run):.1f} (range: {min(levels_run)}-{max(levels_run)})")
    print(f"  Total TRs: {total_trs}")
    print(f"  Train/test (80/20): {int(total_trs*0.8)}/{int(total_trs*0.2)}")

# Overall stats
all_runs_per_sess = []
all_trs_per_sess = []
all_trs_per_run = []
all_levels_per_run = []
all_total_trs = []

for sub in subjects.values():
    all_runs_per_sess.extend(sub["runs_per_session"])
    all_trs_per_sess.extend(sub["trs_per_session"])
    all_trs_per_run.extend(sub["trs_per_run"])
    all_levels_per_run.extend(sub["levels_per_run"])
    all_total_trs.append(sum(sub["trs_per_session"]))

print("\n=== ACROSS ALL SUBJECTS ===")
print(f"Total sessions: {len(all_runs_per_sess)}")
print(f"Runs/session: {np.mean(all_runs_per_sess):.1f} ± {np.std(all_runs_per_sess):.1f}")
print(f"TRs/session: {np.mean(all_trs_per_sess):.0f} ± {np.std(all_trs_per_sess):.0f}")
print(f"TRs/run: {np.mean(all_trs_per_run):.0f} ± {np.std(all_trs_per_run):.0f}")
print(f"Levels/run: {np.mean(all_levels_per_run):.1f} ± {np.std(all_levels_per_run):.1f}")
print(f"Total TRs/subject: {np.mean(all_total_trs):.0f} ± {np.std(all_total_trs):.0f}")
print(f"Grand total TRs: {sum(all_total_trs)}")
