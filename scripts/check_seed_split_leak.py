#!/usr/bin/env python3
"""Pre-flight: assert a seeded run's training sets never touch its own reported sets."""
import json, sys
from pathlib import Path
bad = 0
for splits in sys.argv[1:]:
    root = Path(splits)
    for d in sorted(root.glob("*_*")):
        ds = "_".join(d.name.split("_")[:-1]) if not d.name.startswith(("gtsrb_",)) else d.name.split("_")[0]
        for cand in ["gtsrb", "tt100k_signs", "lisa_signs"]:
            if d.name.startswith(cand): ds = cand; break
        g = lambda n: set(json.load(open(d / f"{ds}_{n}.json"))["indices"])
        rep, held = g("bug_indices"), g("bug_eval_indices")
        tr, val = g("bug_train_indices"), g("bug_val_indices")
        cal, tst = g("clean_calib_indices"), g("clean_test_indices")
        checks = {
            "S_repair n S_held": rep & held,
            "bug_train n S_held": tr & held,
            "bug_val n S_held": val & held,
            "bug_train n bug_val": tr & val,
            "clean_calib n clean_test": cal & tst,
            "clean n bugs": (cal | tst) & (rep | held),
        }
        for name, ov in checks.items():
            if ov:
                bad += 1
                print(f"  LEAK {d.name}: {name} -> {len(ov)} overlapping indices")
print(f"{'FAIL' if bad else 'OK'}: {bad} leaks across {sys.argv[1:]}")
sys.exit(1 if bad else 0)
