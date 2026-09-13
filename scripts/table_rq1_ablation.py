#!/usr/bin/env python3
"""RQ1 ablation table under the CANONICAL per_setting.py metric definitions."""
import csv, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DSS = ["gtsrb", "tt100k_signs", "lisa_signs"]
BBS = ["resnet50", "convnext_tiny", "densenet121", "vgg16"]
SEEDS = [101, 202, 303]
ARMS = ["ep40", "fixedpatch40"]
_tf = lambda x: str(x).lower() == "true"

def crit(ds):
    cfg = json.loads((ROOT / f"artifacts/risk/{ds}_safety_risk_matrix.json").read_text())
    out = set()
    for ids in cfg["critical_signs"].values():
        out.update(int(i) for i in ids)
    return out

def cell(arm, ds, bb, s, cr):
    run = ROOT / f"outputs/patch_ablation/{ds}_{bb}/{arm}_s{s}_kfull"
    pd = run / "deploy" / "predictions"
    dm = run / "deploy" / "metrics.json"
    ci = json.loads(dm.read_text()).get("checkpoint_load_info", {}) if dm.exists() else {}
    dropped = len(ci.get("skipped_keys", [])) + len(ci.get("missing_keys", []))
    fs = [pd / f for f in ("repair_support_seen_predictions.csv",
                           "repair_holdout_unseen_predictions.csv",
                           "clean_eval_predictions.csv")]
    if not all(f.exists() for f in fs):
        return None
    sn, hd, cl = (list(csv.DictReader(f.open())) for f in fs)
    f = ROOT / f"artifacts/bug_sets/v8_splits_seed{s}/{ds}_{bb}/{ds}_clean_test_indices.json"
    idx = set(json.loads(f.read_text())["indices"])
    cl = [r for r in cl if int(r["dataset_index"]) in idx]
    cc = [r for r in cl if int(r["label"]) in cr]
    tm = json.loads((run / "train" / "metrics.json").read_text())
    return {
        "seen": sum(_tf(r["patched_correct"]) for r in sn) / max(len(sn), 1),
        "held": sum(_tf(r["patched_correct"]) for r in hd) / max(len(hd), 1),
        "reg":  sum(1 for r in cl if _tf(r["base_correct"]) and not _tf(r["patched_correct"])) / max(len(cl), 1),
        "creg": sum(1 for r in cc if _tf(r["base_correct"]) and not _tf(r["patched_correct"])) / max(len(cc), 1),
        "ep": len(tm.get("history", [])), "best_ep": tm.get("best_epoch"),
        "drop": dropped, "n_seen": len(sn), "n_held": len(hd), "n_clean": len(cl), "n_crit": len(cc),
    }

rows = {}
for ds in DSS:
    cr = crit(ds)
    for bb in BBS:
        for arm in ARMS:
            cs = [c for s in SEEDS if (c := cell(arm, ds, bb, s, cr))]
            rows[(ds, bb, arm)] = cs

print(f"{'setting':30}{'arm':14}{'RR_seen':>16}{'RR_held':>16}{'Reg':>18}{'CReg':>18}{'ep':>10}{'drop':>6}{'n_seen/n_held':>16}")
for ds in DSS:
    for bb in BBS:
        for arm in ARMS:
            cs = rows[(ds, bb, arm)]
            if not cs:
                print(f"{ds+'/'+bb:30}{arm:14}  MISSING"); continue
            f = lambda k, p=3: (f"{st.mean(c[k] for c in cs):.{p}f}"
                                + (f"±{st.stdev([c[k] for c in cs]):.{p}f}" if len(cs) > 1 else ""))
            eps = "/".join(str(c["ep"]) for c in cs)
            print(f"{ds+'/'+bb:30}{arm:14}{f('seen'):>16}{f('held'):>16}{f('reg',4):>18}{f('creg',4):>18}"
                  f"{eps:>10}{sum(c['drop'] for c in cs):>6}{str(cs[0]['n_seen'])+'/'+str(cs[0]['n_held']):>16}")

print("\n-- deltas (mean over seeds) --")
print(f"{'setting':30}{'dRR_held':>12}{'dRR_seen':>12}{'dReg':>12}{'dCReg':>12}")
d_held = []
for ds in DSS:
    for bb in BBS:
        a, b = rows[(ds, bb, "ep40")], rows[(ds, bb, "fixedpatch40")]
        if not a or not b: continue
        m = lambda cs, k: st.mean(c[k] for c in cs)
        dh = m(a,"held")-m(b,"held"); d_held.append(dh)
        print(f"{ds+'/'+bb:30}{dh:>+12.3f}{m(a,'seen')-m(b,'seen'):>+12.3f}"
              f"{m(a,'reg')-m(b,'reg'):>+12.4f}{m(a,'creg')-m(b,'creg'):>+12.4f}")
print(f"{'median dRR_held':30}{st.median(d_held):>+12.3f}   wins {sum(1 for x in d_held if x>0)}/{len(d_held)}")

print("\n-- fixedpatch per-seed epoch/ceiling audit --")
print(f"{'setting':30}{'seed':>6}{'epochs':>8}{'best_ep':>9}{'RR_seen':>10}{'RR_held':>10}")
mx_s = mx_h = 0
for ds in DSS:
    for bb in BBS:
        for c, s in zip(rows[(ds, bb, "fixedpatch40")], SEEDS):
            mx_s = max(mx_s, c["seen"]); mx_h = max(mx_h, c["held"])
            print(f"{ds+'/'+bb:30}{s:>6}{c['ep']:>8}{str(c['best_ep']):>9}{c['seen']:>10.3f}{c['held']:>10.3f}")
print(f"MAX over all 36 fixedpatch cells: RR_seen {mx_s:.3f}  RR_held {mx_h:.3f}")
