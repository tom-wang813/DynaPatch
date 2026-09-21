# Checkpoint Manifest

Human-readable inventory of the DynaPatch reproduction checkpoints. Machine-readable form: `manifest.json`.

- Basis: **corrected shuffled split, multi-seed 101/202/303 (supersedes retracted repairbench_v7)**
- Objective: `b0_residualln_bugval_smooth_cleanreplay_patchl2`
- Seeds: [101, 202, 303]
- Generated: 2026-07-19
- Layout: all checkpoints moved under `artifacts/checkpoints/` (was split across `outputs/` for backbones/
  DynaPatch and `artifacts/checkpoints/` for baselines only, prior to 2026-09-21) so the whole inventory
  lives in one place, matching this file's own name.

## Frozen Backbones (12)

| Dataset | Backbone | Path |
|---|---|---|
| gtsrb | resnet50 | `artifacts/checkpoints/backbones/gtsrb_resnet50/backbone_last.pt` |
| gtsrb | convnext_tiny | `artifacts/checkpoints/backbones/gtsrb_convnext_tiny/backbone_last.pt` |
| gtsrb | densenet121 | `artifacts/checkpoints/backbones/gtsrb_densenet121/backbone_last.pt` |
| gtsrb | vgg16 | `artifacts/checkpoints/backbones/gtsrb_vgg16/backbone_last.pt` |
| tt100k_signs | resnet50 | `artifacts/checkpoints/backbones/tt100k_signs_resnet50/backbone_last.pt` |
| tt100k_signs | convnext_tiny | `artifacts/checkpoints/backbones/tt100k_signs_convnext_tiny/backbone_last.pt` |
| tt100k_signs | densenet121 | `artifacts/checkpoints/backbones/tt100k_signs_densenet121/backbone_last.pt` |
| tt100k_signs | vgg16 | `artifacts/checkpoints/backbones/tt100k_signs_vgg16/backbone_last.pt` |
| lisa_signs | resnet50 | `artifacts/checkpoints/backbones/lisa_signs_resnet50/backbone_last.pt` |
| lisa_signs | convnext_tiny | `artifacts/checkpoints/backbones/lisa_signs_convnext_tiny/backbone_last.pt` |
| lisa_signs | densenet121 | `artifacts/checkpoints/backbones/lisa_signs_densenet121/backbone_last.pt` |
| lisa_signs | vgg16 | `artifacts/checkpoints/backbones/lisa_signs_vgg16/backbone_last.pt` |

## Repair Checkpoints (36 = 12 settings × 3 seeds)

| Dataset | Backbone | Seed | Path |
|---|---|---|---|
| gtsrb | resnet50 | 101 | `artifacts/checkpoints/dynapatch/gtsrb_resnet50_s101/repair_best.pt` |
| gtsrb | convnext_tiny | 101 | `artifacts/checkpoints/dynapatch/gtsrb_convnext_tiny_s101/repair_best.pt` |
| gtsrb | densenet121 | 101 | `artifacts/checkpoints/dynapatch/gtsrb_densenet121_s101/repair_best.pt` |
| gtsrb | vgg16 | 101 | `artifacts/checkpoints/dynapatch/gtsrb_vgg16_s101/repair_best.pt` |
| tt100k_signs | resnet50 | 101 | `artifacts/checkpoints/dynapatch/tt100k_signs_resnet50_s101/repair_best.pt` |
| tt100k_signs | convnext_tiny | 101 | `artifacts/checkpoints/dynapatch/tt100k_signs_convnext_tiny_s101/repair_best.pt` |
| tt100k_signs | densenet121 | 101 | `artifacts/checkpoints/dynapatch/tt100k_signs_densenet121_s101/repair_best.pt` |
| tt100k_signs | vgg16 | 101 | `artifacts/checkpoints/dynapatch/tt100k_signs_vgg16_s101/repair_best.pt` |
| lisa_signs | resnet50 | 101 | `artifacts/checkpoints/dynapatch/lisa_signs_resnet50_s101/repair_best.pt` |
| lisa_signs | convnext_tiny | 101 | `artifacts/checkpoints/dynapatch/lisa_signs_convnext_tiny_s101/repair_best.pt` |
| lisa_signs | densenet121 | 101 | `artifacts/checkpoints/dynapatch/lisa_signs_densenet121_s101/repair_best.pt` |
| lisa_signs | vgg16 | 101 | `artifacts/checkpoints/dynapatch/lisa_signs_vgg16_s101/repair_best.pt` |
| gtsrb | resnet50 | 202 | `artifacts/checkpoints/dynapatch/gtsrb_resnet50_s202/repair_best.pt` |
| gtsrb | convnext_tiny | 202 | `artifacts/checkpoints/dynapatch/gtsrb_convnext_tiny_s202/repair_best.pt` |
| gtsrb | densenet121 | 202 | `artifacts/checkpoints/dynapatch/gtsrb_densenet121_s202/repair_best.pt` |
| gtsrb | vgg16 | 202 | `artifacts/checkpoints/dynapatch/gtsrb_vgg16_s202/repair_best.pt` |
| tt100k_signs | resnet50 | 202 | `artifacts/checkpoints/dynapatch/tt100k_signs_resnet50_s202/repair_best.pt` |
| tt100k_signs | convnext_tiny | 202 | `artifacts/checkpoints/dynapatch/tt100k_signs_convnext_tiny_s202/repair_best.pt` |
| tt100k_signs | densenet121 | 202 | `artifacts/checkpoints/dynapatch/tt100k_signs_densenet121_s202/repair_best.pt` |
| tt100k_signs | vgg16 | 202 | `artifacts/checkpoints/dynapatch/tt100k_signs_vgg16_s202/repair_best.pt` |
| lisa_signs | resnet50 | 202 | `artifacts/checkpoints/dynapatch/lisa_signs_resnet50_s202/repair_best.pt` |
| lisa_signs | convnext_tiny | 202 | `artifacts/checkpoints/dynapatch/lisa_signs_convnext_tiny_s202/repair_best.pt` |
| lisa_signs | densenet121 | 202 | `artifacts/checkpoints/dynapatch/lisa_signs_densenet121_s202/repair_best.pt` |
| lisa_signs | vgg16 | 202 | `artifacts/checkpoints/dynapatch/lisa_signs_vgg16_s202/repair_best.pt` |
| gtsrb | resnet50 | 303 | `artifacts/checkpoints/dynapatch/gtsrb_resnet50_s303/repair_best.pt` |
| gtsrb | convnext_tiny | 303 | `artifacts/checkpoints/dynapatch/gtsrb_convnext_tiny_s303/repair_best.pt` |
| gtsrb | densenet121 | 303 | `artifacts/checkpoints/dynapatch/gtsrb_densenet121_s303/repair_best.pt` |
| gtsrb | vgg16 | 303 | `artifacts/checkpoints/dynapatch/gtsrb_vgg16_s303/repair_best.pt` |
| tt100k_signs | resnet50 | 303 | `artifacts/checkpoints/dynapatch/tt100k_signs_resnet50_s303/repair_best.pt` |
| tt100k_signs | convnext_tiny | 303 | `artifacts/checkpoints/dynapatch/tt100k_signs_convnext_tiny_s303/repair_best.pt` |
| tt100k_signs | densenet121 | 303 | `artifacts/checkpoints/dynapatch/tt100k_signs_densenet121_s303/repair_best.pt` |
| tt100k_signs | vgg16 | 303 | `artifacts/checkpoints/dynapatch/tt100k_signs_vgg16_s303/repair_best.pt` |
| lisa_signs | resnet50 | 303 | `artifacts/checkpoints/dynapatch/lisa_signs_resnet50_s303/repair_best.pt` |
| lisa_signs | convnext_tiny | 303 | `artifacts/checkpoints/dynapatch/lisa_signs_convnext_tiny_s303/repair_best.pt` |
| lisa_signs | densenet121 | 303 | `artifacts/checkpoints/dynapatch/lisa_signs_densenet121_s303/repair_best.pt` |
| lisa_signs | vgg16 | 303 | `artifacts/checkpoints/dynapatch/lisa_signs_vgg16_s303/repair_best.pt` |

## Gate Checkpoints (36 = 12 settings × 3 seeds)

`src/models/dynapatch/gate.py`'s `FeatureGate`: the paper's 9-feature (3 pre-info + 6 post-info)
multinomial logistic regression, fit on `bug_train + bug_val` (failure side) + `clean_calib` (clean
side) -- the "both" protocol -- from the `ep40ns` effect_dump. This is the gate that actually decides
commit-or-rollback; `deployment_policy.py`'s `GatedDirectPolicy` (the one `deploy.yaml`'s literal
`policy: direct_generalization` resolves to) never rejects anything in practice, since every setting
ships `deployment.gate_threshold: 9999.0` -- see the `dynapatch-deploy-policy-promotion-quirk` memory
note. Load one with `deployment.feature_gate_path=<path>` (`deployment.feature_gate_lambda` sets the
regression-cost λ, default 1.0).

| Dataset | Backbone | Seed | Path |
|---|---|---|---|
| gtsrb | resnet50 | 101 | `artifacts/checkpoints/gates/gtsrb_resnet50_s101.json` |
| gtsrb | convnext_tiny | 101 | `artifacts/checkpoints/gates/gtsrb_convnext_tiny_s101.json` |
| gtsrb | densenet121 | 101 | `artifacts/checkpoints/gates/gtsrb_densenet121_s101.json` |
| gtsrb | vgg16 | 101 | `artifacts/checkpoints/gates/gtsrb_vgg16_s101.json` |
| tt100k_signs | resnet50 | 101 | `artifacts/checkpoints/gates/tt100k_signs_resnet50_s101.json` |
| tt100k_signs | convnext_tiny | 101 | `artifacts/checkpoints/gates/tt100k_signs_convnext_tiny_s101.json` |
| tt100k_signs | densenet121 | 101 | `artifacts/checkpoints/gates/tt100k_signs_densenet121_s101.json` |
| tt100k_signs | vgg16 | 101 | `artifacts/checkpoints/gates/tt100k_signs_vgg16_s101.json` |
| lisa_signs | resnet50 | 101 | `artifacts/checkpoints/gates/lisa_signs_resnet50_s101.json` |
| lisa_signs | convnext_tiny | 101 | `artifacts/checkpoints/gates/lisa_signs_convnext_tiny_s101.json` |
| lisa_signs | densenet121 | 101 | `artifacts/checkpoints/gates/lisa_signs_densenet121_s101.json` |
| lisa_signs | vgg16 | 101 | `artifacts/checkpoints/gates/lisa_signs_vgg16_s101.json` |
| gtsrb | resnet50 | 202 | `artifacts/checkpoints/gates/gtsrb_resnet50_s202.json` |
| gtsrb | convnext_tiny | 202 | `artifacts/checkpoints/gates/gtsrb_convnext_tiny_s202.json` |
| gtsrb | densenet121 | 202 | `artifacts/checkpoints/gates/gtsrb_densenet121_s202.json` |
| gtsrb | vgg16 | 202 | `artifacts/checkpoints/gates/gtsrb_vgg16_s202.json` |
| tt100k_signs | resnet50 | 202 | `artifacts/checkpoints/gates/tt100k_signs_resnet50_s202.json` |
| tt100k_signs | convnext_tiny | 202 | `artifacts/checkpoints/gates/tt100k_signs_convnext_tiny_s202.json` |
| tt100k_signs | densenet121 | 202 | `artifacts/checkpoints/gates/tt100k_signs_densenet121_s202.json` |
| tt100k_signs | vgg16 | 202 | `artifacts/checkpoints/gates/tt100k_signs_vgg16_s202.json` |
| lisa_signs | resnet50 | 202 | `artifacts/checkpoints/gates/lisa_signs_resnet50_s202.json` |
| lisa_signs | convnext_tiny | 202 | `artifacts/checkpoints/gates/lisa_signs_convnext_tiny_s202.json` |
| lisa_signs | densenet121 | 202 | `artifacts/checkpoints/gates/lisa_signs_densenet121_s202.json` |
| lisa_signs | vgg16 | 202 | `artifacts/checkpoints/gates/lisa_signs_vgg16_s202.json` |
| gtsrb | resnet50 | 303 | `artifacts/checkpoints/gates/gtsrb_resnet50_s303.json` |
| gtsrb | convnext_tiny | 303 | `artifacts/checkpoints/gates/gtsrb_convnext_tiny_s303.json` |
| gtsrb | densenet121 | 303 | `artifacts/checkpoints/gates/gtsrb_densenet121_s303.json` |
| gtsrb | vgg16 | 303 | `artifacts/checkpoints/gates/gtsrb_vgg16_s303.json` |
| tt100k_signs | resnet50 | 303 | `artifacts/checkpoints/gates/tt100k_signs_resnet50_s303.json` |
| tt100k_signs | convnext_tiny | 303 | `artifacts/checkpoints/gates/tt100k_signs_convnext_tiny_s303.json` |
| tt100k_signs | densenet121 | 303 | `artifacts/checkpoints/gates/tt100k_signs_densenet121_s303.json` |
| tt100k_signs | vgg16 | 303 | `artifacts/checkpoints/gates/tt100k_signs_vgg16_s303.json` |
| lisa_signs | resnet50 | 303 | `artifacts/checkpoints/gates/lisa_signs_resnet50_s303.json` |
| lisa_signs | convnext_tiny | 303 | `artifacts/checkpoints/gates/lisa_signs_convnext_tiny_s303.json` |
| lisa_signs | densenet121 | 303 | `artifacts/checkpoints/gates/lisa_signs_densenet121_s303.json` |
| lisa_signs | vgg16 | 303 | `artifacts/checkpoints/gates/lisa_signs_vgg16_s303.json` |

## Baseline Checkpoints (72 = 6 methods × 12 settings × seed 101)

Each of the 6 paper baselines (HeadFT, FullFT, Arachne, DistrRep, NNPatch, PatchNAS) has one
checkpoint set per setting at seed 101, produced by `scripts/reproduce_all.sh` (HeadFT, FullFT,
Arachne, DistrRep) and `scripts/baseline_prior_patches.py --checkpoint-dir` (NNPatch, PatchNAS),
kept in a clean per-method tree separate from the noisier per-run `outputs/` training directory
(which stays local/regenerable):

```
artifacts/checkpoints/baselines/<Method>/<dataset>_<backbone>_s101/<file>.pt
```

| Method | File(s) | What it is |
|---|---|---|
| Arachne | `arachne_patched_classifier.pt` | patched classifier-head weight + bias (DE search result) |
| DistrRep | `distrep_repaired.pt` | full repaired backbone state_dict (3-phase PSO merge) |
| HeadFT | `baseline_last.pt` | classifier head after head-only fine-tuning |
| FullFT | `baseline_last.pt` | full model after full fine-tuning |
| NNPatch | `patch_head.pt`, `estimator_head.pt` | patch head + error estimator (penultimate-feature) |
| PatchNAS | `patch_head.pt`, `estimator_head.pt` | searched lightweight patch head + error estimator |

Only seed 101 is produced (matching DistRep(PSO)'s own paper-reported single-seed budget; the
other baselines could be extended to seeds 202/303 the same way, just not done here). See
`manifest.json`'s `baseline_checkpoints` list for the full 72-entry inventory.

## Companion assets

- per-seed splits under `artifacts/bug_sets/shuffled_split_seed{seed}/`
- only the checkpoint file itself is kept per entry above -- the run's own `config_resolved.yaml`/
  `metrics.json`/`bug_bank/`/prediction CSVs are regenerated fresh by whichever reproduction path
  you run (`reproduce_all.sh` or `deploy_from_checkpoints.sh`), not preserved alongside the
  checkpoint.

## Reproduction

- GPU-free tables: `scripts/paper_tables.py --outdir note/tables`
- split leakage check: `scripts/check_seed_split_leak.py artifacts/bug_sets/shuffled_split_seed*`
- retrain (GPU): `scripts/reproduce_all.sh <dataset> <backbone> <seed>`
- deploy-eval directly from these checkpoints (no retraining): `scripts/deploy_from_checkpoints.sh` + `scripts/table_from_checkpoints.py`
- produce the baseline checkpoints above (GPU): `reproduce_all.sh` (Arachne/HeadFT/FullFT/DistrRep,
  with `DYNAPATCH_CHECKPOINT=<path>` to reuse an existing DynaPatch checkpoint instead of retraining
  it) plus `scripts/dump_prior_features.py` + `scripts/baseline_prior_patches.py --checkpoint-dir`
  (NNPatch/PatchNAS)
