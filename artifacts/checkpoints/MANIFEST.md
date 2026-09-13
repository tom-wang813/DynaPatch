# Checkpoint Manifest

Human-readable inventory of the DynaPatch reproduction checkpoints. Machine-readable form: `manifest.json`.

- Basis: **v8 (corrected shuffled split, multi-seed 101/202/303; supersedes retracted repairbench_v7)**
- Objective: `b0_residualln_bugval_smooth_cleanreplay_patchl2`
- Seeds: [101, 202, 303]
- Generated: 2026-07-19

## Frozen Backbones (12)

| Dataset | Backbone | Size | Path |
|---|---|---|---|
| gtsrb | resnet50 | 95 MB | `outputs/exp_gtsrb_resnet50_backbone_public_v2/checkpoints/backbone_last.pt` |
| gtsrb | convnext_tiny | 111 MB | `outputs/exp_gtsrb_convnext_tiny_backbone_public_v2/checkpoints/backbone_last.pt` |
| gtsrb | densenet121 | 29 MB | `outputs/exp_gtsrb_densenet121_backbone_public_v2/checkpoints/backbone_last.pt` |
| gtsrb | vgg16 | 538 MB | `outputs/exp_gtsrb_vgg16_backbone_public_v2/checkpoints/backbone_last.pt` |
| tt100k_signs | resnet50 | 96 MB | `outputs/exp_tt100k_signs_resnet50_backbone_public_v7/checkpoints/backbone_last.pt` |
| tt100k_signs | convnext_tiny | 112 MB | `outputs/exp_tt100k_signs_convnext_tiny_backbone_public_v7/checkpoints/backbone_last.pt` |
| tt100k_signs | densenet121 | 29 MB | `outputs/exp_tt100k_signs_densenet121_backbone_public_v7/checkpoints/backbone_last.pt` |
| tt100k_signs | vgg16 | 540 MB | `outputs/exp_tt100k_signs_vgg16_backbone_public_v7/checkpoints/backbone_last.pt` |
| lisa_signs | resnet50 | 95 MB | `outputs/exp_lisa_signs_resnet50_backbone_public_v7a/checkpoints/backbone_last.pt` |
| lisa_signs | convnext_tiny | 112 MB | `outputs/exp_lisa_signs_convnext_tiny_backbone_public_v7a/checkpoints/backbone_last.pt` |
| lisa_signs | densenet121 | 29 MB | `outputs/exp_lisa_signs_densenet121_backbone_public_v1/checkpoints/backbone_last.pt` |
| lisa_signs | vgg16 | 538 MB | `outputs/exp_lisa_signs_vgg16_backbone_public_v1/checkpoints/backbone_last.pt` |

## Repair Checkpoints (36 = 12 settings × 3 seeds)

| Dataset | Backbone | Seed | Size | Path |
|---|---|---|---|---|
| gtsrb | resnet50 | 101 | 13 MB | `outputs/repairbench_v8_s101/gtsrb/resnet50/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | convnext_tiny | 101 | 7 MB | `outputs/repairbench_v8_s101/gtsrb/convnext_tiny/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | densenet121 | 101 | 13 MB | `outputs/repairbench_v8_s101/gtsrb/densenet121/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | vgg16 | 101 | 7 MB | `outputs/repairbench_v8_s101/gtsrb/vgg16/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | resnet50 | 101 | 13 MB | `outputs/repairbench_v8_s101/tt100k_signs/resnet50/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | convnext_tiny | 101 | 7 MB | `outputs/repairbench_v8_s101/tt100k_signs/convnext_tiny/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | densenet121 | 101 | 8 MB | `outputs/repairbench_v8_s101/tt100k_signs/densenet121/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | vgg16 | 101 | 7 MB | `outputs/repairbench_v8_s101/tt100k_signs/vgg16/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | resnet50 | 101 | 13 MB | `outputs/repairbench_v8_s101/lisa_signs/resnet50/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | convnext_tiny | 101 | 7 MB | `outputs/repairbench_v8_s101/lisa_signs/convnext_tiny/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | densenet121 | 101 | 13 MB | `outputs/repairbench_v8_s101/lisa_signs/densenet121/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | vgg16 | 101 | 7 MB | `outputs/repairbench_v8_s101/lisa_signs/vgg16/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | resnet50 | 202 | 13 MB | `outputs/repairbench_v8_s202/gtsrb/resnet50/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | convnext_tiny | 202 | 7 MB | `outputs/repairbench_v8_s202/gtsrb/convnext_tiny/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | densenet121 | 202 | 13 MB | `outputs/repairbench_v8_s202/gtsrb/densenet121/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | vgg16 | 202 | 7 MB | `outputs/repairbench_v8_s202/gtsrb/vgg16/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | resnet50 | 202 | 13 MB | `outputs/repairbench_v8_s202/tt100k_signs/resnet50/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | convnext_tiny | 202 | 7 MB | `outputs/repairbench_v8_s202/tt100k_signs/convnext_tiny/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | densenet121 | 202 | 8 MB | `outputs/repairbench_v8_s202/tt100k_signs/densenet121/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | vgg16 | 202 | 7 MB | `outputs/repairbench_v8_s202/tt100k_signs/vgg16/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | resnet50 | 202 | 13 MB | `outputs/repairbench_v8_s202/lisa_signs/resnet50/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | convnext_tiny | 202 | 7 MB | `outputs/repairbench_v8_s202/lisa_signs/convnext_tiny/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | densenet121 | 202 | 13 MB | `outputs/repairbench_v8_s202/lisa_signs/densenet121/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | vgg16 | 202 | 7 MB | `outputs/repairbench_v8_s202/lisa_signs/vgg16/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | resnet50 | 303 | 13 MB | `outputs/repairbench_v8_s303/gtsrb/resnet50/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | convnext_tiny | 303 | 7 MB | `outputs/repairbench_v8_s303/gtsrb/convnext_tiny/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | densenet121 | 303 | 13 MB | `outputs/repairbench_v8_s303/gtsrb/densenet121/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| gtsrb | vgg16 | 303 | 7 MB | `outputs/repairbench_v8_s303/gtsrb/vgg16/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | resnet50 | 303 | 13 MB | `outputs/repairbench_v8_s303/tt100k_signs/resnet50/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | convnext_tiny | 303 | 7 MB | `outputs/repairbench_v8_s303/tt100k_signs/convnext_tiny/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | densenet121 | 303 | 8 MB | `outputs/repairbench_v8_s303/tt100k_signs/densenet121/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| tt100k_signs | vgg16 | 303 | 7 MB | `outputs/repairbench_v8_s303/tt100k_signs/vgg16/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | resnet50 | 303 | 13 MB | `outputs/repairbench_v8_s303/lisa_signs/resnet50/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | convnext_tiny | 303 | 7 MB | `outputs/repairbench_v8_s303/lisa_signs/convnext_tiny/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | densenet121 | 303 | 13 MB | `outputs/repairbench_v8_s303/lisa_signs/densenet121/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |
| lisa_signs | vgg16 | 303 | 7 MB | `outputs/repairbench_v8_s303/lisa_signs/vgg16/b0_residualln_bugval_smooth_cleanreplay_patchl2/train/checkpoints/repair_best.pt` |

## Companion assets per repair checkpoint

- resolved config (`config_resolved.yaml`), metrics (`metrics.json`)
- split manifests (`bug_bank/`), deploy prediction CSVs (`deploy_direct/predictions/`)
- per-seed splits under `artifacts/bug_sets/v8_splits_seed{seed}/`

## Reproduction

- GPU-free tables: `scripts/summarize_rq4_v8.py --seeds 101 202 303`
- split leakage check: `scripts/check_seed_split_leak.py artifacts/bug_sets/v8_splits_seed*`
- retrain (GPU): `scripts/run_repairbench_v8.sh`
