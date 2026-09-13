Setting Status
==============

The paper reports 12 dataset x backbone settings, each trained/evaluated under 3 seeds
(101 / 202 / 303) on the v8/protocol-C pipeline (configs/v8_source/).

| Setting | Data shipped | Split manifests shipped | Backbone checkpoint shipped | Repair checkpoints shipped | Final result CSVs shipped |
| --- | --- | --- | --- | --- | --- |
| gtsrb/resnet50 | yes | yes | no (see below) | no (see below) | yes |
| gtsrb/convnext_tiny | yes | yes | no | no | yes |
| gtsrb/vgg16 | yes | yes | no | no | yes |
| gtsrb/densenet121 | yes | yes | no | no | yes |
| tt100k_signs/resnet50 | yes | yes | no | no | yes |
| tt100k_signs/convnext_tiny | yes | yes | no | no | yes |
| tt100k_signs/vgg16 | yes | yes | no | no | yes |
| tt100k_signs/densenet121 | yes | yes | no | no | yes |
| lisa_signs/resnet50 | yes | yes | no | no | yes |
| lisa_signs/convnext_tiny | yes | yes | no | no | yes |
| lisa_signs/vgg16 | yes | yes | no | no | yes |
| lisa_signs/densenet121 | yes | yes | no | no | yes |

Summary: 12/12 settings have shipped raw data + split manifests + final result CSVs (enough for
the GPU-free table-recomputation path in README.md). 0/12 have shipped checkpoint binaries
(backbone or repair) -- these are documented in `artifacts/checkpoints/MANIFEST.md` /
`manifest.json` but not committed to this repository (~2.2 GB of backbone checkpoints alone;
see README.md "Known limitations"). `bash scripts/reproduce_all.sh <dataset> <backbone> <seed>`
regenerates everything from `data/` + `configs/`: the frozen backbone (if its checkpoint is
missing), DynaPatch, and all 6 of the paper's baselines (Arachne, DistrRep, HeadFT, FullFT,
NNPatch, PatchNAS) -- see README.md "Full retrain" for the one disclosed hyperparameter caveat
(HeadFT/FullFT/DistRep-original's training batch size).

Confirm what is actually present on disk with:

```bash
uv run python scripts/check_environment.py
uv run python scripts/validate_assets.py --fail-if-missing   # will report checkpoints MISSING
                                                               # by design, see above
uv run python scripts/check_seed_split_leak.py \
  artifacts/bug_sets/v8_splits_seed101 \
  artifacts/bug_sets/v8_splits_seed202 \
  artifacts/bug_sets/v8_splits_seed303
```
