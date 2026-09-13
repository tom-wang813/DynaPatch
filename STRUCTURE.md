Repository Structure
====================

dynapatch-anon-repro/
  README.md            - start here: what's GPU-free vs full-retrain, and what's NOT included
  ENVIRONMENT.md        - setup + dependency notes
  STRUCTURE.md          - this file
  PAPER_SETTING_STATUS.md
  LICENSE
  pyproject.toml

  src/                          - method implementation
    models/dynapatch/           - DPGen hypernetwork, patch operator, router, deployment
                                  gate/policy, prototype + repair banks
    experiment/                 - training loop (train_stage3.py), deploy-time eval
                                  (deploy_eval.py), shared bundle builder (stage3.py)
    baselines/                  - arachne_de.py (real Arachne, our re-impl., DE search),
                                  distrep_pso.py (real DistRep, our re-impl., PSO search),
                                  head_repair.py (HeadFT/FullFT/DistRep-original config logic),
                                  arachne.py (TopKSearch config logic -- NOT real Arachne, see
                                  scripts/run_fewshot_arachne.sh)
    data/, training/, evaluation/, utils/

  scripts/
    paper_latex_tables.py       - GPU-free: LaTeX tables from outputs/*.csv,*.json (entrypoint)
    names.py                    - shared setting/method-name/column constants
    reproduce_all.sh            - full-retrain: backbone + DynaPatch + all 6 baselines,
                                  one (dataset, backbone, seed) at a time (entrypoint)
    train_backbone.py           - Hydra-composed frozen-backbone trainer
    train_head_repair_baseline.py - HeadFT / FullFT / DistRep-original (--mode flag selects)
    train_arachne_baseline.py   - TopKSearch (NOT real Arachne, see its own driver below)
    prepare_lisa_classification.py, prepare_tt100k_classification.py
                                 - raw annotations -> ImageFolder classification crops
    run_resolved_experiment.py  - config-driven training/deploy-eval runner (DynaPatch)
    run_arachne_de.py, run_distrep_pso.py       - real-baseline entrypoints (self-contained)
    run_mainline_all.sh, queue_mainline_baselines.sh,
    queue_arachne_de.sh, queue_distrep_pso.sh,
    run_fewshot_{arachne,distrep,headrepair}.sh,
    queue_real_baselines.sh      - orchestration across 12 settings x 3 seeds (all self-contained)
    check_environment.py, validate_assets.py, check_seed_split_leak.py   - sanity checks
    sample_frame.py, collect_csv.py, best_data.py, table_rq4_final.py,
    gate_protocol_b.py, gate_zoo.py, analysis_*.py, analyze_*.py    - analysis pipeline that
                                  PRODUCED the shipped outputs/*.csv (kept for methodological
                                  transparency; most need large raw dumps not shipped here to
                                  actually re-run, see README)
    backbone_registry.py         - (dataset, backbone) -> checkpoint dir, used by
                                  dump_prior_features.py (NN-Patching/PatchNAS feature cache)

  configs/
    config.yaml, dataset/, model/, train/, runtime/   - Hydra composition groups for
                                  train_backbone.py only (dataset={gtsrb,tt100k_signs,
                                  lisa_signs}, model={resnet50,convnext,densenet121,vgg16},
                                  train=backbone_finetune, runtime={local_cpu,local_gpu})
    v8_source/<dataset>/<backbone>/{train,deploy}.yaml   - the REAL, live per-setting configs
                                                           for DynaPatch + all baselines except
                                                           backbone training (confirmed: every
                                                           queue/run script reads this tree)
    risk/                        - critical-class definitions (per dataset)
    analysis/                    - configs for a subset of the analysis scripts above
    matrix/traffic_sign_12_settings.csv   - the (dataset, backbone) setting matrix

  artifacts/
    checkpoints/                 - manifest.json + MANIFEST.md (inventory only; NO checkpoint
                                  binaries shipped, ~2.2 GB, see README)
    bug_sets/v8_splits_seed{101,202,303}/   - per-seed split manifests (JSON, ~4 MB total)
    risk/                        - *_safety_risk_matrix.json, paired with configs/risk/

  outputs/                       - ONLY the small (few-MB) final per-cell/per-setting CSV/JSON
                                  files that scripts/paper_latex_tables.py reads directly.
                                  NOT the raw per-sample prediction dumps (tens-150 GB in the
                                  original working repo); see README "Known limitations".

  data/                          - GTSRB / TT100K-Signs / LISA-Signs (traffic-sign
                                  classification images), ~680 MB total

What was deliberately left out of this anonymized repo (and why)
------------------------------------------------------------------

- note/, sanner_2026/ -- internal, largely Chinese-language research process logs
  (session-by-session pitfalls/decisions). Not meant for a public artifact; superseded for the
  purpose of this repo by this file + README.md + ENVIRONMENT.md.

- ~20 scripts from an abandoned "certified non-regression" research direction
  (verify_ball_contracts.py, attack_ball_contracts.py, analyze_certified.py,
  analyze_repairability_bound.py, rq1_phase0.py, make_main_table.py, make_figure1.py,
  and others) -- confirmed via grep that paper.tex never mentions this direction (its one
  nearby hit, "formal verification", describes a rival method, PatchPro, in related work, not
  anything DynaPatch does). These also hardcoded absolute paths to a separate, non-anonymized
  sibling repository.

- configs/paper/*, configs/experiment/* -- an earlier, never-executed "public workflow"
  config scaffold. Confirmed via grep that zero scripts in this codebase actually read
  configs/paper/*; its example configs also point at bug-index manifests and checkpoint
  generations that do not exist on disk. configs/v8_source/* is the real, live tree (27
  scripts reference it) and is what this repo ships instead.

- The bulk of outputs/ (173 GB in the original working repo: raw per-sample prediction
  dumps, dozens of hyperparameter-ablation trees, feature caches) -- see README "Known
  limitations". Only the small final CSVs/JSONs paper_latex_tables.py reads are shipped.

- Frozen backbone checkpoints and repair checkpoints (binaries, ~2.2 GB+) -- not committed to
  this git-based artifact; artifacts/checkpoints/MANIFEST.md documents what they are.

- main.py -- a generic Hydra dispatcher for experiment.stage in {stage3_repair, deploy_eval}.
  Superseded by scripts/run_resolved_experiment.py, which does the same dispatch from a flat
  --config path instead of full Hydra composition and is what every queue/run script actually
  calls; confirmed never invoked by any script, removed rather than shipped as a second,
  redundant entrypoint. (Its Hydra config group, configs/config.yaml, IS shipped -- it is
  needed separately by scripts/train_backbone.py, which is unrelated to main.py's two stages.)

- src/baselines/selective.py -- confirmed not imported by any script in this codebase, kept
  anyway (harmless, no identifying content). src/baselines/arachne.py IS load-bearing here:
  scripts/train_arachne_baseline.py (the "TopKSearch" baseline, see its own header for why it
  must never be called "Arachne") imports it via src/baselines/__init__.py.
