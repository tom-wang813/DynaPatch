# Repository Structure

```
dynapatch-anon-repro/
├── README.md              GPU-free table reproduction, results, pretrained checkpoints
├── TRAINING.md            full retrain from scratch
├── ENVIRONMENT.md         setup + dependency notes
├── STRUCTURE.md           this file
├── LICENSE
├── pyproject.toml
│
├── src/                   method implementation
│   ├── models/dynapatch/  DPGen hypernetwork, patch operator, decomposition, DPGate (gate.py), deployment policy
│   ├── experiment/        training loop, deploy-time eval, shared bundle builder (stage3.py)
│   ├── baselines/         arachne_de.py (Arachne), distrep_pso.py (DistRep), head_repair.py (HeadFT/FullFT)
│   └── data/, training/, utils/
│
├── scripts/               see "scripts/ at a glance" below
│
├── configs/
│   ├── config.yaml, dataset/, model/, train/, runtime/   Hydra groups, train_backbone.py only
│   ├── shuffled_split_source/<dataset>/<backbone>/{train,deploy}.yaml   live per-setting configs (DynaPatch + baselines)
│   ├── risk/              critical-class definitions, per dataset
│   └── analysis/          configs for a subset of scripts/analysis_*.py
│
├── artifacts/
│   ├── checkpoints/       manifest.json + MANIFEST.md (checkpoint inventory; no binaries)
│   ├── bug_sets/shuffled_split_seed{101,202,303}/   per-seed split manifests, ~4 MB total
│   └── risk/              *_safety_risk_matrix.json, paired with configs/risk/
│
├── outputs/               rq1/, rq2/, rq3/, rq4/ -- the ~18 CSV/JSON files paper_tables.py's
│                          matching rq{N}() actually reads (one per table/panel in the paper's
│                          own Results section, nothing extra), a few MB total; a file used by
│                          more than one RQ lives under its primary consumer and the other RQ's
│                          function just points at it
│
└── data/                  GTSRB / TT100K-Signs / LISA-Signs images, ~680 MB, not tracked by git
```

## scripts/ at a glance

46 files. Grouped by role, not alphabetically:

| Role | Files | Notes |
|---|---|---|
| Train / deploy entrypoints | `run_resolved_experiment.py`, `train_backbone.py`, `train_head_repair_baseline.py`, `run_arachne_de.py`, `run_distrep_pso.py`, `dump_prior_features.py`, `baseline_prior_patches.py`, `backbone_registry.py` | actually run training/deploy, produce checkpoints and predictions |
| Orchestration | `reproduce_all.sh`, `run_mainline_all.sh`, `run_fewshot_{headrepair,safepatch}.sh`, `queue_*.sh`, `deploy_from_checkpoints.sh` | loop the entrypoints above across settings/seeds |
| Data prep | `prepare_lisa_classification.py`, `prepare_tt100k_classification.py` | raw annotations → ImageFolder crops |
| Sanity checks | `check_environment.py`, `validate_assets.py`, `check_seed_split_leak.py`, `guards.py` | verify environment/assets, no paper output |
| Shared infra | `names.py`, `common.py`, `raw_md.py`, `collect_csv.py`, `hypernet_param_count.py` | imported by other scripts, or standalone utilities not tied to a specific RQ table; `names.py` is the single source of truth for method/setting names, and its own `check_paper_coverage()` self-test is the guard against a result being reported under the wrong name |
| RQ1–RQ4 evidence pipeline | `analysis_patch_reassignment.py`, `analysis_rq1_alignment.py`, `analysis_rq1_margin_decomposition.py`, `analyze_response_gate_lobo.py`, `probe_gonogo_pre_vs_prepost.py`, `gate_zoo.py`, `gate_report.py`, `gate_protocol_b.py`, `gate_lambda_sweep.py`, `extract_rq3_gate_evidence.py`, `all_rq_data.py`, `table_from_checkpoints.py` | each produces one of the ~18 CSV/JSON files under `outputs/rq{1,2,3,4}/` that `paper_tables.py` reads -- this is the audited computation behind every one of the paper's 9 result tables, not extra scope |
| Table rendering | `paper_tables.py` | reads everything above, writes the paper's 9 LaTeX tables |

Every script in the RQ1–RQ4 pipeline row has been checked (both ways: does anything import it, does
anything read its CSV output) to confirm `paper_tables.py`'s dependency chain actually needs
it. This list is intentionally short: an earlier, larger version of this pipeline (~30 scripts)
computed a long tail of exploratory mechanism/ablation tables that trace to RQ1.x/RQ2.x/RQ3.x/
RQ4.x sub-questions along the way to the paper, not to any table the paper itself reports --
those scripts and their CSVs have been removed rather than kept "for reference"; see git history
if a reviewer question needs one of them reconstructed. `gate_protocol_b.py` also does the
raw-data merge (`_load_split`, last_affine patch-site backbones vs the rest -- see
`scripts/gate_lambda_sweep.py`'s own header) other RQ1-4 scripts rely on for a correct DynaPatch
gate refit; `extract_rq3_gate_evidence.py` derives RQ3's classification/deployment tables
directly from two `gate_protocol_b.py` runs (pre-only, pre+post) rather than from a frozen note,
so it stays regenerable end-to-end from `outputs/effect_dump*/`.
