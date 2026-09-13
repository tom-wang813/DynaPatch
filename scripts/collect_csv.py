#!/usr/bin/env python3
"""Collect every analysis CSV into ONE flat directory: outputs/csv/.

No aggregation, no RQ grouping -- the raw per-cell tables, side by side, so a reader can open
them all from one place. Re-running overwrites; the sources stay where their scripts write them.
"""
import csv
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/csv"

# (destination name, source path, one-line description)
FILES = [
    ("sample_frame.csv.gz", "outputs/sample_frame.csv.gz",
     "the base table everything else is derived from: one row per input, per method, per setting, "
     "per seed, with base and patched predictions side by side"),
    ("audit_coverage.csv", "outputs/analysis_audit.csv",
     "sample counts and event counts for every (method, setting, seed, split)"),
    ("per_class_repair_and_regression.csv", "outputs/p1_regression/per_cell.csv",
     "per-class repair rate, class coverage, count-matched concentration null, and per-class "
     "regression, one row per (method, setting, seed)"),
    ("per_class_repair_and_regression__by_setting.csv", "outputs/p1_regression/per_setting.csv",
     "the same, seeds collapsed"),
    ("inputspecific_per_cell.csv", "outputs/p2_inputspecific/per_cell.csv",
     "FixedPatch vs DynaPatch-NoGate, one row per (method, setting, seed, split)"),
    ("inputspecific_paired.csv", "outputs/p2_inputspecific/paired.csv",
     "the same two arms paired on the same (setting, seed, split)"),
    ("inputspecific_per_class.csv", "outputs/p2_inputspecific/per_class.csv",
     "both arms' repair rate for the same true class, with the stratum set by the FIXED arm"),
    ("postinfo_per_outcome.csv", "outputs/p3_postinfo/per_outcome.csv",
     "apply rate per outcome quadrant, per evidence arm, per r"),
    ("postinfo_br_hr.csv", "outputs/p3_postinfo/br_hr.csv",
     "the same rows pivoted: one row per (arm, r, setting, seed) with BR and HR"),
    ("patchvec_geometry.csv", "outputs/p4_patchvec/per_cell.csv",
     "within-class and between-class distance of the generated patch vectors, with a permutation "
     "null"),
    ("m1_direction_per_cell.csv", "outputs/m1_correction_direction/direction_per_cell.csv",
     "within/between-failure-type cosine and L2 distance of Delta z(x) = patched_logits - "
     "base_logits, with a permutation null, for NN-Patching / PatchNAS / DynaPatch-NoGate side "
     "by side"),
    ("m1_magnitude_margin_per_cell.csv",
     "outputs/m1_correction_direction/magnitude_margin_per_cell.csv",
     "correction magnitude ||Delta z(x)|| and true-vs-wrong margin improvement Delta m(x), same "
     "three methods; NOT comparable in absolute units across methods, see note/ANALYSIS_RAW.md M1"),
    ("m1_cluster_alignment_per_cell.csv",
     "outputs/m1_correction_direction/cluster_alignment_per_cell.csv",
     "spherical k-means on unit Delta z(x), k = number of observed failure types, scored against "
     "the true failure-type labels by purity/ARI/NMI"),
    ("m1_aim_vs_magnitude_per_cell.csv",
     "outputs/m1_correction_direction/aim_vs_magnitude_per_cell.csv",
     "each method's own failures split into repaired / aimed-but-not-repaired / misaimed; "
     "repaired here uses each method's RAW un-routed patch output, not the routed RR_held in "
     "note/RQ2_BASELINE_BEHAVIOR.md -- see note/ANALYSIS_RAW.md M1c"),
    ("m1_routed_aim_breakdown_per_cell.csv",
     "outputs/m1_correction_direction/routed_aim_breakdown_per_cell.csv",
     "NN-Patching/PatchNAS held-out failures split by routing and aim at the tau operating "
     "point (matches note/RQ2_DATA.md RQ2.10) -- see note/ANALYSIS_RAW.md M1d"),
    ("gate_transplant_pre_post_per_cell.csv",
     "outputs/gate_transplant/per_cell_4_pre+post.csv",
     "NN-Patching/PatchNAS's raw patch output routed by DynaPatch's own gate (full pre+post "
     "feature set) instead of their own estimator -- see note/ANALYSIS_RAW.md GateTransplant_4_pre+post"),
    ("gate_transplant_pre_only_per_cell.csv",
     "outputs/gate_transplant/per_cell_2_prestrong.csv",
     "the same gate transplant, pre-information features only -- RQ3 robustness check: does "
     "post-info help NN-Patching/PatchNAS's gate too? -- see note/ANALYSIS_RAW.md GateTransplant_2_prestrong"),
    ("m3_distrep_expert_merge_per_failure.csv",
     "outputs/m3_distrep_expert_merge/per_failure.csv",
     "DistRep pre-integration experts vs merged model: correction direction/magnitude per "
     "held-out failure, gtsrb/resnet50 s101 only -- see note/ANALYSIS_RAW.md M3"),
    ("mechanism_failure_type_per_type.csv",
     "outputs/mechanism_profiles/failure_type_per_type.csv",
     "ordered-pair failure-type repair counts and rates for every method/setting/seed/type"),
    ("mechanism_failure_type_per_cell.csv",
     "outputs/mechanism_profiles/failure_type_summary_per_cell.csv",
     "failure-type repair summaries before seed aggregation"),
    ("mechanism_failure_type_per_setting.csv",
     "outputs/mechanism_profiles/failure_type_summary_per_setting.csv",
     "failure-type repair summaries for all 12 settings"),
    ("mechanism_failure_type_overall.csv",
     "outputs/mechanism_profiles/failure_type_summary_overall.csv",
     "setting-balanced failure-type repair summaries"),
    ("mechanism_regression_per_cell.csv",
     "outputs/mechanism_profiles/regression_profile_per_cell.csv",
     "clean-test class-level regression distribution before seed aggregation"),
    ("mechanism_regression_per_setting.csv",
     "outputs/mechanism_profiles/regression_profile_per_setting.csv",
     "clean-test class-level regression distribution for all 12 settings"),
    ("mechanism_regression_overall.csv",
     "outputs/mechanism_profiles/regression_profile_overall.csv",
     "setting-balanced clean-test regression distribution"),
    ("mechanism_postinfo_outcome_per_cell.csv",
     "outputs/mechanism_profiles/postinfo_outcome_per_cell.csv",
     "gate application rates by patch outcome, arm, operating point, setting and seed"),
    ("mechanism_postinfo_outcome_per_setting.csv",
     "outputs/mechanism_profiles/postinfo_outcome_per_setting.csv",
     "gate application rates by patch outcome for all 12 settings"),
    ("mechanism_postinfo_br_hr_per_cell.csv",
     "outputs/mechanism_profiles/postinfo_br_hr_per_cell.csv",
     "beneficial and harmful gate-application rates before seed aggregation"),
    ("mechanism_postinfo_br_hr_per_setting.csv",
     "outputs/mechanism_profiles/postinfo_br_hr_per_setting.csv",
     "beneficial and harmful gate-application rates for all 12 settings"),
    ("mechanism_postinfo_br_hr_overall.csv",
     "outputs/mechanism_profiles/postinfo_br_hr_overall.csv",
     "setting-balanced beneficial and harmful gate-application rates"),
    ("fixed_alignment_per_failure_type_seed.csv",
     "outputs/fixed_patch_alignment/per_failure_type_seed.csv",
     "FixedPatch-to-type centroid alignment paired with FixedPatch and DynaPatch held-test RR"),
    ("fixed_alignment_per_failure_type_setting.csv",
     "outputs/fixed_patch_alignment/per_failure_type_setting.csv",
     "the same alignment table with matching failure types pooled across seeds"),
    ("fixed_alignment_per_cell_correlations.csv",
     "outputs/fixed_patch_alignment/per_cell_correlations.csv",
     "within-cell Spearman associations at failure-type support thresholds 2, 3 and 5"),
    ("fixed_alignment_per_setting_correlations.csv",
     "outputs/fixed_patch_alignment/per_setting_correlations.csv",
     "descriptive per-setting associations after pooling matching types across seeds"),
    ("fixed_alignment_support_sensitivity.csv",
     "outputs/fixed_patch_alignment/support_sensitivity.csv",
     "sign and magnitude summary of within-cell associations across support thresholds"),
    ("directional_separation_vs_rr_gain.csv",
     "outputs/fixed_patch_alignment/directional_separation_vs_rr_gain.csv",
     "all 12 settings' direction heterogeneity and input-specific held-test RR gain"),
    ("directional_separation_leave_one_out.csv",
     "outputs/fixed_patch_alignment/directional_separation_leave_one_out.csv",
     "leave-one-setting-out sensitivity for direction heterogeneity versus RR gain"),
    ("fixed_alignment_pca_coordinates_g_cn_s101.csv",
     "outputs/fixed_patch_alignment/pca_coordinates_g_cn_s101.csv",
     "PCA coordinates for the predeclared illustrative G-CN seed-101 cell"),
    ("distrep_expert_oracle_arm_metrics.csv",
     "outputs/distrep_expert_oracle_gtsrb_resnet50_s101/expert_oracle_arm_metrics.csv",
     "merged DistRep, five pre-integration experts and the non-deployable test-label oracle"),
    ("distrep_expert_unique_repairs.csv",
     "outputs/distrep_expert_oracle_gtsrb_resnet50_s101/expert_unique_repairs.csv",
     "held-test failures repaired uniquely by each pre-integration DistRep expert"),
    ("distrep_expert_overlap_histogram.csv",
     "outputs/distrep_expert_oracle_gtsrb_resnet50_s101/expert_repair_overlap_histogram.csv",
     "held-test repair overlap across the five pre-integration DistRep experts"),
    ("rq2_operating_points_per_setting.csv",
     "outputs/rq2_baseline_behavior/operating_points_per_setting.csv",
     "RQ2 repair-regression operating points for every method and all 12 settings"),
    ("rq2_operating_points_overall.csv",
     "outputs/rq2_baseline_behavior/operating_points_overall.csv",
     "setting-balanced RQ2 repair-regression operating points"),
    ("rq2_consistency_per_setting.csv",
     "outputs/rq2_baseline_behavior/consistency_per_setting.csv",
     "every setting-level comparison underlying the RQ2 consistency counts"),
    ("rq2_consistency_summary.csv",
     "outputs/rq2_baseline_behavior/consistency_summary.csv",
     "win, tie, loss and not-estimable counts for the predeclared RQ2 comparisons"),
    ("rq2_type_rr_distribution_summary.csv",
     "outputs/rq2_baseline_behavior/type_rr_distribution_summary.csv",
     "hierarchically balanced failure-type RR quantiles and partial/zero/full shares"),
    ("rq2_type_rr_ecdf.csv",
     "outputs/rq2_baseline_behavior/type_rr_ecdf.csv",
     "weighted failure-type RR observations and cumulative mass used by the RQ2 ECDF"),
    ("rq2_failure_type_wtl_per_setting.csv",
     "outputs/rq2_baseline_behavior/failure_type_wtl_per_setting.csv",
     "paired per-setting failure-type comparisons for DynaPatch-NoGate and every RQ2 baseline"),
    ("rq2_failure_type_wtl_summary.csv",
     "outputs/rq2_baseline_behavior/failure_type_wtl_summary.csv",
     "jointly estimable win, tie and loss counts for three failure-type metrics"),
    ("rq2_regression_behavior_per_setting.csv",
     "outputs/rq2_baseline_behavior/regression_behavior_per_setting.csv",
     "overall, worst-class and affected-class regression behavior for every method and setting"),
    ("rq2_regression_behavior_overall.csv",
     "outputs/rq2_baseline_behavior/regression_behavior_overall.csv",
     "setting-balanced coordinates for the regression breadth-versus-severity figure"),
    ("rq2_regression_wtl_per_setting.csv",
     "outputs/rq2_baseline_behavior/regression_wtl_per_setting.csv",
     "paired per-setting regression comparisons for DynaPatch (gated) and every RQ2 baseline"),
    ("rq2_regression_wtl_summary.csv",
     "outputs/rq2_baseline_behavior/regression_wtl_summary.csv",
     "win, tie and loss counts for three clean-test regression metrics"),
    ("rq2_distrrep_widest_affected_per_setting.csv",
     "outputs/rq2_baseline_behavior/distrrep_widest_affected_per_setting.csv",
     "settings in which DistrRep has the widest affected clean-class fraction"),
    ("rq2_arachne_localized_spike_per_setting.csv",
     "outputs/rq2_baseline_behavior/arachne_localized_spike_per_setting.csv",
     "paired checks for narrower affected-class fraction but higher worst-class Reg in Arachne"),
    ("rq2_arachne_localized_spike_summary.csv",
     "outputs/rq2_baseline_behavior/arachne_localized_spike_summary.csv",
     "per-comparator counts for the Arachne localized-spike pattern"),
    ("rq2_positive_type_rr_weighted.csv",
     "outputs/rq2_baseline_behavior/positive_type_rr_weighted.csv",
     "estimable positive-RR failure types with hierarchical conditional weights"),
    ("rq2_positive_type_rr_summary.csv",
     "outputs/rq2_baseline_behavior/positive_type_rr_summary.csv",
     "conditional failure-type RR mean and quartiles for every method"),
    ("arachne_capacity.csv", "outputs/p5_arachne_capacity/per_cell.csv",
     "Arachne with its localisation relaxed to N in {16,64,256,1024} crossed with bound_scale; "
     "seed 101 only"),
    ("params_changed.csv", "outputs/p6_capacity/per_cell.csv",
     "how many parameters each method actually changes, plus BatchNorm buffer drift"),
    ("all_metrics_per_seed.csv", "outputs/ALL_RQ_DATA.csv",
     "every method's four headline metrics, one row per (method, setting, seed, metric)"),
    ("all_metrics_best_config.csv", "outputs/BEST_DATA.csv",
     "the same, each method at its best configuration"),
    ("rq4_final.csv", "outputs/rq4_final.csv",
     "the table-4 source: RR_repair, RR_held, Reg, CReg per (method, setting, seed)"),
    ("gate_curves.csv", "outputs/gate_zoo_curves.csv",
     "the gate's full risk-coverage curve (protocol C), per gate, per setting"),
    ("gate_performance.csv", "outputs/gate_performance.csv",
     "the gate as a classifier: AUROC/AUPRC/F1 per evidence arm, per setting"),
]


def shape(p: Path) -> tuple[int, list[str]]:
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        return sum(1 for _ in r), head


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, src, desc in FILES:
        s = ROOT / src
        if not s.exists():
            print(f"[skip] {src} missing")
            continue
        dst = OUT / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        # the 23 MB base table is symlinked, not duplicated; everything else is a real copy so
        # the directory survives being moved or zipped for the artefact package.
        if s.stat().st_size > 5 << 20:
            dst.symlink_to(Path("../..") / src)
        else:
            shutil.copy2(s, dst)
        n, cols = shape(dst)
        rows.append((name, src, desc, n, cols))
        print(f"{name:46s} {n:>9,} rows")

    lines = ["# Raw data — every analysis CSV in one directory", "",
             "`outputs/csv/`. Flat, not grouped by RQ. Copied by `scripts/collect_csv.py`; the",
             "sources stay where their own scripts write them, so re-running an analysis and then",
             "re-running this script refreshes the copy.", "",
             "No aggregation and no interpretation in this file — row counts and column names only.",
             "", f"{len(rows)} files.", ""]
    for name, src, desc, n, cols in rows:
        lines += [f"### [`{name}`](../outputs/csv/{name}) — {n:,} rows", "",
                  desc + f"  Source: `{src}`.", "",
                  "columns: " + ", ".join(f"`{c}`" for c in cols), ""]
    (ROOT / "note/RAW_CSV.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {ROOT / 'note/RAW_CSV.md'}  ({len(rows)} files)")


if __name__ == "__main__":
    main()
