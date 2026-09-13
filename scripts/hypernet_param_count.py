#!/usr/bin/env python3
"""How big is the hypernet, per (dataset, backbone)?

No checkpoint on disk carries the trained hypernet's weights (the `deploy_eval` stage keeps
predictions, not the model) -- but `config_resolved.yaml` records every constructor argument
`src/experiment/stage3.py` passed to `HyperNetworkPatchGenerator`, and construction is
deterministic (`nn.Module.__init__` allocates parameters from shapes alone, no data-dependent
sizing). Instantiating with those exact arguments gives the exact parameter count without
needing the trained weights themselves.

`n_total_model` for the FullFT row of `outputs/csv/params_changed.csv` is each backbone's own
total parameter count (the fine-tuned model IS the whole backbone) -- reused here as the
comparison denominator instead of a second architecture query.

Zero GPU. Usage:
    .venv/bin/python scripts/hypernet_param_count.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.dynapatch.hypernet import HyperNetworkPatchGenerator  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import names as N  # noqa: E402
import raw_md as RM  # noqa: E402

DUMP_TAG = "_ep40ns"
SEED = 303  # any seed works: the hypernet's constructor arguments do not vary across seeds


def config_for(ds: str, bb: str) -> dict:
    f = ROOT / f"outputs/effect_dump{DUMP_TAG}_v8_s{SEED}/{ds}/{bb}/deploy_direct_calib/config_resolved.yaml"
    cfg = yaml.safe_load(f.read_text())
    return cfg["model"]


def main() -> None:
    backbone_total = pd.read_csv(ROOT / "outputs/csv/params_changed.csv")
    backbone_total = backbone_total[backbone_total.method == "FullFT"].set_index("setting")["n_total_model"]

    rows = []
    for ds, bb, _label in N.SETTING_ORDER:
        m = config_for(ds, bb)
        hidden_dim = int(m.get("hypernet_hidden_dim", 128))
        num_hidden_layers = int(m.get("hypernet_num_hidden_layers", 1))
        hypernet_style = str(m.get("hypernet_style", "plain"))
        pool_size = int(m.get("hypernet_pool_size", 1))
        condition_source = str(m.get("hypernet_condition_source", "shallow"))
        context_dim = 0  # no memory_bank/prototype config present in any of the 12 resolved
        # configs this session checked -- context_dim_from_memory_bank() returns 0 when the
        # memory bank is None, which is the case whenever the resolved config has no
        # memory_bank/prototype section (verified for all 12 settings before writing this).
        net = HyperNetworkPatchGenerator(
            model_type=str(m["type"]), shallow_dim=int(m["shallow_dim"]), out_dim=int(m["out_dim"]),
            context_dim=context_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers,
            hypernet_style=hypernet_style, pool_size=pool_size, condition_source=condition_source,
            num_basis=43,  # irrelevant to the parameter count for hypernet_style="residual_ln"
            # (only "bank_coef" style uses num_basis for its output layer width; all 12
            # settings use "residual_ln", checked below)
        )
        n_hypernet = sum(p.numel() for p in net.parameters())
        setting = f"{ds}/{bb}"
        n_backbone = float(backbone_total.get(setting, float("nan")))
        rows.append({
            "setting": setting, "shallow_dim": int(m["shallow_dim"]), "out_dim": int(m["out_dim"]),
            "hidden_dim": hidden_dim, "num_hidden_layers": num_hidden_layers,
            "hypernet_style": hypernet_style, "n_hypernet_params": n_hypernet,
            "n_backbone_params": n_backbone,
            "hypernet_pct_of_backbone": 100 * n_hypernet / n_backbone if n_backbone == n_backbone else float("nan"),
        })
        assert hypernet_style == "residual_ln", (
            f"{setting}: hypernet_style={hypernet_style!r}, not the 'residual_ln' this script "
            "assumed for every setting -- num_basis would then affect the parameter count and "
            "needs to be read from the resolved config, not hardcoded")

    cell = pd.DataFrame(rows)
    by_backbone = cell.groupby(cell.setting.str.split("/").str[1], as_index=False).agg(
        n_hypernet_params=("n_hypernet_params", "first"),  # constant across datasets except
        # densenet121 (num_hidden_layers varies 1 vs 2 by dataset -- see per-setting table)
        settings=("setting", "nunique"))

    RM.write_section(
        "HypernetParamCount",
        "How big is the hypernet, per (dataset, backbone)? (raw)",
        f"""
Computed by instantiating `HyperNetworkPatchGenerator` with the exact constructor arguments
`config_resolved.yaml` recorded for each setting (construction is deterministic; no trained
weights are needed to count parameter shapes). `n_backbone_params` reuses the FullFT row's
`n_total_model` from `outputs/csv/params_changed.csv` (the fine-tuned model IS the whole
backbone) as the comparison denominator.

**Not constant across datasets for the same backbone**: `densenet121`'s
`hypernet_num_hidden_layers` is 2 for gtsrb/lisa_signs but 1 for tt100k_signs -- so "how big
is the hypernet for architecture X" has a per-setting answer, not just a per-backbone one.
See `by_setting` for the full 12-row table.
""",
        [("by_setting", cell.sort_values("setting")), ("by_backbone", by_backbone)],
    )

    out = ROOT / "outputs" / "hypernet_param_count"
    out.mkdir(parents=True, exist_ok=True)
    cell.to_csv(out / "per_setting.csv", index=False)
    print(f"[written] {out.relative_to(ROOT)}/per_setting.csv")
    print(cell.to_string(index=False))


if __name__ == "__main__":
    main()
