"""DPGate: the paper's 9-feature (3 pre-info + 6 post-info) commit/rollback gate.

Ported from the offline analysis pipeline (`scripts/gate_protocol_b.py`,
`scripts/analyze_response_gate_lobo.py`, `scripts/probe_gonogo_pre_vs_prepost.py`) into a
deployable component. Feature formulas are copied byte-for-byte from
`analyze_response_gate_lobo.build_features` (including its `dH`/`H_base` both being normalized
by `log C` -- a known asymmetry with the paper's literal text, which the paper's ΔH is specified
unnormalized; kept as-is on purpose so this matches the already-verified offline gate rather than
silently drifting from it).

Two independent online mechanisms were checked before writing this and found to be permanently
inert for every shipped setting (see the `dynapatch-deploy-policy-promotion-quirk` memory note):
`deployment.gate_threshold: 9999.0` makes the deploy-time router accept everything, and
`model.route_mode: always_on` makes the in-model distance router a no-op multiplier. This module
is the first thing in `src/` that can actually reject a patch for a real reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Order matches scripts/gate_lambda_sweep.py's `LAYERS["4 pre+post"]` -- the paper's shipped
# 9-feature gate. Do not reorder without also updating FeatureGate.FEATURE_NAMES call sites.
FEATURE_NAMES: tuple[str, ...] = (
    "pB_max", "pB_margin", "H_base",
    "pP_max", "kl", "dH", "dp_c", "rho", "rho_worst",
)
_EPS = 1e-8


def compute_gate_features(base_logits: torch.Tensor, patched_logits: torch.Tensor) -> torch.Tensor:
    """Compute the 9 gate features from pre- and post-patch logits.

    `base_logits`/`patched_logits`: [N, C]. Returns [N, 9] in `FEATURE_NAMES` order. `c`/`j` (the
    original top-1/runner-up classes) are fixed from `base_logits` and never recomputed from the
    patched side, matching the paper's definition and the offline script's own guarantee.
    """
    n, num_classes = base_logits.shape
    log_c = float(torch.log(torch.tensor(float(num_classes))))
    p_ori = torch.softmax(base_logits, dim=1)
    p_pat = torch.softmax(patched_logits, dim=1)

    order = torch.argsort(base_logits, dim=1, descending=True)
    c = order[:, 0]
    j = order[:, 1]
    rows = torch.arange(n, device=base_logits.device)

    m_base = base_logits[rows, c] - base_logits[rows, j]
    dlog = patched_logits - base_logits
    dm = dlog[rows, c] - dlog[rows, j]
    rho = (dm / (m_base.abs() + _EPS)).clamp(-10.0, 10.0)

    dlog_masked = dlog.clone()
    dlog_masked[rows, c] = float("-inf")
    dm_worst = dlog[rows, c] - dlog_masked.max(dim=1).values
    rho_worst = (dm_worst / (m_base.abs() + _EPS)).clamp(-10.0, 10.0)

    def entropy(p: torch.Tensor) -> torch.Tensor:
        return -(p * (p + 1e-12).log()).sum(dim=1) / log_c

    kl = ((p_pat * ((p_pat + 1e-12).log() - (p_ori + 1e-12).log())).sum(dim=1) / log_c).clamp(0.0, 50.0)

    p_b_max = p_ori[rows, c]
    p_b_margin = p_b_max - p_ori[rows, j]
    h_base = entropy(p_ori)
    p_p_max = p_pat.max(dim=1).values
    d_h = entropy(p_pat) - h_base
    dp_c = p_pat[rows, c] - p_ori[rows, c]

    return torch.stack([p_b_max, p_b_margin, h_base, p_p_max, kl, d_h, dp_c, rho, rho_worst], dim=1)


def gain_labels(base_correct: torch.Tensor, patched_correct: torch.Tensor) -> torch.Tensor:
    """u(x,y) in {-1,0,+1}: +1 repaired, -1 regressed, 0 otherwise (Eq. patch_outcome)."""
    repaired = (~base_correct) & patched_correct
    regressed = base_correct & (~patched_correct)
    return torch.where(repaired, 1, torch.where(regressed, -1, 0))


class FeatureGate:
    """Multinomial logistic regression over the 9 gate features, scored as s_lambda(x).

    Fitting needs `scikit-learn` (a fit-time convenience only); scoring after `load()` is pure
    torch and does not need it. Persist a fit with `save()` so a deploy run can load a specific,
    auditable set of coefficients instead of silently refitting each time (see
    `dynapatch-deploy-policy-promotion-quirk` for why an unaudited online gate is the exact kind
    of thing this repo has been repeatedly burned by).
    """

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, coef: torch.Tensor,
                 intercept: torch.Tensor, classes: tuple[int, ...]) -> None:
        self.mean = mean
        self.scale = scale
        self.coef = coef
        self.intercept = intercept
        self.classes = classes

    @classmethod
    def fit(cls, features: torch.Tensor, gain: torch.Tensor, *, class_weight: str | None = None) -> "FeatureGate":
        """Fit on features [N,9] and gain labels [N] in {-1,0,+1} (Eq. patch_outcome)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        x = features.detach().cpu().numpy()
        y = gain.detach().cpu().numpy()
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(max_iter=3000, random_state=0, class_weight=class_weight)
        model.fit(scaler.transform(x), y)
        return cls(
            mean=torch.as_tensor(scaler.mean_, dtype=torch.float32),
            scale=torch.as_tensor(scaler.scale_, dtype=torch.float32),
            coef=torch.as_tensor(model.coef_, dtype=torch.float32),
            intercept=torch.as_tensor(model.intercept_, dtype=torch.float32),
            classes=tuple(int(v) for v in model.classes_),
        )

    def _class_probs(self, features: torch.Tensor) -> torch.Tensor:
        device, dtype = features.device, features.dtype
        mean = self.mean.to(device=device, dtype=dtype)
        scale = self.scale.to(device=device, dtype=dtype)
        coef = self.coef.to(device=device, dtype=dtype)
        intercept = self.intercept.to(device=device, dtype=dtype)
        z = (features - mean) / scale
        logits = z @ coef.T + intercept
        if logits.shape[1] == 1:
            # sklearn's binary-class special case: one logit is P(classes_[1]) vs the rest.
            p1 = torch.sigmoid(logits[:, 0])
            return torch.stack([1.0 - p1, p1], dim=1)
        return torch.softmax(logits, dim=1)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """s(x) = P(gain=+1) - P(gain=-1)."""
        probs = self._class_probs(features)
        pos = probs[:, self.classes.index(1)] if 1 in self.classes else torch.zeros(features.size(0), device=probs.device, dtype=probs.dtype)
        neg = probs[:, self.classes.index(-1)] if -1 in self.classes else torch.zeros(features.size(0), device=probs.device, dtype=probs.dtype)
        return pos - neg

    def decide(self, features: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        """Commit (apply the patch) iff s(x) > 0 under regression cost `lam` (Eq. gate_decision_ind)."""
        probs = self._class_probs(features)
        pos = probs[:, self.classes.index(1)] if 1 in self.classes else torch.zeros(features.size(0), device=probs.device, dtype=probs.dtype)
        neg = probs[:, self.classes.index(-1)] if -1 in self.classes else torch.zeros(features.size(0), device=probs.device, dtype=probs.dtype)
        return (pos - float(lam) * neg) > 0.0

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "classes": list(self.classes),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureGate":
        payload = json.loads(Path(path).read_text())
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError(
                f"Gate file feature order {payload['feature_names']} does not match "
                f"the current FEATURE_NAMES {FEATURE_NAMES}."
            )
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            scale=torch.tensor(payload["scale"], dtype=torch.float32),
            coef=torch.tensor(payload["coef"], dtype=torch.float32),
            intercept=torch.tensor(payload["intercept"], dtype=torch.float32),
            classes=tuple(payload["classes"]),
        )
