"""Shared definitions for the Dyck matched-depth control (DYCK_MDEPTH_PREREG.md).

A2f, the registered primary metric. Hewitt's distance-averaged closing accuracy (A2 in
eval_dyck_literature) scores every prefix where a closer is GRAMMATICAL (depth > 0). Under a
fixed-(L, D) sampler some of those prefixes are FORCED OPENS: closing there would make it
impossible to reach depth D. At L32 D12 they are 4,875 of 15,131 scored prefixes (32%). The
CE-optimal predictor puts exactly zero mass on both closers there, so its A2 is 0.940, not 1
(gate G3) -- and a model TRAINED at D12 learns that, so its A2 at those prefixes is a ratio of
two vanishing probabilities. A2f scores only the prefixes where the SAMPLER can emit a closer;
the CE-optimal predictor scores exactly 1.000 on every cell. On the ladder's D4-trained 4-layer
checkpoints A2f and A2 differ by <= 0.003 (RoPE 0.752 vs 0.755, MapWM 0.922 vs 0.923), so the
depth-OOD reference numbers carry over unchanged.

The feasibility mask is read from DyckWorld's own per-prefix sampler entropy (rule 7): 1.5 ln2
= open or close allowed, 0 = forced close, ln2 = forced open (or depth 0).
"""
import math
import numpy as np
import torch

from mapformer.environment_dyck import DyckWorld, CLOSE_P, CLOSE_B
from mapformer.eval_dyck_literature import cell_data, metrics, N

CELLS = [(32, 4), (32, 12), (128, 12)]


def cell_with_mask(w, L, D):
    """eval_dyck_literature.cell_data (the ladder's exact sequences) plus the closer-feasible mask."""
    d = cell_data(w, L, D)
    inp, _, _, ent = w.batch(N, L, D, np.random.default_rng(424242 + 1000 * L + D))
    assert torch.equal(inp, d["inp"]), "cell regeneration diverged from cell_data"
    d["close_ok"] = (ent - 1.5 * math.log(2)).abs().lt(1e-9) | ent.abs().lt(1e-12)
    return d


def a2_masked(P, d, mask=None):
    """A2 (mean over distances j of the renormalised legal-closer probability), on dp & mask."""
    dp = d["dp"] if mask is None else d["dp"] & mask
    pc = P.gather(-1, d["corr"].unsqueeze(-1)).squeeze(-1)
    pw = P.gather(-1, d["wrng"].unsqueeze(-1)).squeeze(-1)
    ratio = pc / (pc + pw).clamp_min(1e-12)
    js = sorted({int(x) for x in d["dist"][dp].unique()})
    return float(np.mean([float(ratio[dp & (d["dist"] == j)].mean()) for j in js]))


def all_metrics(P, d, logits=None):
    """The ladder's metric dict plus A2f (primary) and the forced-open share."""
    m = metrics(P, d, logits)
    m["A2f_close_acc_dist_feasible"] = a2_masked(P, d, d["close_ok"])
    m["closer_mass_at_forced_open"] = float((P[..., CLOSE_P] + P[..., CLOSE_B])[d["dp"] & ~d["close_ok"]].mean()) \
        if bool((d["dp"] & ~d["close_ok"]).any()) else float("nan")
    return m
