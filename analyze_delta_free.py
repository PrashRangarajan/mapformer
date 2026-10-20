"""Registered analysis for DELTA_FREE_PREREG.md: is PoPE's peak pinning (delta stuck at 0) what makes its polar score
hurt an index phase and help a path phase? CPU.
  python3 -m mapformer.analyze_delta_free [--root runs/delta_free] [--raw _DELTA_FREE_RAW.json] [--seeds 1600-1607]
                                          [--floors docs/audits/2026-10-20/delta_free_floors_out.txt]
                                          [--json-out DELTA_FREE_VERDICTS.json | --no-json] [--no-probe]
  python3 -m mapformer.analyze_delta_free --selftest   # every label from synthetic data + code checks on tiny models
  python3 -m mapformer.analyze_delta_free --power      # synthetic power table for the primary
Layout: <root>/p0/<arm>_s<S>/<arm>.pt (train_variant), eval = eval_noise_refine's JSON (keys "0.0|<arm>|<T>", values
[seed, acc, nll]; held-out map env seed 10000, np seed 1234 + s, 100 walks, T = 128 / 512 / 1024).

Arms (phase x delta, + one):  PI = PoPE-Flat (index, pinned)        FI = PoPE-FreeDelta (index, free)
                              PP = MapPoPE-Flat (path, pinned)      FP = MapPoPE-FreeDelta (path, free)
                              ZP = MapPoPE-FrozenDelta (path, delta frozen at its U(-2 pi, 0) draw; secondary S1)
Per seed s (all arms share the seed's map, training stream and, within a phase, every non-delta initial weight):
  dI_s = FI_s - PI_s,  dP_s = FP_s - PP_s,  I_s = dP_s - dI_s
I has PAPER2X2's orientation (path minus index of a treatment's effect; the PoPE score's was +0.154), so the peak-pinning
account predicts I < 0: freeing delta favours the index phase, the pin favoured the path phase.

PRIMARY (T = 128 held-out accuracy), tests = exact sign-flip over the 8 seeds (stats_core.signflip_p, min p 2/256),
TAU_I = 0.05, TAU_P = 0.01, alpha 0.05; ub(dI) = upper end of the 90% sign-flip-inversion CI (one-sided 95% bound).
In this order:
  UNREADABLE                              mean PP < 0.95 (the pinned path arm did not learn) or mean PI >= the own-stream
                                          retrace floor (no index deficit to repair);
  NOT SUPPORTED (freeing hurts index)     mean dI < 0 and p(dI) < .05;
  NOT SUPPORTED (helps neither)           ub(dI) < TAU_I (a gain of TAU_I on the index arm is excluded);
  PEAK-PINNING SUPPORTED                  mean dI >= TAU_I and p(dI) < .05 (registered contrast) AND mean I <= -TAU_I and
                                          p(I) < .05 (the interaction reverses sign) AND the path arm is not helped
                                          (not: mean dP > 0 with p(dP) < .05, nor more SOLVED with Fisher p < .05);
                                          qualifier "path cost" if mean dP <= -TAU_P with p(dP) < .05 or fewer SOLVED with
                                          Fisher p < .05, else "path unchanged within noise";
  NOT SUPPORTED (helps both)              mean dI >= TAU_I, p(dI) < .05, and the path arm helped (as above);
  UNMEASURED                              otherwise.
Amendment 1 (docs/audits/2026-10-20/delta_free_audit.md): "path helped" needs mean dP >= +TAU_P (mirror of "path cost");
registered secondary P1 (FrozenDelta / FreeDelta - MapPoPE-Flat at T = 1024, Holm over 2, TAU_P1 = 0.03); S4 decay-only
references; S7 (fast / slow half delta reset on PoPE-FreeDelta); dropout-rescaled dI (descriptive); non-finite accuracies
refused; every checkpoint's recipe asserted.
"""
import argparse, importlib.util, json, math, os, re, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.stats_core import (signflip_p, paired_p, perm2_p, fisher_solved, classify_run, mde, MDE_K,
                                  bimodality)

REPO = "/home/prashr/mapformer"
PI, FI, PP, FP, ZP = "PoPE-Flat", "PoPE-FreeDelta", "MapPoPE-Flat", "MapPoPE-FreeDelta", "MapPoPE-FrozenDelta"
ARMS = (PI, FI, PP, FP, ZP)
INDEX_ARMS, PATH_ARMS = (PI, FI), (PP, FP, ZP)
TAU_I, TAU_P, ALPHA = 0.05, 0.01, 0.05
TAU_P1 = 0.03                           # Amendment 1, P1: path-arm T = 1024 cost threshold (~1.3 x PAPER2X2's MapPoPE sd)
DECAY_ONLY_REPINNED = 0.036             # Amendment 1: re-pinned share of U(-2 pi x 0.4471, 0) (decay alone, no learning)
# Amendment 1 (MINOR 3): the recipe every checkpoint must carry (config / args), per driver mode
RECIPES = {
    "batch": dict(epochs=300, lr=1e-3, schedule="cosine", n_steps=128, batch_size=128, n_batches=98, n_layers=1,
                  n_heads=2, d_model=128, n_landmarks=0, data_workers=0, fast_attn=False),
    "smoke": dict(epochs=2, lr=1e-3, schedule="cosine", n_steps=32, batch_size=8, n_batches=2, n_layers=1,
                  n_heads=2, d_model=128, n_landmarks=0, data_workers=0, fast_attn=False),
}
PATH_PREMISE = 0.95
RETRACE_DEFAULT = 0.934                 # PAPER2X2's stream (seeds 0-7); the batch uses its own stream's mean (--floors)
LENGTHS = (128, 512, 1024)
SUP, NS, UNM, UNR = "PEAK-PINNING SUPPORTED", "NOT SUPPORTED", "UNMEASURED", "UNREADABLE"
NA, NO = 4, 17                          # action tokens 0..3, observation tokens 4..20 (16 types + blank)
KP_PATH = f"{REPO}/docs/paper/expansion/scripts/mappope/kernel_probe.py"


# ------------------------------------------------------------------------------ statistics
_SIGNS = {}


def _signs(n):
    if n not in _SIGNS:
        import itertools
        _SIGNS[n] = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    return _SIGNS[n]


def sf_ci(d, alpha=0.10, step=0.0005):
    """Test-inversion CI of mean(d) for the exact sign-flip test: every shift s with p(d - s) >= alpha. The accepted set
    lies inside [min d, max d] (beyond it every d - s has one sign and p = 2 / 2^n). Returns (lo, hi)."""
    d = np.asarray(d, float); n = d.size; S = _signs(n)
    grid = np.arange(d.min() - step, d.max() + 2 * step, step)
    X = d[None, :] - grid[:, None]                                    # (G, n)
    null = np.abs(X @ S.T) / n                                        # (G, 2^n)
    obs = np.abs(X.mean(1))
    p = (null >= obs[:, None] - 1e-12).mean(1)
    ok = grid[p >= alpha]
    if ok.size == 0:
        m = float(d.mean()); return (m, m)
    return float(ok.min()), float(ok.max())


def contrast(a, b, seeds, unpaired=True):
    """b - a, paired by seed. Every number rule 5 asks for (the unpaired permutation p only where it means something)."""
    x = np.array([a[s] for s in seeds], float); y = np.array([b[s] for s in seeds], float)
    d = y - x; n = d.size; sd = float(d.std(ddof=1)) if n > 1 else 0.0
    lo, hi = sf_ci(d)
    return {"mean": float(d.mean()), "sd": sd, "n": n, "pos": int((d > 0).sum()),
            "p_sf": signflip_p(d)["p"], "p_t": paired_p(d), "p_perm": perm2_p(x, y)["p"] if unpaired else None,
            "mde_house": MDE_K * sd / math.sqrt(n), "mde_exact": mde(sd, n) if sd > 0 else 0.0,
            "ci90": [lo, hi], "per_seed": d.tolist()}


def fmt(name, c):
    return (f"{name:44s} {c['mean']:+.4f} (sd {c['sd']:.4f}; MDE house {c['mde_house']:.4f} / exact-t {c['mde_exact']:.4f}; "
            f"{c['pos']}/{c['n']} +) sign-flip p {c['p_sf']:.4f}, t p {c['p_t']:.4f}"
            + (f", unpaired perm p {c['p_perm']:.4f}" if c['p_perm'] is not None else "") + "; "
            f"90% sign-flip CI [{c['ci90'][0]:+.4f}, {c['ci90'][1]:+.4f}]")


def holm(ps):
    ps = np.asarray(ps, float); k = len(ps); o = np.argsort(ps); out = np.empty(k); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (k - r) * ps[i])); out[i] = run
    return out.tolist()


# ------------------------------------------------------------------------------ registered rule
def primary_label(acc, solved, seeds, retrace):
    """acc[arm][seed] = T=128 accuracy; solved[arm][seed] = bool. Returns (label, details)."""
    dI = contrast(acc[PI], acc[FI], seeds)
    dP = contrast(acc[PP], acc[FP], seeds)
    Iser = {s: (acc[FP][s] - acc[PP][s]) - (acc[FI][s] - acc[PI][s]) for s in seeds}
    I = contrast({s: 0.0 for s in seeds}, Iser, seeds, unpaired=False)
    kPP, kFP = sum(solved[PP][s] for s in seeds), sum(solved[FP][s] for s in seeds)
    pF = fisher_solved(kPP, len(seeds), kFP, len(seeds))
    mPP = float(np.mean([acc[PP][s] for s in seeds])); mPI = float(np.mean([acc[PI][s] for s in seeds]))
    path_helped = (dP["mean"] >= TAU_P and dP["p_sf"] < ALPHA) or (kFP > kPP and pF < ALPHA)   # Amendment 1, MAJOR 1
    path_hurt = (dP["mean"] <= -TAU_P and dP["p_sf"] < ALPHA) or (kFP < kPP and pF < ALPHA)
    index_helped = dI["mean"] >= TAU_I and dI["p_sf"] < ALPHA
    det = {"dI": dI, "dP": dP, "I": I, "solved_PP": kPP, "solved_FP": kFP, "fisher_path": pF,
           "mean_PP": mPP, "mean_PI": mPI, "retrace": retrace, "ub_dI": dI["ci90"][1],
           "path_helped": path_helped, "path_hurt": path_hurt, "index_helped": index_helped}
    if mPP < PATH_PREMISE or mPI >= retrace:
        why = (f"mean PP {mPP:.4f} < {PATH_PREMISE}" if mPP < PATH_PREMISE else
               f"mean PI {mPI:.4f} >= retrace floor {retrace:.4f}")
        return f"{UNR} ({why})", det
    if dI["mean"] < 0 and dI["p_sf"] < ALPHA:
        return f"{NS} (freeing delta hurts the index arm)", det
    if dI["ci90"][1] < TAU_I:
        return f"{NS} (helps neither: the index gain is below {TAU_I} at one-sided 95%)", det
    if index_helped and I["mean"] <= -TAU_I and I["p_sf"] < ALPHA and not path_helped:
        return f"{SUP} ({'path cost' if path_hurt else 'path unchanged within noise'})", det
    if index_helped and path_helped:
        return f"{NS} (helps both)", det
    return UNM, det


def frozen_label(acc, solved, seeds, ref=PP, arm=ZP):
    """Declared secondary S1: does a fixed random delta (no route back to 0) cost the path arm?"""
    c = contrast(acc[ref], acc[arm], seeds)
    k0, k1 = sum(solved[ref][s] for s in seeds), sum(solved[arm][s] for s in seeds)
    pF = fisher_solved(k0, len(seeds), k1, len(seeds))
    c.update(solved_ref=k0, solved_arm=k1, fisher=pF)
    if (c["mean"] <= -TAU_P and c["p_sf"] < ALPHA) or (k1 < k0 and pF < ALPHA):
        return "COSTS THE PATH ARM", c
    if (c["mean"] >= TAU_P and c["p_sf"] < ALPHA) or (k1 > k0 and pF < ALPHA):
        return "HELPS THE PATH ARM", c
    if c["ci90"][0] > -TAU_P and k1 >= k0:
        return f"NO COST (a loss of {TAU_P} excluded at one-sided 95%)", c
    return UNM, c


P1_COST, P1_NONE = "REMOVING THE PIN COSTS MAPPOPE'S LENGTH ROBUSTNESS", "NO OOD COST"


def p1_labels(acc1024, seeds):
    """Registered secondary P1 (Amendment 1, MAJOR 4; robustness, rule 10): X - MapPoPE-Flat at T = 1024 for X in
    {FrozenDelta, FreeDelta}, exact sign-flip, Holm over the 2. Cost if mean <= -TAU_P1 and Holm p < .05; NO OOD COST if
    the 90% sign-flip CI's lower end > -TAU_P1; else UNMEASURED."""
    cs = {X: contrast(acc1024[PP], acc1024[X], seeds) for X in (ZP, FP)}
    hp = dict(zip(cs, holm([c["p_sf"] for c in cs.values()])))
    out = {}
    for X, c in cs.items():
        c["p_holm"] = hp[X]
        if c["mean"] <= -TAU_P1 and hp[X] < ALPHA:
            lab = P1_COST
        elif c["ci90"][0] > -TAU_P1:
            lab = P1_NONE
        else:
            lab = UNM
        out[X] = (lab, c)
    return out


def check_finite(acc, seeds):
    """Amendment 1 (MINOR 2): refuse non-finite accuracies instead of letting NaN fall through to a label."""
    bad = [(T, a, s) for T in LENGTHS for a in ARMS for s in seeds
           if not np.isfinite(acc.get(T, {}).get(a, {}).get(s, float("nan")))]
    if bad:
        raise SystemExit(f"non-finite or missing accuracies (refusing to label): {bad}")


def check_recipe(b, arm, s, recipe):
    """Amendment 1 (MINOR 3): the checkpoint must carry the registered recipe, its arm, its seed and every epoch."""
    want = RECIPES[recipe]; c = b["config"]; A = c.get("args") or {}
    got = {k: c.get(k, A.get(k)) for k in want}
    got["data_workers"] = A.get("data_workers"); got["fast_attn"] = c.get("fast_attn", A.get("fast_attn"))
    bad = {k: (got[k], v) for k, v in want.items()
           if got[k] is None or not (abs(got[k] - v) < 1e-12 if isinstance(v, float) else got[k] == v)}
    if b.get("variant") != arm or b.get("seed") != s or len(b.get("losses", [])) != want["epochs"]:
        bad["identity"] = ((b.get("variant"), b.get("seed"), len(b.get("losses", []))), (arm, s, want["epochs"]))
    if bad:
        raise SystemExit(f"{arm} s{s}: checkpoint recipe differs from '{recipe}' (got, want): {bad}")


def decay_factor(epochs=300, n_batches=98, lr=1e-3, wd=0.05):
    """prod_k (1 - lr_k wd) under train.py's cosine schedule (5% warmup, cosine to 0.1 x lr): what AdamW's decoupled decay
    alone does to a parameter over the run (0.4471 for the batch recipe)."""
    T = epochs * n_batches; warm = max(1, int(0.05 * T)); f = 1.0
    for st in range(T):
        x = (st + 1) / warm if st < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((st - warm) / max(1, T - warm), 1)))
        f *= 1 - lr * x * wd
    return f


def reset_half(m, half):
    """S7: a copy of a free-delta model with delta set to 0 on the fast half (c < d/2) or the slow half (c >= d/2)."""
    import copy
    m2 = copy.deepcopy(m); lay = m2.layers[0]; n = lay.pope_delta.shape[-1]
    sl = slice(0, n // 2) if half == "fast" else slice(n // 2, n)
    with torch.no_grad():
        lay.pope_delta[:, sl] = 0.0
    return m2


# ------------------------------------------------------------------------------ data
def parse_seeds(s):
    if "-" in s and "," not in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def load_raw(path, what="acc"):
    raw = json.load(open(path)); acc = defaultdict(lambda: defaultdict(dict))
    for k, v in raw.items():
        p, arm, T = k.split("|")
        for s, a, nll in v:
            x = a if what == "acc" else nll
            acc[int(T)][arm][int(s)] = float("nan") if x is None else float(x)
    return acc


def retrace_mean(path, seeds):
    if not path or not os.path.exists(path):
        return RETRACE_DEFAULT, "default (PAPER2X2 stream)"
    vals = {}
    for line in open(path):
        m = re.match(r"T=128 s(\d+): .* retrace ([0-9.]+)", line)
        if m:
            vals[int(m.group(1))] = float(m.group(2))
    got = [vals[s] for s in seeds if s in vals]
    if len(got) != len(seeds):
        return RETRACE_DEFAULT, f"default (floors file lacks seeds {sorted(set(seeds) - set(vals))})"
    return float(np.mean(got)), f"own stream, mean over seeds {seeds[0]}-{seeds[-1]}"


def build(arm, cfg):
    from mapformer.train_variant import VARIANT_MAP
    return VARIANT_MAP[arm](vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                            n_layers=cfg["n_layers"], grid_size=cfg["grid_size"])


def load_run(root, arm, s):
    b = torch.load(f"{root}/p0/{arm}_s{s}/{arm}.pt", map_location="cpu", weights_only=False)
    m = build(arm, b["config"]); m.load_state_dict(b["model_state_dict"])
    return m.eval(), b


# ------------------------------------------------------------------------------ probe (secondaries; eval-only)
def delta_eff(layer):
    from mapformer.model_pope_delta import delta_eff as de
    return de(layer)


def phases(m, tok):
    """Angles (H, L, n) as the model builds them (index: t * theta_c; path: cumsum(W_out W_in x) * omega)."""
    L = tok.shape[1]; H = m.n_heads
    if hasattr(m, "theta_c"):
        return torch.outer(torch.arange(L, dtype=torch.float32), m.theta_c)[None].expand(H, -1, -1)
    x = m.token_emb(tok)
    return (torch.cumsum(m.action_to_lie(x), 1) * m.path_integrator.omega)[0].transpose(0, 1)


@torch.no_grad()
def forward_manual(m, tok, rescale=1.0):
    """Layer-0 rebuild of a 1-layer PoPE model with the EFFECTIVE delta (raw for free / frozen, clamped for pinned).
    rescale multiplies o_proj's input (softmax @ V), as DROPOUT_RESCORE's hook does (1 / (1 - p) = the descriptive
    dropout-scale re-score, Amendment 1 MINOR 4); 1.0 = the model's own eval forward."""
    lay = m.layers[0]; B, L = tok.shape; H, dh = m.n_heads, lay.d_head
    x = m.token_emb(tok); h = lay.norm1(x)
    Q = lay.q_proj(h).view(B, L, H, dh).transpose(1, 2)
    K = lay.k_proj(h).view(B, L, H, dh).transpose(1, 2)
    V = lay.v_proj(h).view(B, L, H, dh).transpose(1, 2)
    ang = phases(m, tok); ca, sa = ang.cos()[None], ang.sin()[None]
    mq, mk = F.softplus(Q), F.softplus(K)
    d = delta_eff(lay).view(1, H, 1, -1)
    cK = ca * d.cos() - sa * d.sin(); sK = sa * d.cos() + ca * d.sin()
    S = (torch.matmul(mq * ca, (mk * cK).transpose(-1, -2)) + torch.matmul(mq * sa, (mk * sK).transpose(-1, -2))) / math.sqrt(dh)
    S = S.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
    A = F.softmax(S, -1)
    out = lay.o_proj(rescale * torch.matmul(A, V).transpose(1, 2).reshape(B, L, -1))
    x = x + out; x = x + lay.ffn(lay.norm2(x))
    return m.out_proj(m.out_norm(x)), A[0], Q[0], K[0], ang


def retrace_class(a):
    """Per step t: j (step within a run that reverses the previous run, j <= previous run length) or 0.
    Copy of kernel_probe.retrace_class (sign_floors.py's retrace rule; checked equal in --selftest)."""
    T = len(a); out = np.zeros(T, dtype=np.int64); prev_run = cur = 0
    for t in range(T):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
            cur = 1
        if prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0:
            out[t] = cur
    return out


@torch.no_grad()
def coherence(m, Q, K, ang, tq, ts):
    """Target coherence per head (kernel_probe's PoPE branch with the effective delta): C = sum_c A_c cos(dphi_c - delta_c)
    / sum_c A_c, C0 = the same with delta removed."""
    lay = m.layers[0]
    dphi = ang[:, tq] - ang[:, ts]
    A = F.softplus(Q[:, tq]) * F.softplus(K[:, ts])
    d = delta_eff(lay)[:, None, :]
    C = (A * torch.cos(dphi - d)).sum(-1) / A.sum(-1)
    C0 = (A * torch.cos(dphi)).sum(-1) / A.sum(-1)
    return C.numpy(), C0.numpy()


@torch.no_grad()
def kernel_checks(m, decay=None):
    """Index arms: share of content pairs (head, action query, observation key) with K(3) > K(1), argmax over D in 1..16,
    with the effective delta. Every arm: the delta distribution."""
    lay = m.layers[0]; H, dh = m.n_heads, lay.d_head
    raw = lay.pope_delta.detach(); de = delta_eff(lay).detach()
    w = torch.atan2(torch.sin(de), torch.cos(de))                     # wrapped to (-pi, pi]
    out = {"deadzone_share": float((raw >= 0).float().mean()), "abs_eff_mean": float(de.abs().mean()),
           "repinned_share": float((w.abs() < 0.1).float().mean()), "abs_wrapped_mean": float(w.abs().mean()),
           "outside_clamp_share": float(((raw < -2 * math.pi) | (raw > 0)).float().mean())}
    if hasattr(lay, "pope_delta_init"):
        out["moved_abs_mean"] = float((raw - lay.pope_delta_init).abs().mean())
        out["moved_abs_max"] = float((raw - lay.pope_delta_init).abs().max())
        if lay.pope_delta.requires_grad:                          # free arms: what decay alone would have done
            dec = (DECAY if decay is None else decay) * lay.pope_delta_init
            res = (raw - dec).abs()                                   # (H, d): the learned part
            n = res.shape[-1]
            out["decay_resid_quartiles"] = [float(res[:, q * n // 4:(q + 1) * n // 4].mean()) for q in range(4)]
            wd_ = torch.atan2(torch.sin(dec), torch.cos(dec))
            out["decay_only_repinned"] = float((wd_.abs() < 0.1).float().mean())
    if hasattr(m, "theta_c"):
        E = m.token_emb.weight
        q = lay.q_proj(lay.norm1(E[:NA])).view(NA, H, dh); k = lay.k_proj(lay.norm1(E[NA:NA + NO])).view(NO, H, dh)
        D = torch.arange(0, 17, dtype=torch.float32)
        Kd = torch.einsum("ahc,ohc,hcd->haod", F.softplus(q), F.softplus(k),
                          torch.cos(m.theta_c[None, :, None] * D[None, None] - de[..., None]))
        out["frac_K3_gt_K1"] = float((Kd[..., 3] > Kd[..., 1]).float().mean())
        am = Kd[..., 1:].argmax(-1) + 1
        out["argmax_offset_hist"] = np.bincount(am.flatten().numpy(), minlength=17).tolist()
    return out


@torch.no_grad()
def probe_run(m, s, T=128, trials=100, rescale=1.0):
    from mapformer.environment import GridWorld
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    np.random.seed(1234 + s)
    rec = defaultdict(list); verify = 0.0; ok = tot = 0
    for i in range(trials):
        tok, _o, rev = env.generate_trajectory(T)
        locs = env.visited_locations
        inp = tok[None, :-1]
        logits, A, Q, K, ang = forward_manual(m, inp, rescale)
        if i < 2 and rescale == 1.0:
            verify = max(verify, float((logits - m(inp)).abs().max()))
        pred = logits[0].argmax(-1).numpy(); tgt = tok[1:].numpy(); msk = rev[1:].numpy()
        a = tok[0::2].numpy(); cell = np.array([x * 1000 + y for x, y in locs]); rc = retrace_class(a)
        An = A.numpy(); tq_l, ts_l = [], []
        for j in np.nonzero(msk)[0]:
            t = j // 2; qi = 2 * t
            keys = 2 * np.nonzero(cell[:t] == cell[t])[0] + 1
            rec["ok"].append(pred[qi] == tgt[qi]); rec["rc"].append(min(rc[t], 3))
            rec["mass_off1"].append(An[:, qi, qi - 1])
            rec["mass_rt"].append(An[:, qi, qi - (4 * rc[t] - 1)] if rc[t] > 0 else np.full(m.n_heads, np.nan))
            rec["mass_cell"].append(An[:, qi, keys].sum(-1))
            tq_l.append(qi); ts_l.append(keys[-1])
        if tq_l and not hasattr(m, "theta_c"):
            C, C0 = coherence(m, Q, K, ang, np.array(tq_l), np.array(ts_l))
            rec["C"].extend(list(C.T)); rec["C0"].extend(list(C0.T))
    R = {k: np.array(v) for k, v in rec.items()}
    h = int(np.argmax(R["mass_cell"].mean(0)))
    out = {"acc": float(R["ok"].mean()), "n": int(len(R["ok"])), "verify": verify, "head": h,
           "mass_off1": float(R["mass_off1"].mean(0).max()), "mass_cell": float(R["mass_cell"][:, h].mean())}
    for j, nm in ((1, "R1"), (2, "R2"), (3, "R3+"), (0, "none")):
        sel = R["rc"] == j
        out[f"acc_{nm}"] = float(R["ok"][sel].mean()) if sel.any() else float("nan")
        out[f"share_{nm}"] = float(sel.mean())
    if "C" in R:
        out["C"] = float(R["C"][:, h].mean()); out["C0"] = float(R["C0"][:, h].mean())
    return out


DECAY = decay_factor()


# ------------------------------------------------------------------------------ report
def analyse(root, raw, seeds, floors, json_out, do_probe, probe_trials=100, recipe="batch"):
    L = []; P = lambda s="": (print(s, flush=True), L.append(s))
    acc = load_raw(raw); nll = load_raw(raw, "nll")
    missing = [(a, s) for a in ARMS for s in seeds if s not in acc[128].get(a, {})]
    if missing:
        raise SystemExit(f"eval JSON lacks {missing}")
    check_finite(acc, seeds)
    rc_ = RECIPES[recipe]; dfac = decay_factor(rc_["epochs"], rc_["n_batches"], rc_["lr"])   # this recipe's decay-only factor
    retrace, rsrc = retrace_mean(floors, seeds)
    runs = {}; solved = defaultdict(dict); cls = defaultdict(dict); auc = defaultdict(dict); fin = defaultdict(dict)
    for a in ARMS:
        for s in seeds:
            b = torch.load(f"{root}/p0/{a}_s{s}/{a}.pt", map_location="cpu", weights_only=False)
            check_recipe(b, a, s, recipe)
            l = np.asarray(b["losses"], float); c = classify_run(l)
            solved[a][s] = c["registered"] == "SOLVED" and bool(np.isfinite(l).all())
            cls[a][s] = c["cls"] if np.isfinite(l).all() else "DIVERGED"
            auc[a][s] = float(np.mean(l)); fin[a][s] = float(l[-max(1, round(0.05 * len(l))):].mean())
            runs[(a, s)] = b
    P(f"# DELTA_FREE analysis (DELTA_FREE_PREREG.md), seeds {seeds[0]}-{seeds[-1]}, n = {len(seeds)}")
    P(f"retrace floor {retrace:.4f} ({rsrc}); TAU_I {TAU_I}, TAU_P {TAU_P}, TAU_P1 {TAU_P1}, alpha {ALPHA}, tests exact "
      f"sign-flip; recipe '{recipe}' asserted on all {len(ARMS) * len(seeds)} checkpoints; decay-only factor {dfac:.4f}")
    P("\n## Per arm (held-out accuracy, mean +/- sd; SOLVED = final-5% training loss < 0.05)")
    for a in ARMS:
        cells = " | ".join(f"T={T} {np.mean([acc[T][a][s] for s in seeds]):.4f} +/- "
                           f"{np.std([acc[T][a][s] for s in seeds], ddof=1):.4f}" for T in LENGTHS)
        P(f"{a:20s} {cells} | SOLVED {sum(solved[a].values())}/{len(seeds)} | final loss "
          f"{np.mean(list(fin[a].values())):.4f} ({min(fin[a].values()):.4f}-{max(fin[a].values()):.4f})")
    P("per seed T=128: " + "; ".join(f"{a} " + " ".join(f"{acc[128][a][s]:.4f}" for s in seeds) for a in ARMS))

    lab, det = primary_label(acc[128], solved, seeds, retrace)
    P("\n## PRIMARY: delta x phase interaction at T = 128")
    P(fmt("dI = PoPE-FreeDelta - PoPE-Flat (index)", det["dI"]))
    P(fmt("dP = MapPoPE-FreeDelta - MapPoPE-Flat (path)", det["dP"]))
    P(fmt("I = dP - dI (PAPER2X2 orientation)", det["I"]))
    P(f"path SOLVED pinned {det['solved_PP']}/8 vs free {det['solved_FP']}/8, Fisher p {det['fisher_path']:.4f}; "
      f"ub(dI) {det['ub_dI']:+.4f}; index helped {det['index_helped']}, path helped {det['path_helped']}, "
      f"path hurt {det['path_hurt']}")
    P(f"**PRIMARY: {lab}**")
    V = {"primary": lab, "details": {k: v for k, v in det.items()}, "seeds": seeds, "retrace": retrace}
    fl, fc = frozen_label(acc[128], solved, seeds)
    V["S1_frozen"] = fl; V["S1_details"] = fc
    p1 = p1_labels(acc[1024], seeds)
    V["P1"] = {X: lab for X, (lab, _c) in p1.items()}; V["P1_details"] = {X: c for X, (_l, c) in p1.items()}
    P("\n## REGISTERED SECONDARY P1 (Amendment 1): path arms vs MapPoPE-Flat at T = 1024 (robustness, rule 10), Holm over 2")
    for X, (lab, c) in p1.items():
        P(fmt(f"P1 {X} - MapPoPE-Flat, T=1024", c) + f"; Holm p {c['p_holm']:.4f} -> **{lab}**")
    P("declared beside P1 (descriptive): the same contrasts at T = 512, loss AUC (bimodal across PAPER2X2 seeds) and "
      "T = 128 held-out NLL")
    for X in (ZP, FP):
        P(fmt(f"  {X} - MapPoPE-Flat, T=512", contrast(acc[512][PP], acc[512][X], seeds)))
        P(fmt(f"  {X} - MapPoPE-Flat, loss AUC", contrast(auc[PP], auc[X], seeds)))
        P(fmt(f"  {X} - MapPoPE-Flat, T=128 NLL", contrast(nll[128][PP], nll[128][X], seeds)))
    if json_out:
        json.dump(V, open(json_out, "w"), indent=2, default=float)

    P("\n## Declared secondaries (no verdict depends on them)")
    P(fmt("S1 MapPoPE-FrozenDelta - MapPoPE-Flat", fc) + f"; SOLVED {fc['solved_arm']}/8 vs {fc['solved_ref']}/8 "
      f"Fisher p {fc['fisher']:.4f} -> {fl}")
    P(fmt("S1b MapPoPE-FrozenDelta - MapPoPE-FreeDelta", contrast(acc[128][FP], acc[128][ZP], seeds))
      + " [decay + learning vs neither: FP's delta also decays x0.447, ZP's does not]")
    m_below = {a: sum(acc[128][a][s] < retrace for s in seeds) for a in PATH_ARMS}
    m_above = {a: sum(acc[128][a][s] > retrace for s in seeds) for a in INDEX_ARMS}
    P(f"runs below the retrace floor (path arms): {m_below}; above it (index arms): {m_above}")
    P(f"I relative to PAPER2X2's score x phase interaction +0.154 (cross-batch, descriptive): I = {det['I']['mean']:+.4f}; "
      f"I <= -0.154 would mean freeing delta undoes at least the whole pinned interaction")

    P("\n### S3 out of distribution (trained at 128; robustness, rule 10). Holm over the 6 tests")
    rows = []
    for T in (512, 1024):
        for nm, (x, y) in (("dI", (PI, FI)), ("dP", (PP, FP))):
            rows.append((T, nm, contrast(acc[T][x], acc[T][y], seeds)))
        Iser = {s: (acc[T][FP][s] - acc[T][PP][s]) - (acc[T][FI][s] - acc[T][PI][s]) for s in seeds}
        rows.append((T, "I", contrast({s: 0.0 for s in seeds}, Iser, seeds, unpaired=False)))
    hp = holm([r[2]["p_sf"] for r in rows])
    for (T, nm, c), h in zip(rows, hp):
        P(fmt(f"T={T} {nm}", c) + f"; Holm p {h:.4f}")

    P("\n### S6 training: loss AUC (mean per-epoch loss), final-5% loss, run classes; r(final loss, acc) per length")
    for a in ARMS:
        cc = defaultdict(int)
        for s in seeds:
            cc[cls[a][s]] += 1
        P(f"{a:20s} AUC {np.mean(list(auc[a].values())):.4f} final {np.mean(list(fin[a].values())):.4f} classes {dict(cc)} "
          f"bimodality {bimodality([acc[128][a][s] for s in seeds], [solved[a][s] for s in seeds])}")
    for nm, (x, y) in (("AUC FI - PI", (PI, FI)), ("AUC FP - PP", (PP, FP)), ("AUC ZP - PP", (PP, ZP))):
        P(fmt(nm, contrast(auc[x], auc[y], seeds)))
    for T in LENGTHS:
        xs = [fin[a][s] for a in ARMS for s in seeds]; ys = [acc[T][a][s] for a in ARMS for s in seeds]
        P(f"r(final loss, acc) T={T}: {np.corrcoef(xs, ys)[0, 1]:+.3f} over {len(xs)} runs")

    W = {}
    if do_probe:
        P(f"\n### S2 / S4 / S5 probe (eval-only, CPU; T = 128 eval stream, {probe_trials} walks; layer 0 rebuilt with the "
          "effective delta)")
        pr = {}
        for a in ARMS:
            for s in seeds:
                m, _ = load_run(root, a, s)
                pr[(a, s)] = {**probe_run(m, s, trials=probe_trials), **kernel_checks(m, dfac)}
                d = abs(pr[(a, s)]["acc"] - acc[128][a][s])
                if pr[(a, s)]["verify"] > 1e-4 or d > 0.002:
                    P(f"WARNING {a} s{s}: rebuild max |dlogit| {pr[(a, s)]['verify']:.1e}, probe acc - eval acc {d:+.4f}")
        if any(pr[(ZP, s)].get("moved_abs_max", 1.0) != 0.0 for s in seeds):
            P("CODE CHECK FAILED: a frozen delta moved -- MapPoPE-FrozenDelta did not freeze")
        P(f"verify: max |dlogit| {max(v['verify'] for v in pr.values()):.1e}; max |probe acc - eval acc| "
          f"{max(abs(pr[(a, s)]['acc'] - acc[128][a][s]) for a in ARMS for s in seeds):.4f}")
        P("S2 accuracy by retrace stratum (mean over seeds; share of targets):")
        for a in ARMS:
            P(f"  {a:20s} " + "  ".join(f"{nm} {np.nanmean([pr[(a, s)][f'acc_{nm}'] for s in seeds]):.4f} "
                                        f"({np.mean([pr[(a, s)][f'share_{nm}'] for s in seeds]):.3f})"
                                        for nm in ("R1", "R2", "R3+", "none")))
        P("S2 dI by stratum (sign-flip, Holm over 4) and its share-weighted contribution to dI:")
        rs = []
        for nm in ("R1", "R2", "R3+", "none"):
            x = {s: pr[(PI, s)][f"acc_{nm}"] for s in seeds}; y = {s: pr[(FI, s)][f"acc_{nm}"] for s in seeds}
            contrib = np.mean([pr[(FI, s)][f"acc_{nm}"] * pr[(FI, s)][f"share_{nm}"]
                               - pr[(PI, s)][f"acc_{nm}"] * pr[(PI, s)][f"share_{nm}"] for s in seeds])
            rs.append((nm, contrast(x, y, seeds), contrib))
        for (nm, c, contrib), h in zip(rs, holm([r[1]["p_sf"] for r in rs])):
            P(fmt(f"  dI {nm}", c) + f"; Holm p {h:.4f}; contribution {contrib:+.4f}")
        P("S4 delta: dead-zone share (raw >= 0), re-pinned share (|wrap(delta_eff)| < 0.1), mean |wrap|, "
          "mean |delta - init|, share outside [-2pi, 0] (means over seeds):")
        for a in ARMS:
            g = lambda k: np.mean([pr[(a, s)].get(k, np.nan) for s in seeds])
            P(f"  {a:20s} deadzone {g('deadzone_share'):.3f} repinned {g('repinned_share'):.3f} |wrap| "
              f"{g('abs_wrapped_mean'):.3f} moved {g('moved_abs_mean'):.3f} outside {g('outside_clamp_share'):.3f}")
            if a in (FI, FP):
                q = np.mean([pr[(a, s)]["decay_resid_quartiles"] for s in seeds], 0)
                P(f"  {'':20s} decay-only reference: re-pinned {g('decay_only_repinned'):.3f} from this arm's own draws "
                  f"(batch-recipe analytic {DECAY_ONLY_REPINNED}); learned part |delta - {dfac:.4f} x init| by frequency quartile "
                  f"fast->slow " + " ".join(f"{x:.3f}" for x in q))
        P("S5 index kernel: share K(3) > K(1) per seed; argmax_D in 1..16 pooled:")
        for a in INDEX_ARMS:
            P(f"  {a:20s} " + " ".join(f"{pr[(a, s)]['frac_K3_gt_K1']:.2f}" for s in seeds) + " | argmax "
              + " ".join(f"{d}:{c}" for d, c in enumerate(np.sum([pr[(a, s)]['argmax_offset_hist'] for s in seeds], 0)) if c))
        P("S5 attention on the newest observation (offset 1, max over heads), index arms: "
          + "; ".join(f"{a} {np.mean([pr[(a, s)]['mass_off1'] for s in seeds]):.3f}" for a in INDEX_ARMS))
        P("S5 path target coherence C / C0 (best head, most recent same-cell observation key):")
        for a in PATH_ARMS:
            P(f"  {a:20s} C {np.mean([pr[(a, s)]['C'] for s in seeds]):.3f}  C0 {np.mean([pr[(a, s)]['C0'] for s in seeds]):.3f}"
              f"  same-cell mass {np.mean([pr[(a, s)]['mass_cell'] for s in seeds]):.3f}")
        P("\nS7 (Amendment 1, eval-only, diagnostic): PoPE-FreeDelta with delta reset to 0 on the fast half (c < 32) or the "
          "slow half (c >= 32); T = 128 accuracy and R3+ accuracy (means over seeds), and the share of dI each reset removes")
        s7 = {}
        for s in seeds:
            m, _ = load_run(root, FI, s)
            for half in ("fast", "slow"):
                s7[(half, s)] = probe_run(reset_half(m, half), s, trials=probe_trials)
        dI_mean = np.mean([pr[(FI, s)]["acc"] - pr[(PI, s)]["acc"] for s in seeds])
        for half in ("fast", "slow"):
            c = contrast({s: pr[(FI, s)]["acc"] for s in seeds}, {s: s7[(half, s)]["acc"] for s in seeds}, seeds,
                         unpaired=False)
            P(f"  reset {half:4s}: acc {np.mean([s7[(half, s)]['acc'] for s in seeds]):.4f}  R3+ "
              f"{np.nanmean([s7[(half, s)]['acc_R3+'] for s in seeds]):.4f}  (FI intact {np.mean([pr[(FI, s)]['acc'] for s in seeds]):.4f}"
              f" / R3+ {np.nanmean([pr[(FI, s)]['acc_R3+'] for s in seeds]):.4f}; PI {np.mean([pr[(PI, s)]['acc'] for s in seeds]):.4f})"
              f"; share of dI removed {(-c['mean'] / dI_mean) if dI_mean else float('nan'):+.2f}")
            P(fmt(f"  S7 reset {half} - FI intact", c))
        P("\nDropout-scale re-score (descriptive, Amendment 1 MINOR 4; o_proj input x 1/(1-p), DROPOUT_RESCORE's method), "
          "index arms:")
        rsc = {}
        for a in INDEX_ARMS:
            for s in seeds:
                m, _ = load_run(root, a, s)
                rsc[(a, s)] = probe_run(m, s, trials=probe_trials, rescale=1.0 / (1.0 - m.layers[0].dropout.p))["acc"]
        P(fmt("  dI re-scored", contrast({s: rsc[(PI, s)] for s in seeds}, {s: rsc[(FI, s)] for s in seeds}, seeds)))
        W = {f"{a}|{s}": v for (a, s), v in pr.items()}
        W.update({f"S7|{h}|{s}": v for (h, s), v in s7.items()})
        W.update({f"rescored|{a}|{s}": v for (a, s), v in rsc.items()})
    if json_out:
        json.dump({**V, "secondaries": W}, open(json_out, "w"), indent=2, default=float)
    return lab


# ------------------------------------------------------------------------------ selftest / power
_Z = np.array([-1.6, -1.0, -0.55, -0.15, 0.15, 0.55, 1.0, 1.6]); _Z = (_Z - _Z.mean()) / _Z.std(ddof=1)
_Z2 = np.array([0.15, -1.6, 1.0, -0.55, 1.6, -0.15, -1.0, 0.55]); _Z2 = (_Z2 - _Z2.mean()) / _Z2.std(ddof=1)


def _synth(mPI, mFI, mPP, mFP, sdI=0.03, sdP=0.001, kPP=8, kFP=8, mZP=None, kZP=8):
    """Deterministic synthetic seeds: every per-seed difference = mean + sd * a fixed standardised pattern."""
    S = list(range(8)); zp = mPP if mZP is None else mZP
    acc = {PI: {s: mPI + 0.05 * _Z2[s] for s in S}, FI: {s: mPI + 0.05 * _Z2[s] + (mFI - mPI) + sdI * _Z[s] for s in S},
           PP: {s: mPP for s in S}, FP: {s: mFP + sdP * _Z[s] for s in S}, ZP: {s: zp + sdP * _Z2[s] for s in S}}
    sol = {PI: {s: False for s in S}, FI: {s: False for s in S}, PP: {s: s < kPP for s in S},
           FP: {s: s < kFP for s in S}, ZP: {s: s < kZP for s in S}}
    return acc, sol, S


def selftest():
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got.startswith(want); ok &= good
        print(f"{'ok ' if good else 'BAD'} {name}: {got}")

    R = 0.934
    cases = [("unreadable: path did not learn", (0.68, 0.80, 0.90, 0.90), {}, UNR),
             ("unreadable: no index deficit", (0.95, 0.97, 0.999, 0.999), {}, UNR),
             ("hurts index", (0.68, 0.60, 0.999, 0.999), {}, f"{NS} (freeing delta hurts"),
             ("helps neither", (0.68, 0.685, 0.999, 0.999), {"sdI": 0.02}, f"{NS} (helps neither"),
             ("supported, path unchanged", (0.68, 0.80, 0.999, 0.999), {}, f"{SUP} (path unchanged"),
             ("supported, path cost", (0.68, 0.80, 0.999, 0.97), {"sdP": 0.005}, f"{SUP} (path cost"),
             ("supported, path cost by SOLVED", (0.68, 0.80, 0.999, 0.999), {"kFP": 1}, f"{SUP} (path cost"),
             ("helps both", (0.68, 0.80, 0.96, 0.99), {"sdP": 0.003}, f"{NS} (helps both"),
             ("helps both by SOLVED", (0.68, 0.80, 0.97, 0.97), {"kPP": 1, "kFP": 8}, f"{NS} (helps both"),
             ("unmeasured (index gain, noisy)", (0.68, 0.73, 0.999, 0.999), {"sdI": 0.08}, UNM)]
    for name, (a, b, c, d), kw, want in cases:
        acc, sol, S = _synth(a, b, c, d, **kw)
        chk(name, primary_label(acc, sol, S, R)[0], want)
    for name, kw, want in [("frozen costs", {"mZP": 0.95, "sdP": 0.005}, "COSTS"),
                           ("frozen costs by SOLVED", {"kZP": 1}, "COSTS"),
                           ("frozen no cost", {"mZP": 0.999}, "NO COST"),
                           ("frozen helps", {"mZP": 0.99, "sdP": 0.002, "_pp": 0.96}, "HELPS"),
                           ("frozen unmeasured", {"mZP": 0.962, "sdP": 0.02, "_pp": 0.97}, UNM)]:
        pp = kw.pop("_pp", 0.999)
        acc, sol, S = _synth(0.68, 0.80, pp, pp, **kw)
        chk(name, frozen_label(acc, sol, S)[0], want)
    # Amendment 1, MAJOR 1: a sub-noise ceiling gain on the path arm must not read "helps both"
    acc, sol, S = _synth(0.68, 0.80, 0.999, 0.999)
    for s in S:
        acc[PP][s] = 0.998 if s < 6 else 1.0; acc[FP][s] = 1.0
    chk("ceiling: FP 1.000 everywhere, PP 0.998 on six seeds", primary_label(acc, sol, S, R)[0], f"{SUP} (path unchanged")
    # P1 (T = 1024): every label
    for name, mz, sdz, want in (("P1 costs", 0.85, 0.02, P1_COST), ("P1 no OOD cost", 0.94, 0.005, P1_NONE),
                                ("P1 unmeasured", 0.92, 0.04, UNM)):
        a1 = {PP: {s: 0.94 + 0.01 * _Z2[s] for s in S}, ZP: {s: mz + sdz * _Z[s] for s in S},
              FP: {s: mz + sdz * _Z[s] for s in S}}
        got = p1_labels(a1, S)
        chk(name, got[ZP][0], want); chk(name + " (FreeDelta)", got[FP][0], want)
    # MINOR 2: NaN refused
    acc2 = {T: {a: {s: 0.5 for s in S} for a in ARMS} for T in LENGTHS}; acc2[128][FI][3] = float("nan")
    try:
        check_finite(acc2, S); good = False
    except SystemExit:
        good = True
    acc2[128][FI][3] = 0.5
    try:
        check_finite(acc2, S)
    except SystemExit:
        good = False
    ok &= good; print(f"{'ok ' if good else 'BAD'} non-finite accuracy refused, finite accepted")
    # MINOR 3: recipe assertion
    cfg0 = dict(RECIPES["batch"]); A0 = {"data_workers": 0, "fast_attn": False}
    blob = {"variant": FI, "seed": 1600, "losses": [1.0] * 300,
            "config": {**{k: v for k, v in cfg0.items() if k not in ("data_workers",)}, "args": A0}}
    try:
        check_recipe(blob, FI, 1600, "batch"); good = True
    except SystemExit:
        good = False
    for bad_blob in ({**blob, "config": {**blob["config"], "lr": 3e-4}}, {**blob, "seed": 1601},
                     {**blob, "losses": [1.0] * 299}, {**blob, "config": {**blob["config"], "args": {"data_workers": 3}}}):
        try:
            check_recipe(bad_blob, FI, 1600, "batch"); good = False
        except SystemExit:
            pass
    ok &= good; print(f"{'ok ' if good else 'BAD'} recipe assertion: the batch recipe passes, wrong lr / seed / epochs / workers fail")
    good = abs(decay_factor() - 0.4471) < 1e-4
    ok &= good; print(f"{'ok ' if good else 'BAD'} decay-only factor {decay_factor():.4f} (audit: 0.4471)")
    # sign-flip CI: the mean is inside; the ends are where the test starts rejecting
    d = np.random.default_rng(3).normal(0.05, 0.04, 8); lo, hi = sf_ci(d)
    good = lo <= d.mean() <= hi and signflip_p(d - hi - 0.002)["p"] < 0.10 and signflip_p(d - hi)["p"] >= 0.10
    ok &= good; print(f"{'ok ' if good else 'BAD'} sf_ci [{lo:+.4f}, {hi:+.4f}] around {d.mean():+.4f}")
    # code checks on tiny untrained models: pairing, rebuild, effective delta, kernel_probe agreement
    from mapformer.train_variant import VARIANT_MAP as VM
    from mapformer.model_pope import DELTA_MIN, DELTA_MAX
    cfg = dict(vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)

    def mk(v, s=7):
        torch.manual_seed(s); return VM[v](**cfg).eval()
    for pin, fr in ((PI, FI), (PP, FP), (PP, ZP)):
        a, b = mk(pin), mk(fr); sa, sb = a.state_dict(), b.state_dict()
        diff = [k for k in sa if not torch.equal(sa[k], sb[k])]
        good = diff == ["layers.0.pope_delta"] and set(sb) - set(sa) == {"layers.0.pope_delta_init"}
        ok &= good; print(f"{'ok ' if good else 'BAD'} pairing {pin} / {fr}: only delta differs ({diff})")
    good = torch.equal(mk(FI).layers[0].pope_delta, mk(ZP).layers[0].pope_delta)
    ok &= good; print(f"{'ok ' if good else 'BAD'} same delta draw in FI and ZP at a seed")
    torch.manual_seed(11); VM[PI](**cfg); x = torch.rand(5)
    torch.manual_seed(11); VM[FI](**cfg); y = torch.rand(5)
    good = torch.equal(x, y); ok &= good; print(f"{'ok ' if good else 'BAD'} building a free arm leaves the global RNG untouched")
    good = (not mk(ZP).layers[0].pope_delta.requires_grad) and mk(FP).layers[0].pope_delta.requires_grad
    ok &= good; print(f"{'ok ' if good else 'BAD'} frozen delta has requires_grad False, free True")
    tok = torch.randint(0, 21, (1, 60))
    for v in ARMS:
        m = mk(v)
        with torch.no_grad():
            if v in (PI, PP):
                m.layers[0].pope_delta.uniform_(-1.0, 1.0)          # half in the dead zone, so the clamp matters
        with torch.no_grad():
            err = float((forward_manual(m, tok)[0] - m(tok)).abs().max())
        good = err < 1e-4; ok &= good; print(f"{'ok ' if good else 'BAD'} rebuild {v}: max |dlogit| {err:.1e}")
    m = mk(PP)
    with torch.no_grad():
        m.layers[0].pope_delta.uniform_(-1.0, 1.0)
    good = torch.equal(delta_eff(m.layers[0]), m.layers[0].pope_delta.clamp(DELTA_MIN, DELTA_MAX))
    m2 = mk(FP)
    with torch.no_grad():
        m2.layers[0].pope_delta.fill_(0.7)
    good &= torch.equal(delta_eff(m2.layers[0]), m2.layers[0].pope_delta) and float(m2.layers[0].pope_delta.detach().max()) > 0
    ok &= good; print(f"{'ok ' if good else 'BAD'} delta_eff: clamp for pinned, raw for free")
    try:
        spec = importlib.util.spec_from_file_location("kernel_probe", KP_PATH); kp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kp)
        rng = np.random.default_rng(0)
        good = all(np.array_equal(kp.retrace_class(a), retrace_class(a))
                   for a in (rng.integers(0, 4, 200) for _ in range(20)))
        _, A, Q, K, ang = forward_manual(m, tok)
        tq, ts = np.array([20, 40, 58]), np.array([3, 11, 31])
        c1 = coherence(m, Q, K, ang, tq, ts); c2 = kp.coherence(m, Q, K, ang, tq, ts)
        good &= np.allclose(c1[0], c2[0]) and np.allclose(c1[1], c2[1])
        mi = mk(PI)
        with torch.no_grad():
            mi.layers[0].pope_delta.uniform_(-1.0, 1.0)
        good &= abs(kernel_checks(mi)["frac_K3_gt_K1"] - kp.content_kernel_checks(mi)["frac_K3_gt_K1"]) < 1e-12
        ok &= good; print(f"{'ok ' if good else 'BAD'} retrace_class / coherence / K(3)>K(1) equal kernel_probe's on pinned models")
    except Exception as e:                                       # noqa: BLE001
        ok = False; print(f"BAD kernel_probe comparison failed: {e!r}")
    # S7: the reset touches only the named half, and only delta
    mf = mk(FI); n = mf.layers[0].pope_delta.shape[-1]
    rf, rs = reset_half(mf, "fast"), reset_half(mf, "slow")
    d0 = mf.layers[0].pope_delta
    good = (torch.all(rf.layers[0].pope_delta[:, :n // 2] == 0) and torch.equal(rf.layers[0].pope_delta[:, n // 2:], d0[:, n // 2:])
            and torch.all(rs.layers[0].pope_delta[:, n // 2:] == 0) and torch.equal(rs.layers[0].pope_delta[:, :n // 2], d0[:, :n // 2])
            and bool(torch.all(d0 != 0)))
    sa, sb = mf.state_dict(), rf.state_dict()
    good &= [k for k in sa if not torch.equal(sa[k], sb[k])] == ["layers.0.pope_delta"]
    ok &= good; print(f"{'ok ' if good else 'BAD'} S7 reset_half zeroes only the named half of delta, original untouched")
    # rescale: 1.0 is the model's forward; 1/(1-p) multiplies o_proj's input
    with torch.no_grad():
        good = torch.equal(forward_manual(mf, tok, 1.0)[0], mf(tok))
        good &= not torch.equal(forward_manual(mf, tok, 1 / 0.9)[0], mf(tok))
    ok &= good; print(f"{'ok ' if good else 'BAD'} rescale 1.0 = model forward; 1/(1-p) differs")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def power(n_sim=2000):
    print("Primary power (synthetic): dI_s ~ N(m, sd), PP = 0.999, FP = PP + N(0, 0.0015) capped at 1, retrace 0.934")
    print(f"{'m':>6s} {'sd':>5s}  SUPPORTED  UNMEASURED  NOT SUPPORTED")
    rng = np.random.default_rng(0); S = list(range(8))
    for sd in (0.05, 0.08):
        for m in (0.0, 0.03, 0.05, 0.08, 0.126):
            cnt = defaultdict(int)
            for _ in range(n_sim):
                pi = 0.68 + rng.normal(0, 0.05, 8); fi = pi + rng.normal(m, sd, 8)
                pp = np.minimum(1, 0.999 + rng.normal(0, 0.0015, 8)); fp = np.minimum(1, pp + rng.normal(0, 0.0015, 8))
                acc = {PI: dict(zip(S, pi)), FI: dict(zip(S, fi)), PP: dict(zip(S, pp)), FP: dict(zip(S, fp))}
                sol = {PP: {s: True for s in S}, FP: {s: True for s in S}}
                lab = primary_label(acc, sol, S, 0.934)[0]
                cnt[lab.split(" (")[0]] += 1
            print(f"{m:6.3f} {sd:5.2f}  {cnt[SUP] / n_sim:9.3f}  {cnt[UNM] / n_sim:10.3f}  {cnt[NS] / n_sim:13.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/delta_free")
    ap.add_argument("--raw", default=f"{REPO}/_DELTA_FREE_RAW.json")
    ap.add_argument("--seeds", default="1600-1607")
    ap.add_argument("--floors", default=f"{REPO}/docs/audits/2026-10-20/delta_free_floors_out.txt")
    ap.add_argument("--json-out", default=f"{REPO}/DELTA_FREE_VERDICTS.json")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--recipe", default="batch", choices=sorted(RECIPES), help="recipe asserted on every checkpoint")
    ap.add_argument("--probe-trials", type=int, default=100, help="= the eval's --n-trials (the probe re-scores its stream)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--power", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.power:
        power(); return
    analyse(a.root, a.raw, parse_seeds(a.seeds), a.floors, None if a.no_json else a.json_out, not a.no_probe,
            a.probe_trials, a.recipe)


if __name__ == "__main__":
    main()
