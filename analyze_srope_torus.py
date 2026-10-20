"""Registered analysis for SROPE_TORUS_PREREG.md: does faithful Selective RoPE learn the 2D torus map? CPU.
  python3 -m mapformer.analyze_srope_torus [--root runs/srope_torus] [--eval-dir REPO] [--seeds 1700-1707] [--json-out P | --no-json]
  python3 -m mapformer.analyze_srope_torus --selftest     # every label and reading from synthetic inputs (no run dirs)
Layout: <root>/<arm>/p0/<V>_s<S>/<V>.pt (train_variant, the GENERALITY torus recipe), evals SROPE_TORUS_<arm>.json
(eval_noise_refine, T 1024 / 2048) and SROPE_TORUS_<arm>_STRATA.json (eval_rank_strata, T 1024) in --eval-dir.
SOLVED = stats_core.classify_run (final 5% of training loss < 0.05); Fisher exact on SOLVED, unpaired permutation on accuracy.
"""
import argparse, json, os

import numpy as np
import torch

from mapformer.stats_core import perm2_p, perm2_ci, fisher_solved, classify_run, signflip_p
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"
ARMS = {"SRoPE": "SRoPE", "SRoPE_NoSiLU": "SRoPE_NoSiLU", "SRoPE_ST": "SRoPE_ST", "SRoPEGen": "SRoPEGen", "MapWM": "Signed_r4",
        "RoPE": "RoPE"}                                          # SRoPE_ST: Amendment 1
CFG = {"d_model": 128, "n_layers": 1, "n_heads": 2, "n_steps": 1024, "lr": 1e-3, "batch_size": 16, "n_batches": 98,
       "schedule": "cosine", "fast_attn": True, "grid_size": 64, "n_obs_types": 16, "n_landmarks": 0, "init_from": None,
       "data_seed_offset": 0, "deterministic": False, "p_empty": 0.5}
FLOORS = {1024: {"blank": 0.5087, "ngram": 0.5779, "retrace": 0.8454}, 2048: {"blank": 0.5086, "ngram": 0.5663, "retrace": 0.7913}}
LEARNS, FAILS, UNM = "LEARNS THE MAP", "FAILS", "UNMEASURED"
BEATS, WORSE, NDIST = "BEATS INDEX", "WORSE THAN INDEX", "NOT DISTINGUISHED"
N_ACT, BLANK = 4, 20


def map_label(acc_x, acc_ref, k_x, k_ref, n):
    """S1 rule (arm x vs the in-batch MapWM reference). Gate first: the reference must solve >= 6/n."""
    acc_x, acc_ref = np.asarray(acc_x, float), np.asarray(acc_ref, float)
    pf = fisher_solved(k_x, n, k_ref, n); pp = perm2_p(acc_ref, acc_x)["p"]; d = float(acc_x.mean() - acc_ref.mean())
    if k_ref < 6:
        lab = f"UNREADABLE (MapWM SOLVED {k_ref} < 6)"
    elif (pf < 0.05 and k_ref > k_x) or (pp < 0.05 and d <= -0.05):
        lab = FAILS
    elif k_x >= 6 and d >= -0.01:
        lab = LEARNS
    else:
        lab = UNM
    return lab, {"d": d, "perm_p": pp, "fisher_p": pf}


def index_label(acc_x, acc_idx, k_x, k_idx, n):
    """S2 rule (arm x vs in-batch index RoPE)."""
    acc_x, acc_idx = np.asarray(acc_x, float), np.asarray(acc_idx, float)
    pf = fisher_solved(k_x, n, k_idx, n); pp = perm2_p(acc_idx, acc_x)["p"]; d = float(acc_x.mean() - acc_idx.mean())
    if (pp < 0.05 and d > 0) or (pf < 0.05 and k_x > k_idx):
        lab = BEATS
    elif (pp < 0.05 and d < 0) or (pf < 0.05 and k_idx > k_x):
        lab = WORSE
    else:
        lab = NDIST
    return lab, {"d": d, "perm_p": pp, "fisher_p": pf}


def silu_label(acc_nos, acc_s, k_nos, k_s, n):
    """Secondary: SRoPE_NoSiLU vs SRoPE (d = NoSiLU - SRoPE)."""
    acc_nos, acc_s = np.asarray(acc_nos, float), np.asarray(acc_s, float)
    pf = fisher_solved(k_nos, n, k_s, n); pp = perm2_p(acc_s, acc_nos)["p"]; d = float(acc_nos.mean() - acc_s.mean())
    if (pf < 0.05 and k_nos > k_s) or (pp < 0.05 and d >= 0.05):
        lab = "SiLU COSTS"
    elif (pf < 0.05 and k_s > k_nos) or (pp < 0.05 and d <= -0.05):
        lab = "SiLU HELPS"
    else:
        lab = "NO DIFFERENCE SHOWN"
    return lab, {"d": d, "perm_p": pp, "fisher_p": pf}


def phase_label(intact, ablated):
    """Amendment 1 (eval-only): does the arm use its phase? Zero the angle (theta = 0) in the trained model and re-score the
    same walks. USES ITS PHASE if mean(ablated - intact) <= -0.05 with sign-flip p < .05 (paired over seeds)."""
    dd = np.asarray(ablated, float) - np.asarray(intact, float); p = signflip_p(dd)["p"]
    return ("USES ITS PHASE" if dd.mean() <= -0.05 and p < 0.05 else "PHASE NOT SHOWN USED"), {"d": float(dd.mean()), "p": p}


def reading(s1, s2, phase="USES ITS PHASE"):
    """Overall reading, fixed in SROPE_TORUS_PREREG.md (phase branch: Amendment 1)."""
    if s1.startswith("UNREADABLE"):
        return "UNREADABLE (the in-batch MapWM reference failed its gate)"
    if s1 == LEARNS:
        return "SELECTIVE ROPE (BEST LM CONFIG) LEARNS THE TORUS MAP (the prediction holds)"
    if s1 == FAILS:
        if s2 == BEATS:
            return ("SELECTIVE ROPE (BEST LM CONFIG) FAILS THE MAP BUT USES ITS CONTENT PHASE (beats index RoPE; zeroing its angle costs)"
                    if phase == "USES ITS PHASE" else
                    "SELECTIVE ROPE (BEST LM CONFIG) FAILS THE MAP; IT BEATS INDEX ROPE WITHOUT A SHOWN USE OF ITS PHASE")
        return (
                f"SELECTIVE ROPE (BEST LM CONFIG) FAILS THE MAP AND IS {'WORSE THAN' if s2 == WORSE else 'NOT DISTINGUISHED FROM'} INDEX ROPE")
    return f"UNMEASURED (S1 neither learns nor fails at n = 8; S2 {s2})"


def selftest():
    n = 8
    one, lo, mid = [1.0] * n, [0.80] * n, [0.995, 0.999, 1.0, 0.985, 1.0, 0.99, 0.996, 1.0]
    rope = [0.72, 0.75, 0.70, 0.73, 0.73, 0.70, 0.73, 0.72]
    c = {"LEARNS": map_label(mid, one, 7, 8, n)[0], "FAILS_F": map_label(one, one, 2, 8, n)[0],
         "FAILS_P": map_label(lo, one, 5, 6, n)[0], "UNM": map_label([0.97] * 4 + [1.0] * 4, one, 5, 8, n)[0],
         "UNR": map_label(one, one, 8, 5, n)[0], "LEARNS_BETTER": map_label(one, lo, 8, 6, n)[0]}
    for k, v in c.items():
        print(f"selftest S1 {k}: {v}")
    assert c["LEARNS"] == LEARNS and c["FAILS_F"] == FAILS and c["FAILS_P"] == FAILS and c["UNM"] == UNM
    assert c["UNR"].startswith("UNREADABLE") and c["LEARNS_BETTER"] == LEARNS
    i = {"BEATS": index_label(lo, rope, 0, 0, n)[0], "WORSE": index_label([0.6] * n, rope, 0, 0, n)[0],
         "NDIST": index_label(rope[::-1], rope, 0, 0, n)[0], "BEATS_F": index_label(rope, rope, 5, 0, n)[0]}
    for k, v in i.items():
        print(f"selftest S2 {k}: {v}")
    assert i["BEATS"] == BEATS and i["WORSE"] == WORSE and i["NDIST"] == NDIST and i["BEATS_F"] == BEATS
    s = {"COSTS": silu_label(one, lo, 8, 2, n)[0], "HELPS": silu_label(lo, one, 2, 8, n)[0],
         "NONE": silu_label(mid, one, 7, 8, n)[0]}
    for k, v in s.items():
        print(f"selftest SiLU {k}: {v}")
    assert s["COSTS"] == "SiLU COSTS" and s["HELPS"] == "SiLU HELPS" and s["NONE"] == "NO DIFFERENCE SHOWN"
    ph = {"USED": phase_label([0.80] * n, [0.70, 0.71, 0.69, 0.72, 0.70, 0.68, 0.71, 0.70])[0],
          "NOT": phase_label([0.80] * n, [0.79, 0.81, 0.80, 0.80, 0.78, 0.80, 0.81, 0.79])[0]}
    print(f"selftest phase: {ph}")
    assert ph["USED"] == "USES ITS PHASE" and ph["NOT"] == "PHASE NOT SHOWN USED"
    for a, b, f in ((LEARNS, BEATS, ph["USED"]), (FAILS, BEATS, ph["USED"]), (FAILS, BEATS, ph["NOT"]), (FAILS, NDIST, ph["NOT"]),
                    (FAILS, WORSE, ph["NOT"]), (UNM, BEATS, ph["USED"]), (c["UNR"], BEATS, ph["USED"])):
        print(f"selftest reading ({a[:10]}, {b}, {f}): {reading(a, b, f)}")
    print("selftest passed: every S1 / S2 / SiLU / phase label and every reading reached")


# ------------------------------------------------------------------------------------------------- angle statistics
@torch.no_grad()
def layer1_angles(m, tok):
    """Layer-1 rotary angle (B, T, C) with C = heads x blocks in channel order h * nb + c; None for index RoPE."""
    if hasattr(m, "srope"):
        th = m.srope[0](m.layers[0].norm1(m.token_emb(tok)))
    elif hasattr(m, "angle"):                                            # SRoPEGen (model_selective)
        th = m.angle(m.token_emb(tok))
    elif hasattr(m, "path_integrator"):
        th = torch.cumsum(m.action_to_lie(m.token_emb(tok)), 1) * m.path_integrator.omega
    else:
        return None
    return th.reshape(th.shape[0], th.shape[1], -1).double()


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.no_grad()
def angle_stats(m, seed, n_walks=8, T=1024, n_heads=2, big=0.05, tol=0.1, min_step=1e-3):
    """Declared secondary readouts on n_walks eval-stream walks (held-out map, env seed 10000, np seed 1234 + seed); layer-1
    angle theta, per-position increment = first difference of theta. Token-class versions:
    neg_share   per token class (moves 0-3, observation, blank): share of |increment| mass carried by negative increments, over
                channels whose mean |increment| at that class exceeds min_step (1e-3 rad);
    obs_ratio   mean |increment| at observation tokens / at move tokens, over channels where some move's mean step has
                |wrap| > big;
    ctx_cv      median over (move, selected channel) of std / |mean| of that move's increment across positions (0 = context-free).
    Pair versions (Amendment 1; robust to the conv placing a move's step one token late): the increment of the (action k,
    observation k) pair = theta at observation k - theta at observation k-1, keyed by action k:
    cancel      wrap-aware cancellation of opposite moves (pairs (0,1), (2,3)) on mean pair increments: over channels with
                |wrap(mean a)| > big, share with |wrap(mean a + mean b)| < tol (also reported on token-class means: cancel_tok);
    eff_rank    per head, participation ratio (sum s^2)^2 / sum s^4 of the singular values s of the 4 x nb matrix of mean pair
                increments (NOT wrap-aware: a wrapping solution reads higher); top2 = top-2 energy share."""
    from mapformer.environment import GridWorld
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    np.random.seed(1234 + seed)
    tok = torch.stack([env.generate_trajectory(T)[0] for _ in range(n_walks)])
    m.eval()
    th = layer1_angles(m, tok)
    if th is None:
        return None
    inc = torch.cat([th[:, :1], th[:, 1:] - th[:, :-1]], 1)            # (B, 2T, C) per-position increment
    t = tok.reshape(-1); I = inc.reshape(-1, inc.shape[-1])
    classes = {f"move{a}": t == a for a in range(N_ACT)}
    classes["obs"] = (t >= N_ACT) & (t != BLANK); classes["blank"] = t == BLANK
    out = {"neg_share": {}}
    for k, msk in classes.items():
        X = I[msk]
        if X.shape[0] == 0:
            continue
        sel = X.abs().mean(0) > min_step
        Xs = X[:, sel]
        out["neg_share"][k] = float(Xs.clamp(max=0).abs().sum() / Xs.abs().sum()) if Xs.numel() and Xs.abs().sum() > 0 else float("nan")
    M = torch.stack([I[classes[f"move{a}"]].mean(0) for a in range(N_ACT)])   # (4, C) token-class means
    S = torch.stack([I[classes[f"move{a}"]].std(0) for a in range(N_ACT)])
    ob_th = th[:, 1::2]                                                 # theta at observation k (B, T, C)
    pinc = torch.cat([ob_th[:, :1], ob_th[:, 1:] - ob_th[:, :-1]], 1).reshape(-1, th.shape[-1])
    act = tok[:, 0::2].reshape(-1)
    P = torch.stack([pinc[act == a].mean(0) for a in range(N_ACT)])       # (4, C) pair means

    def cancel(Mx):
        ws = []
        for i, j in ((0, 1), (2, 3)):
            sel = wrap(Mx[i]).abs() > big
            ws.append(wrap(Mx[i] + Mx[j])[sel].abs())
        w = torch.cat(ws)
        return float((w < tol).double().mean()) if w.numel() else float("nan")
    out["cancel"], out["cancel_tok"] = cancel(P), cancel(M)
    cv = [(S[i] / M[i].abs())[wrap(M[i]).abs() > big] for i in range(N_ACT)]
    cvv = torch.cat(cv); out["ctx_cv"] = float(cvv.median()) if cvv.numel() else float("nan")
    anysel = (wrap(M).abs() > big).any(0)
    mv = I[torch.stack([classes[f"move{a}"] for a in range(N_ACT)]).any(0)][:, anysel].abs().mean()
    ob = I[classes["obs"]][:, anysel].abs().mean()
    out["obs_ratio"] = float(ob / mv) if anysel.any() else float("nan")
    er, t2 = [], []
    for Mh in P.reshape(N_ACT, n_heads, -1).transpose(0, 1):
        s2 = torch.linalg.svdvals(Mh) ** 2
        er.append(float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else float("nan"))
        t2.append(float(s2[:2].sum() / s2.sum()) if s2.sum() > 0 else float("nan"))
    out["eff_rank"], out["top2"] = er, t2
    return out


@torch.no_grad()
def zero_angle_acc(m, seed, n_trials, T=1024):
    """Amendment 1: accuracy with the rotary angle forced to 0 (SRoPE*: temperature 0; SRoPEGen: exp(log_temp) = 0; MapWM:
    omega 0), on eval_noise_refine's exact stream (held-out map, env seed 10000, np seed 1234 + seed, n_trials walks; with
    n_trials = 100 these are the walks behind the intact accuracy). CPU, explicit attention path."""
    import copy
    from mapformer.environment import GridWorld
    from mapformer.eval_noise_refine import evaluate
    z = copy.deepcopy(m).eval()
    if hasattr(z, "srope"):
        for g in z.srope:
            g.temperature.zero_()
    elif hasattr(z, "angle"):
        z.angle.log_temp.fill_(float("-inf"))
    elif hasattr(z, "path_integrator"):
        z.path_integrator.omega.zero_()
    else:
        return None
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    return evaluate(z, env, T, n_trials, 0.0, "cpu", seed=1234 + seed)[0]


# ------------------------------------------------------------------------------------------------------- loading
def load_runs(root, ev_dir, seeds, epochs, n_batches=98, stats_walks=8, ablate_trials=100):
    R = {}; want = {**CFG, "n_batches": n_batches}
    for arm, V in ARMS.items():
        ev = json.load(open(f"{ev_dir}/SROPE_TORUS_{arm}.json"))
        st_f = f"{ev_dir}/SROPE_TORUS_{arm}_STRATA.json"; st = json.load(open(st_f)) if os.path.exists(st_f) else {}
        for s in seeds:
            b = torch.load(f"{root}/{arm}/p0/{V}_s{s}/{V}.pt", map_location="cpu", weights_only=False)
            c = b["config"]
            bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
            if c["args"].get("data_workers") != 3:
                bad["data_workers"] = (c["args"].get("data_workers"), 3)
            assert not bad and b["variant"] == V and b["seed"] == s and c["epochs"] == epochs and len(b["losses"]) == epochs, \
                (arm, s, bad, c["epochs"], len(b["losses"]))
            acc = {T: dict((x[0], x[1]) for x in ev[f"0.0|{V}|{T}"])[s] for T in (1024, 2048)}
            k = classify_run(b["losses"]); k10 = classify_run(b["losses"], solved_loss=0.1)
            m = VARIANT_MAP[V](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                               grid_size=c["grid_size"])
            m.load_state_dict(b["model_state_dict"])
            R[arm, s] = {"acc": acc[1024], "acc2": acc[2048], "solved": k["registered"] == "SOLVED",
                         "solved10": k10["registered"] == "SOLVED", "cls": k["cls"], "auc": float(np.mean(b["losses"])),
                         "tail": float(k["tail"]), "strata": st.get(f"{V}|{s}|1024"),
                         "angles": angle_stats(m, s, n_walks=stats_walks) if stats_walks else None,
                         "acc_zero": zero_angle_acc(m, s, ablate_trials) if ablate_trials else None}
    return R


def contrast(name, fn, R, x, y, seeds, n):
    ax = np.array([R[x, s]["acc"] for s in seeds]); ay = np.array([R[y, s]["acc"] for s in seeds])
    kx = sum(R[x, s]["solved"] for s in seeds); ky = sum(R[y, s]["solved"] for s in seeds)
    lab, st = fn(ax, ay, kx, ky, n)
    ci = perm2_ci(ay, ax, step=0.005, grid=(-1.0, 1.0))
    f3 = lambda v: "n/a" if v is None else f"{v:+.3f}"             # perm2_ci returns None when the CI is degenerate
    print(f"{name}: {lab} -- {x} {ax.mean():.4f} SOLVED {kx}/{n} vs {y} {ay.mean():.4f} SOLVED {ky}/{n}; d {st['d']:+.4f} "
          f"[95% CI {f3(ci['lo'])}, {f3(ci['hi'])}] (perm p {st['perm_p']:.4f}); Fisher p {st['fisher_p']:.4f}")
    return {"label": lab, f"{x}_solved": int(kx), f"{y}_solved": int(ky), f"{x}_acc": float(ax.mean()),
            f"{y}_acc": float(ay.mean()), **st, "ci": [ci["lo"], ci["hi"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/srope_torus")
    ap.add_argument("--eval-dir", default=REPO)
    ap.add_argument("--seeds", default="1700-1707")
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-batches", type=int, default=98, help="smoke only")
    ap.add_argument("--stats-walks", type=int, default=8, help="walks for the angle statistics (0 = skip; smoke)")
    ap.add_argument("--ablate-trials", type=int, default=100, help="walks for the zero-angle ablation (smoke: fewer)")
    ap.add_argument("--json-out", default=f"{REPO}/SROPE_TORUS_VERDICTS.json")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    R = load_runs(a.root, a.eval_dir, seeds, a.epochs, a.n_batches, a.stats_walks, a.ablate_trials)

    print(f"== torus 64x64, T = 1024, {a.epochs} epochs; per seed: accuracy at T = 1024 and run class ==")
    print("seed | " + " | ".join(ARMS))
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{R[x, s]['acc']:.4f} {R[x, s]['cls'][:4]}" for x in ARMS))
    print(f"floors (own eval stream, seeds {seeds[0]}-{seeds[-1]}, T = 1024): blank {FLOORS[1024]['blank']}, n-gram "
          f"{FLOORS[1024]['ngram']}, retrace {FLOORS[1024]['retrace']} (best trivial predictor)")

    print("\n== registered contrasts ==")
    V = {"primary": {}, "secondary": {}}
    V["primary"]["S1"] = contrast("S1 SRoPE vs MapWM (Signed_r4)", map_label, R, "SRoPE", "MapWM", seeds, n)
    V["primary"]["S2"] = contrast("S2 SRoPE vs index RoPE", index_label, R, "SRoPE", "RoPE", seeds, n)
    if a.ablate_trials != 100:
        print(f"NOTE: zero-angle ablation on {a.ablate_trials} walks, not the 100 behind the intact accuracy (smoke only)")
    ph, pst = phase_label([R["SRoPE", s]["acc"] for s in seeds], [R["SRoPE", s]["acc_zero"] for s in seeds])
    V["primary"]["phase"] = {"label": ph, **pst, "intact": float(np.mean([R["SRoPE", s]["acc"] for s in seeds])),
                             "zero_angle": float(np.mean([R["SRoPE", s]["acc_zero"] for s in seeds]))}
    print(f"phase (eval-only, Amendment 1) SRoPE: {ph} -- intact {V['primary']['phase']['intact']:.4f} vs zero angle "
          f"{V['primary']['phase']['zero_angle']:.4f}; d {pst['d']:+.4f} (sign-flip p {pst['p']:.4f})")
    V["reading"] = reading(V["primary"]["S1"]["label"], V["primary"]["S2"]["label"], ph)
    print(f"\nReading: {V['reading']}")
    print("\n== declared secondary contrasts (labelled, no family-wise correction; not part of the reading) ==")
    V["secondary"]["SiLU"] = contrast("SiLU: SRoPE_NoSiLU vs SRoPE", silu_label, R, "SRoPE_NoSiLU", "SRoPE", seeds, n)
    V["secondary"]["NoSiLU_vs_MapWM"] = contrast("SRoPE_NoSiLU vs MapWM", map_label, R, "SRoPE_NoSiLU", "MapWM", seeds, n)
    V["secondary"]["SRoPEGen_vs_MapWM"] = contrast("SRoPEGen vs MapWM", map_label, R, "SRoPEGen", "MapWM", seeds, n)
    V["secondary"]["SRoPEGen_vs_RoPE"] = contrast("SRoPEGen vs index RoPE", index_label, R, "SRoPEGen", "RoPE", seeds, n)
    V["secondary"]["NoSiLU_vs_RoPE"] = contrast("SRoPE_NoSiLU vs index RoPE", index_label, R, "SRoPE_NoSiLU", "RoPE", seeds, n)
    V["secondary"]["ST_vs_MapWM"] = contrast("SRoPE_ST vs MapWM", map_label, R, "SRoPE_ST", "MapWM", seeds, n)
    V["secondary"]["ST_vs_RoPE"] = contrast("SRoPE_ST vs index RoPE", index_label, R, "SRoPE_ST", "RoPE", seeds, n)
    for x in ("SRoPE_NoSiLU", "SRoPE_ST", "SRoPEGen", "MapWM"):
        lab, st = phase_label([R[x, s]["acc"] for s in seeds], [R[x, s]["acc_zero"] for s in seeds])
        V["secondary"][f"phase_{x}"] = {"label": lab, **st}
        print(f"phase {x}: {lab} -- d {st['d']:+.4f} (sign-flip p {st['p']:.4f})")
    if not a.no_json:                                       # verdicts written before any descriptive readout can fail
        json.dump(V, open(a.json_out, "w"), indent=2)

    print("\n== declared secondary readouts (descriptive) ==")
    for x in ARMS:
        k10 = sum(R[x, s]["solved10"] for s in seeds)
        print(f"{x}: acc T 2048 (OOD) {np.mean([R[x, s]['acc2'] for s in seeds]):.4f} (floor retrace {FLOORS[2048]['retrace']}); "
              f"loss AUC {np.mean([R[x, s]['auc'] for s in seeds]):.3f}; SOLVED at loss 0.1 {k10}/{n}; classes "
              f"{' '.join(R[x, s]['cls'][:4] for s in seeds)}")
        rows = [R[x, s]["strata"] for s in seeds if R[x, s]["strata"]]
        if rows:
            print("   strata: " + "; ".join(
                f"{k} {np.mean([r[k]['acc'] for r in rows if r[k]['acc'] is not None]):.3f}"
                for k in ("plain_lag<128", "plain_lag>=128", "wrap") if any(r[k]['acc'] is not None for r in rows)))
        z = [R[x, s]["acc_zero"] for s in seeds if R[x, s]["acc_zero"] is not None]
        if z:
            print(f"   zero-angle accuracy {np.mean(z):.4f} (intact {np.mean([R[x, s]['acc'] for s in seeds]):.4f})")
        A = [R[x, s]["angles"] for s in seeds if R[x, s]["angles"]]
        if A:
            neg = {k: np.nanmean([q["neg_share"].get(k, np.nan) for q in A]) for k in A[0]["neg_share"]}
            print("   angles: negative-mass share " + ", ".join(f"{k} {v:.2f}" for k, v in neg.items())
                  + f"; wrap-aware cancel share (pair) {np.nanmean([q['cancel'] for q in A]):.2f} (token-class "
                  f"{np.nanmean([q['cancel_tok'] for q in A]):.2f}); context CV "
                  f"{np.nanmedian([q['ctx_cv'] for q in A]):.3f}; obs/move |step| {np.nanmean([q['obs_ratio'] for q in A]):.3f}; "
                  f"pair move-map participation ratio per head {np.nanmean([q['eff_rank'] for q in A], 0).round(2).tolist()} "
                  f"(top-2 energy {np.nanmean([q['top2'] for q in A], 0).round(3).tolist()})")
    for x, y in (("SRoPE", "MapWM"), ("SRoPE_NoSiLU", "SRoPE")):
        k10x = sum(R[x, s]["solved10"] for s in seeds); k10y = sum(R[y, s]["solved10"] for s in seeds)
        print(f"threshold sensitivity {x} vs {y}: SOLVED at loss 0.1 {k10x}/{n} vs {k10y}/{n} (Fisher p "
              f"{fisher_solved(k10x, n, k10y, n):.4f})")


if __name__ == "__main__":
    main()
