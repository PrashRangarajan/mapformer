"""Rule 14 existence for the G-1 (RANKCLOCK) arms: does each class CONTAIN a torus solution, and does training HOLD it?

A failure of an arm in the batch is evidence about SEARCH only if its class can represent a solution and keep one under
training at the batch's learning rate (rule 14: frozen AND trainable; the clock slots move a top-channel phase by ~12.9 rad
per Adam step after 2048 tokens, so the trainable half is not a formality -- Amendment 1 finding 3).

Source: a TRAINED shared rank-4 model (`Vanilla_r4mi`, pilot seed 1990, outside the batch). Every construction keeps the
source's content weights and omega and rebuilds only the step map. Candidate step tables (V x C):
  r2proj  best rank-2 approximation of the source table S (SVD)
  slot    rank-2 approximation of S - 1 c^T, plus c in the per-token slot; c = the FREQUENCY-WEIGHTED mean step over the
          token stream (measured on fresh trajectories), not the plain vocabulary average (Amendment 1 finding 6)
  xy      move rows minus their axis slot c_x = (S(+x) + S(-x)) / 2, c_y likewise, at rank 2, plus the two axis slots
  AO      the four move rows of S at rank 2, observation steps 0
  DF      moves antisymmetrised per inverse pair, observation steps 0
A class is scored by the BEST frozen construction it can represent (finding 6: DF's table is also representable in r2 and
AO, so a spurious failure of one construction cannot sink a class):
  r2: r2proj, AO, DF      slot: + slot      slotxy: + xy      AO: AO, DF      DF: DF
Each class's best construction is then saved as a train_variant checkpoint for the trainable warm start
(`--warmstart-dir`), which the driver of this check (`run_rankclock_existence.sh`) trains for 60 cosine epochs at lr 1e-3.

  cd /home/prashr && python3 -m mapformer.construct_rankclock [--device cuda:1] [--warmstart-dir DIR]
"""
import argparse
import copy
import json
import math
import os

import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.eval_noise_refine import evaluate
from mapformer.model_rankclock import TORUS_ACTIONS, TORUS_PAIRS
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"
SOURCE = f"{REPO}/runs/rankclock_pilot/Vanilla_r4mi_s1990/Vanilla_r4mi.pt"
CLASSES = {"r2": ("Vanilla", ("r2proj", "AO", "DF")),
           "slot": ("Vanilla_r2clock", ("r2proj", "AO", "DF", "slot")),
           "slotxy": ("Vanilla_r2clockxy", ("r2proj", "AO", "DF", "xy")),
           "AO": ("Vanilla_r2AO", ("AO", "DF")),
           "DF": ("Vanilla_r2DF", ("DF",))}


def load(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    c = b["config"]
    m = VARIANT_MAP[b["variant"]](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                                   n_layers=c["n_layers"], grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"])
    return m.eval(), b


@torch.no_grad()
def step_table(m):
    st = m.step_table() if hasattr(m, "step_table") else m.action_to_lie(m.token_emb.weight[None])[0]
    return st.reshape(st.shape[0], -1).double()


def token_freq(V, n_traj=200, T=1024, seed=10000):
    """Frequency of each token id in the training stream (fresh trajectories; numpy global RNG, as the env draws)."""
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=seed)
    np.random.seed(seed)
    cnt = np.zeros(V)
    for _ in range(n_traj):
        tok, _om, _rev = env.generate_trajectory(T)
        cnt += np.bincount(tok.numpy(), minlength=V)[:V]
    return torch.tensor(cnt / cnt.sum(), dtype=torch.float64)


def tables(S, freq):
    """Candidate step tables, each as (table, slot dict)."""
    out = {}
    V, C = S.shape

    def r2(T, rows):
        U, s, Vt = torch.linalg.svd(T[rows], full_matrices=False)
        A = torch.zeros(V, C, dtype=torch.float64)
        A[rows] = (U[:, :2] * s[:2]) @ Vt[:2]
        return A

    allrows = list(range(V))
    out["r2proj"] = (r2(S, allrows), {})
    c = (freq[:, None] * S).sum(0)
    out["slot"] = (r2(S - c[None], allrows) + c[None], {"clock": c})
    cx, cy = (S[TORUS_PAIRS[0][0]] + S[TORUS_PAIRS[0][1]]) / 2, (S[TORUS_PAIRS[1][0]] + S[TORUS_PAIRS[1][1]]) / 2
    Sm = S.clone()
    for i in TORUS_ACTIONS[:2]:
        Sm[i] = S[i] - cx
    for i in TORUS_ACTIONS[2:]:
        Sm[i] = S[i] - cy
    T_xy = r2(Sm, allrows)
    for i in TORUS_ACTIONS[:2]:
        T_xy[i] = T_xy[i] + cx
    for i in TORUS_ACTIONS[2:]:
        T_xy[i] = T_xy[i] + cy
    out["xy"] = (T_xy, {"clock_x": cx, "clock_y": cy})
    out["AO"] = (r2(S, list(TORUS_ACTIONS)), {})
    T_df = torch.zeros(V, C, dtype=torch.float64)
    for p, q in TORUS_PAIRS:
        d = (S[p] - S[q]) / 2
        T_df[p], T_df[q] = d, -d
    out["DF"] = (T_df, {})
    return out


def realise(cls, variant, table, slots, src, b):
    """Build `variant` with the source's content and a step map realising `table` exactly (exact where the class can)."""
    c = b["config"]
    m = VARIANT_MAP[variant](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                             n_layers=c["n_layers"], grid_size=c["grid_size"])
    sd = m.state_dict()
    for k, v in src.state_dict().items():
        if k in sd and not k.startswith("action_to_lie.") and sd[k].shape == v.shape:
            sd[k] = v.clone()
    m.load_state_dict(sd)
    E = src.token_emb.weight.detach().double()
    with torch.no_grad():
        a2l = m.action_to_lie
        base = table.clone()
        for name, vec in slots.items():
            getattr(a2l, name).copy_(vec.view_as(getattr(a2l, name)).to(getattr(a2l, name).dtype))
        if "clock" in slots:
            base = base - slots["clock"][None]
        if "clock_x" in slots:
            for i in TORUS_ACTIONS[:2]:
                base[i] = base[i] - slots["clock_x"]
            for i in TORUS_ACTIONS[2:]:
                base[i] = base[i] - slots["clock_y"]
        if variant == "Vanilla_r2DF":
            A = torch.stack([(E[p] - E[q]) / math.sqrt(2) for p, q in TORUS_PAIRS])
            cn = float(A.norm(dim=1).mean())
            W_in = cn * torch.linalg.pinv(A).T
            W_out = torch.stack([base[p] for p, _ in TORUS_PAIRS], 1) / cn
        else:
            rows = list(TORUS_ACTIONS) if variant == "Vanilla_r2AO" else list(range(E.shape[0]))
            U, s, Vt = torch.linalg.svd(base[rows], full_matrices=False)
            L = U[:, :2] * s[:2]
            W_in = (torch.linalg.pinv(E[rows]) @ L).T
            W_out = Vt[:2].T
        a2l.w_in.weight.copy_(W_in.to(a2l.w_in.weight.dtype))
        a2l.w_out.weight.copy_(W_out.to(a2l.w_out.weight.dtype))
    return m.eval()


def score_warm(warm_root, device, n_trials, T):
    """Rule 14 trainable half: each class's construction after 60 cosine epochs at peak lr 1e-3 (warm start). HELD = the
    warm-started model is still SOLVED (final-5% loss < 0.05) and its frozen accuracy is within 0.02 of the construction's."""
    from mapformer.stats_core import classify_run
    rep = json.load(open(f"{REPO}/runs/rankclock_pilot/construct_rankclock.json"))
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    out = {}
    for cls, (variant, _) in CLASSES.items():
        p = os.path.join(warm_root, f"warm_{cls}", f"{variant}.pt")
        if not os.path.exists(p):
            print(f"  {cls:6s}: no warm-start checkpoint at {p}")
            continue
        m, b = load(p)
        L = np.asarray(b["losses"], float)
        k = classify_run(L)
        acc, nll = evaluate(m.to(device), env, T, n_trials, 0.0, device, seed=12345)
        a0 = rep[cls]["best"]["acc"]
        held = k["registered"] == "SOLVED" and acc >= a0 - 0.02
        out[cls] = {"construction_acc": a0, "after_acc": acc, "after_nll": nll, "class": k["cls"],
                    "loss_first": float(L[0]), "loss_max": float(L.max()), "loss_tail": float(k["tail"]), "held": held}
        print(f"  {cls:6s}: construction {a0:.4f} -> after 60 ep {acc:.4f} (nll {nll:.4f}) | loss first {L[0]:.4f} "
              f"max {L.max():.4f} tail {k['tail']:.4f} {k['cls']} | {'HELD' if held else 'LOST'}")
    rep["warm"] = out
    json.dump(rep, open(f"{REPO}/runs/rankclock_pilot/construct_rankclock.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--warmstart-dir", default=None)
    ap.add_argument("--score-warm", default=None, help="score the warm-started runs under this root, then exit")
    a = ap.parse_args()
    if a.score_warm:
        score_warm(a.score_warm, a.device, a.n_trials, a.T)
        return
    src, b = load(a.source)
    S = step_table(src)
    om = src.path_integrator.omega.detach().reshape(-1).double()
    freq = token_freq(S.shape[0])
    cand = tables(S, freq)
    env = GridWorld(size=b["config"]["grid_size"], n_obs_types=16, p_empty=0.5, seed=10000)
    sv = torch.linalg.svdvals(S)
    # score COPIES on the device: the source and every construction stay on CPU, where the linear algebra runs
    acc, nll = evaluate(copy.deepcopy(src).to(a.device), env, a.T, a.n_trials, 0.0, a.device, seed=12345)
    rep = {"source_path": a.source, "source": {"acc": acc, "nll": nll},
           "source_final_loss": float(np.mean(b["losses"][-45:])),
           "source_step_table_singular_values": [float(x) for x in sv[:6]],
           "token_freq_moves": [float(freq[i]) for i in TORUS_ACTIONS]}
    print(f"source r4: frozen acc {acc:.4f} nll {nll:.4f}; final loss {rep['source_final_loss']:.4f}; "
          f"|S| singular values {[round(float(x), 3) for x in sv[:6]]}")
    for cls, (variant, allowed) in CLASSES.items():
        best = None
        for cname in allowed:
            table, slots = cand[cname]
            m = realise(cls, variant, table, slots, src, b)
            got = step_table(m)
            err = float(torch.angle(torch.exp(1j * om[None] * (got - table))).abs().max())
            dev = float(torch.angle(torch.exp(1j * om[None] * (got - S))).abs().max())
            acc, nll = evaluate(copy.deepcopy(m).to(a.device), env, a.T, a.n_trials, 0.0, a.device, seed=12345)
            sc = {k: float(getattr(m.action_to_lie, k).weight.detach().abs().max()) for k in ("w_in", "w_out")}
            row = {"construction": cname, "acc": acc, "nll": nll, "realisation_err": err, "dev_from_source": dev,
                   "max_w_in": sc["w_in"], "max_w_out": sc["w_out"]}
            rep.setdefault(cls, {"tried": []})["tried"].append(row)
            print(f"  {cls:6s} via {cname:6s}: frozen acc {acc:.4f} nll {nll:.4f} | realisation err {err:.1e}, "
                  f"max angle dev from source {dev:.3f} | max|w_in| {sc['w_in']:.3f} max|w_out| {sc['w_out']:.3f}")
            if best is None or acc > best[0]:
                best = (acc, cname, m)
        rep[cls]["best"] = {"construction": best[1], "acc": best[0]}
        print(f"  {cls:6s} BEST {best[1]} frozen acc {best[0]:.4f}")
        if a.warmstart_dir:
            d = os.path.join(a.warmstart_dir, cls)
            os.makedirs(d, exist_ok=True)
            torch.save({"model_state_dict": {k: v.cpu() for k, v in best[2].state_dict().items()}, "losses": [],
                        "variant": variant, "seed": 1990, "config": dict(b["config"])},
                       os.path.join(d, f"{variant}.pt"))
    src_ref = {k: float(getattr(src.action_to_lie, k).weight.detach().abs().max()) for k in ("w_in", "w_out")}
    rep["source_scales"] = src_ref
    print(f"source scales max|w_in| {src_ref['w_in']:.3f} max|w_out| {src_ref['w_out']:.3f}")
    json.dump(rep, open(f"{REPO}/runs/rankclock_pilot/construct_rankclock.json", "w"), indent=2)


if __name__ == "__main__":
    main()
