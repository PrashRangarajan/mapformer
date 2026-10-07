"""RANK_NOWRAP Amendments 2-3, CPU only: validate the plain-hard-target accuracy p (stratum hard_plain = retrace_miss
and not wrap-only; Amendment 3) as the registered quantity -- and h (all hard, Amendment 2) beside it -- on stored
per-head runs; describe the hard-target composition of both grids; build the per-stratum pools for the wrap-aware
transport power simulation.

Runs (T=1024, 900 epochs): RANK_ND D2 Vanilla_r2ph / Vanilla_r3ph seeds 0-7 (ND 32-torus; held-out map seed 10000,
walk seed 10000 + s, as eval_nd), RANK_MI Vanilla_r2ph and RANK3 Vanilla_r3ph seeds 0-7 (paper 64-torus,
environment.GridWorld, held-out map seed 10000, walk seed 1234 + s, as eval_noise_refine). 60 walks per run.
Per run: accuracy and NLL on the strata copy / blank_out / retrace_miss (analyze_rank_nowrap.strat, imported), and the
training-loss class. Then
  (1) HIT (p >= HIT_P; and h >= 0.90, Amendment 2) vs loss-SOLVED, per run;
  (3) Amendment 4: the per-seed sampling error of p by a CLUSTER bootstrap over walks (errors cluster within walks),
      at 60 walks and projected to the registered 400 (x sqrt(60/400)); near-cut runs (|p - HIT_P| < 0.02)
      re-evaluated on a second, independent 60-walk stream;
  (2) TRANSPORT to the 256-torus: keep each run's per-stratum accuracies / NLLs, reweight by the 256-torus's stratum
      shares (measured on its held-out stream) -- raw accuracy, Amendment 1's rel, and the mean NLL that a loss cut would
      see. h and HIT are invariant under this transport by construction; rel and the loss cut are not (the audit's point).
Writes rank_nowrap_hard_pool.json (input of rank_nowrap_power.py). Run from /home/prashr, CPU:
  CUDA_VISIBLE_DEVICES= python3 mapformer/docs/audits/2026-10-06/rank_nowrap_hard_validate.py
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
torch.set_num_threads(4)
from mapformer import analyze_rank_nowrap as A          # noqa: E402
from mapformer.environment import GridWorld             # noqa: E402
from mapformer.stats_core import classify_run           # noqa: E402
from mapformer.train_variant import VARIANT_MAP         # noqa: E402

RP = "/home/prashr/mapformer"; NW = 60; T = 1024
ST3 = ("copy", "blank_out", "retrace_miss")
ST4 = ("copy", "blank_out", "hard_wrap", "hard_plain")      # the transport strata (Amendment 3)


def gw_stream(s, n=NW):
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(1234 + s); out = []
    for _ in range(n):
        tok, _o, rev = env.generate_trajectory(T)
        out.append((tok, rev[1::2].numpy()) + A.traj_info(env, tok, T))
    return env, out


def load_model(ck):
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]
    m = VARIANT_MAP[b["variant"]](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                                  n_layers=c["n_layers"], grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); m.eval()
    return m, b


def run_one(ck, st):
    """Per-walk strata (Amendment 4: kept per walk for the cluster bootstrap), aggregated exactly as one strat() call."""
    m, b = load_model(ck)
    ks = ST3 + ("hard_wrap", "hard_plain", "all")
    ok = {k: 0.0 for k in ks}; nl = {k: 0.0 for k in ks}; cnt = {k: 0 for k in ks}; walks = []
    for w in st:
        acc, c_, nll = A.strat(m, [w], "cpu", want_nll=True)
        for k in ks:
            if c_[k]:
                ok[k] += acc[k] * c_[k]; nl[k] += nll[k] * c_[k]; cnt[k] += c_[k]
        walks.append((round((acc["hard_plain"] or 0) * c_["hard_plain"]), c_["hard_plain"]))
    return {"acc": {k: (ok[k] / cnt[k] if cnt[k] else None) for k in ks}, "nll": {k: (nl[k] / cnt[k] if cnt[k] else None) for k in ks},
            "n": cnt, "walks_plain": walks, "solved": classify_run(b["losses"])["registered"] == "SOLVED",
            "tail": float(classify_run(b["losses"])["tail"])}


def cluster_se(walks, n_boot=2000, seed=0):
    w = np.array(walks, float); rng = np.random.default_rng(seed); out = []
    for _ in range(n_boot):
        x = w[rng.integers(0, len(w), len(w))]
        out.append(x[:, 0].sum() / max(x[:, 1].sum(), 1))
    return float(np.std(out, ddof=1))


def main():
    sets = [("ND32", 2, f"{RP}/runs/rank_nd/D2/Vanilla_r2ph_s{{s}}/Vanilla_r2ph.pt"),
            ("ND32", 3, f"{RP}/runs/rank_nd/D2/Vanilla_r3ph_s{{s}}/Vanilla_r3ph.pt"),
            ("T64", 2, f"{RP}/runs/rank_mi/p0/Vanilla_r2ph_s{{s}}/Vanilla_r2ph.pt"),
            ("T64", 3, f"{RP}/runs/rank3/p0/Vanilla_r3ph_s{{s}}/Vanilla_r3ph.pt")]
    share = {}
    for N in (32, 256):
        env, st = A.stream(N, 0, n=100); f = A.floors(st, env.unified_blank)
        assert f["bad_copy"] == 0; share[N] = f["share"]
        # wrap-aware composition of the hard set (Amendment 3): counts per sequence, lag, and turns (direction changes
        # between the previous visit to the cell and the target)
        comp = {"hard_wrap": [], "hard_plain": []}; tot = 0
        for tok, r, o, wrap, lag, ret, inr in st:
            a = tok[0::2].numpy(); turns = np.concatenate([[0], np.cumsum(a[1:] != a[:-1])])
            for t in np.nonzero(r)[0]:
                tot += 1
                if ret[t] != o[t]:
                    comp["hard_wrap" if wrap[t] else "hard_plain"].append((lag[t], turns[t] - turns[t - lag[t]] if lag[t] <= t else -1))
        for k in comp:
            share[N][k] = len(comp[k]) / tot
        print(f"ND {N}-torus stratum shares (held-out stream, 100 walks): " + "  ".join(f"{k} {v:.3f}" for k, v in share[N].items())
              + f"; retrace-or-blank floor {f['retrace']:.3f}; bad copies {f['bad_copy']}", flush=True)
        for k, v in comp.items():
            if v:
                v = np.array(v)
                print(f"    {k}: {len(v) / 100:.1f} per sequence, {len(v) / max(1, len(comp['hard_wrap']) + len(comp['hard_plain'])):.2f} "
                      f"of the hard set, lag median {np.median(v[:, 0]):.0f}, turns median {np.median(v[:, 1]):.0f}", flush=True)
    pool = []
    for tag, rank, fmt in sets:
        for s in range(8):
            if tag == "ND32":
                env, st = A.stream(32, s, n=NW)
            else:
                env, st = gw_stream(s)
            fl = A.floors(st, env.unified_blank); assert fl["bad_copy"] == 0, (tag, s)
            r = run_one(fmt.format(s=s), st); r.update({"set": tag, "rank": rank, "seed": s, "floor": fl["retrace"]})
            pool.append(r)
            print(f"  {tag} r{rank} s{s}: p {r['acc']['hard_plain']:.3f} wrap-only {r['acc']['hard_wrap'] if r['acc']['hard_wrap'] is not None else float('nan'):.3f} h {r['acc']['retrace_miss']:.3f} copy {r['acc']['copy']:.3f} blank_out "
                  f"{r['acc']['blank_out']:.3f} all {r['acc']['all']:.3f} | NLL all {r['nll']['all']:.3f} | loss-SOLVED "
                  f"{r['solved']} (tail {r['tail']:.3f})", flush=True)
    section3(pool)
    json.dump({"share": share, "pool": pool, "n_walks": NW}, open(f"{RP}/docs/audits/2026-10-06/rank_nowrap_hard_pool.json", "w"), indent=1)

    for key, thr, nm in (("hard_plain", A.HIT_P, "p (plain hard; registered, Amendment 3)"), ("retrace_miss", 0.90, "h (all hard; Amendment 2)")):
        print(f"\n(1) HIT = {nm} >= {thr} vs training-loss SOLVED, per run:")
        for tag in ("ND32", "T64"):
            for rank in (2, 3):
                P = [p for p in pool if p["set"] == tag and p["rank"] == rank]
                print(f"  {tag} rank {rank}: HIT {sum(p['acc'][key] >= thr for p in P)}/8, SOLVED {sum(p['solved'] for p in P)}/8, "
                      f"agree {sum((p['acc'][key] >= thr) == p['solved'] for p in P)}/8; sorted "
                      + " ".join(f"{p['acc'][key]:.3f}{'S' if p['solved'] else ''}" for p in sorted(P, key=lambda p: p['acc'][key])))
        hs = sorted(p["acc"][key] for p in pool if p["solved"]); hu = sorted(p["acc"][key] for p in pool if not p["solved"])
        print(f"  over all 32 runs: lowest among SOLVED {hs[0]:.4f}, highest among not SOLVED {hu[-1]:.4f}; gap {hs[0] - hu[-1]:+.4f}; "
              f"threshold {thr} margin to SOLVED {hs[0] - thr:+.4f}, to not SOLVED {thr - hu[-1]:+.4f}")

    print("\n(2) transport of the ND 32-torus runs to the 256-torus stratum mix (per-stratum accuracy / NLL held fixed):")
    f32 = share[32]["copy"] + share[32]["blank_out"]; f256 = share[256]["copy"] + share[256]["blank_out"]
    print(f"  floor-predicted share: 32 {f32:.3f}, 256 {f256:.3f}; an error on a floor-predicted target costs "
          f"f/(1-f) = {f32 / (1 - f32):.2f} (32) vs {f256 / (1 - f256):.2f} (256) units of rel")
    for p in [p for p in pool if p["set"] == "ND32"]:
        raw = {N: sum(share[N][k] * p["acc"][k] for k in ST3) for N in (32, 256)}
        nll = {N: sum(share[N][k] * p["nll"][k] for k in ST3) for N in (32, 256)}
        rel = {N: (raw[N] - (share[N]["copy"] + share[N]["blank_out"])) / share[N]["retrace_miss"] for N in (32, 256)}
        print(f"  r{p['rank']} s{p['seed']}: h {p['acc']['retrace_miss']:.3f} (h-HIT@0.90 {p['acc']['retrace_miss'] >= 0.90}; p {p['acc']['hard_plain']:.3f}) | raw 32 "
              f"{raw[32]:.3f} -> 256 {raw[256]:.3f} | rel 32 {rel[32]:+.3f} -> 256 {rel[256]:+.3f} (rel-HIT {rel[32] >= 0.9} -> "
              f"{rel[256] >= 0.9}) | held-out NLL 32 {nll[32]:.3f} -> 256 {nll[256]:.3f} (< 0.05: {nll[32] < 0.05} -> {nll[256] < 0.05})")


def section3(pool):
    print(f"\n(3) per-seed sampling error of p (cluster bootstrap over walks, 2000 resamples; binomial for comparison):")
    for p in pool:
        se = cluster_se(p["walks_plain"]); nt = p["n"]["hard_plain"]; pv = p["acc"]["hard_plain"]
        binom = np.sqrt(max(pv * (1 - pv), 1e-12) / max(nt, 1))
        p["se_cluster_60"] = se
        print(f"  {p['set']} r{p['rank']} s{p['seed']}: p {pv:.4f} on {nt} targets / {len(p['walks_plain'])} walks | SE cluster {se:.4f} "
              f"(binomial {binom:.4f}); projected to 400 walks {se * np.sqrt(len(p['walks_plain']) / 400):.4f}")
    near = [p for p in pool if p["se_cluster_60"] > 0 and abs(p["acc"]["hard_plain"] - A.HIT_P) < 0.02]
    if near:
        print(f"  near-cut runs (|p - {A.HIT_P}| < 0.02) re-evaluated on a second, independent 60-walk stream (walk seed +20000):")
    for p in near:
        fmt = {("ND32", 2): "runs/rank_nd/D2/Vanilla_r2ph", ("ND32", 3): "runs/rank_nd/D2/Vanilla_r3ph",
               ("T64", 2): "runs/rank_mi/p0/Vanilla_r2ph", ("T64", 3): "runs/rank3/p0/Vanilla_r3ph"}[(p["set"], p["rank"])]
        v = fmt.rsplit("/", 1)[1]; ck = f"{RP}/{fmt}_s{p['seed']}/{v}.pt"
        if p["set"] == "ND32":
            env = A.GridWorldND(dims=2, size=32, seed=10000)
        else:
            env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
        np.random.seed(20000 + p["seed"]); st = []
        for _ in range(NW):
            tok, _o, rev = env.generate_trajectory(T)
            st.append((tok, rev[1::2].numpy()) + A.traj_info(env, tok, T))
        r2 = run_one(ck, st)
        print(f"    {p['set']} r{p['rank']} s{p['seed']} (loss-SOLVED {p['solved']}): p {p['acc']['hard_plain']:.4f} -> second stream "
              f"{r2['acc']['hard_plain']:.4f}")
    nd = [p for p in pool if p["set"] == "ND32"]
    hs = [p["acc"]["hard_plain"] for p in nd if p["solved"]]; hu = [p["acc"]["hard_plain"] for p in nd if not p["solved"]]
    print(f"  ND 32-torus runs only: lowest p among SOLVED {min(hs):.4f}, highest among not SOLVED {max(hu):.4f} (the narrow "
          f"0.975-0.990 gap of all 32 runs comes from the paper-torus runs)")


if __name__ == "__main__":
    main()
