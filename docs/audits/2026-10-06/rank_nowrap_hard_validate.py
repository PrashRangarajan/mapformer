"""RANK_NOWRAP Amendment 2, CPU only: validate the hard-target accuracy h (stratum retrace_miss) as the registered
quantity, on stored per-head runs, and build the per-stratum pools for the transport power simulation.

Runs (T=1024, 900 epochs): RANK_ND D2 Vanilla_r2ph / Vanilla_r3ph seeds 0-7 (ND 32-torus; held-out map seed 10000,
walk seed 10000 + s, as eval_nd), RANK_MI Vanilla_r2ph and RANK3 Vanilla_r3ph seeds 0-7 (paper 64-torus,
environment.GridWorld, held-out map seed 10000, walk seed 1234 + s, as eval_noise_refine). 60 walks per run.
Per run: accuracy and NLL on the strata copy / blank_out / retrace_miss (analyze_rank_nowrap.strat, imported), and the
training-loss class. Then
  (1) HIT (h >= HIT_H) vs loss-SOLVED, per run;
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


def gw_stream(s, n=NW):
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(1234 + s); out = []
    for _ in range(n):
        tok, _o, rev = env.generate_trajectory(T)
        out.append((tok, rev[1::2].numpy()) + A.traj_info(env, tok, T))
    return env, out


def run_one(ck, st):
    b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]
    m = VARIANT_MAP[b["variant"]](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                                  n_layers=c["n_layers"], grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); m.eval()
    acc, cnt, nll = A.strat(m, st, "cpu", want_nll=True)
    return {"acc": {k: acc[k] for k in ST3 + ("all",)}, "nll": {k: nll[k] for k in ST3 + ("all",)},
            "n": {k: cnt[k] for k in ST3}, "solved": classify_run(b["losses"])["registered"] == "SOLVED",
            "tail": float(classify_run(b["losses"])["tail"])}


def main():
    sets = [("ND32", 2, f"{RP}/runs/rank_nd/D2/Vanilla_r2ph_s{{s}}/Vanilla_r2ph.pt"),
            ("ND32", 3, f"{RP}/runs/rank_nd/D2/Vanilla_r3ph_s{{s}}/Vanilla_r3ph.pt"),
            ("T64", 2, f"{RP}/runs/rank_mi/p0/Vanilla_r2ph_s{{s}}/Vanilla_r2ph.pt"),
            ("T64", 3, f"{RP}/runs/rank3/p0/Vanilla_r3ph_s{{s}}/Vanilla_r3ph.pt")]
    share = {}
    for N in (32, 256):
        env, st = A.stream(N, 0, n=100); f = A.floors(st, env.unified_blank)
        assert f["bad_copy"] == 0; share[N] = f["share"]
        print(f"ND {N}-torus stratum shares (held-out stream, 100 walks): " + "  ".join(f"{k} {v:.3f}" for k, v in f["share"].items())
              + f"; retrace-or-blank floor {f['retrace']:.3f}; bad copies {f['bad_copy']}", flush=True)
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
            print(f"  {tag} r{rank} s{s}: h {r['acc']['retrace_miss']:.3f} copy {r['acc']['copy']:.3f} blank_out "
                  f"{r['acc']['blank_out']:.3f} all {r['acc']['all']:.3f} | NLL all {r['nll']['all']:.3f} | loss-SOLVED "
                  f"{r['solved']} (tail {r['tail']:.3f})", flush=True)
    json.dump({"share": share, "pool": pool, "n_walks": NW}, open(f"{RP}/docs/audits/2026-10-06/rank_nowrap_hard_pool.json", "w"), indent=1)

    print(f"\n(1) HIT = h >= {A.HIT_H} vs training-loss SOLVED, per run:")
    for tag in ("ND32", "T64"):
        for rank in (2, 3):
            P = [p for p in pool if p["set"] == tag and p["rank"] == rank]
            print(f"  {tag} rank {rank}: HIT {sum(p['acc']['retrace_miss'] >= A.HIT_H for p in P)}/8, SOLVED {sum(p['solved'] for p in P)}/8, "
                  f"agree {sum((p['acc']['retrace_miss'] >= A.HIT_H) == p['solved'] for p in P)}/8; h sorted "
                  + " ".join(f"{p['acc']['retrace_miss']:.3f}{'S' if p['solved'] else ''}" for p in sorted(P, key=lambda p: p['acc']['retrace_miss'])))
    hs = sorted(p["acc"]["retrace_miss"] for p in pool if p["solved"]); hu = sorted(p["acc"]["retrace_miss"] for p in pool if not p["solved"])
    print(f"  over all 32 runs: lowest h among SOLVED {hs[0]:.3f}, highest among not SOLVED {hu[-1]:.3f}")

    print("\n(2) transport of the ND 32-torus runs to the 256-torus stratum mix (per-stratum accuracy / NLL held fixed):")
    f32 = share[32]["copy"] + share[32]["blank_out"]; f256 = share[256]["copy"] + share[256]["blank_out"]
    print(f"  floor-predicted share: 32 {f32:.3f}, 256 {f256:.3f}; an error on a floor-predicted target costs "
          f"f/(1-f) = {f32 / (1 - f32):.2f} (32) vs {f256 / (1 - f256):.2f} (256) units of rel")
    for p in [p for p in pool if p["set"] == "ND32"]:
        raw = {N: sum(share[N][k] * p["acc"][k] for k in ST3) for N in (32, 256)}
        nll = {N: sum(share[N][k] * p["nll"][k] for k in ST3) for N in (32, 256)}
        rel = {N: (raw[N] - (share[N]["copy"] + share[N]["blank_out"])) / share[N]["retrace_miss"] for N in (32, 256)}
        print(f"  r{p['rank']} s{p['seed']}: h {p['acc']['retrace_miss']:.3f} (HIT {p['acc']['retrace_miss'] >= A.HIT_H}) | raw 32 "
              f"{raw[32]:.3f} -> 256 {raw[256]:.3f} | rel 32 {rel[32]:+.3f} -> 256 {rel[256]:+.3f} (rel-HIT {rel[32] >= 0.9} -> "
              f"{rel[256] >= 0.9}) | held-out NLL 32 {nll[32]:.3f} -> 256 {nll[256]:.3f} (< 0.05: {nll[32] < 0.05} -> {nll[256] < 0.05})")


if __name__ == "__main__":
    main()
