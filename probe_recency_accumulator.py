"""H2: does the UNCONSTRAINED signed arm learn a DIFFERENT accumulator per task?

The primary claim after the 2026-09-07 amendment. `Signed_r4` is unconstrained --
its Delta may take either sign -- so it CONTAINS the monotone solution and can
adopt whichever code the task rewards. The clock/map dichotomy says the two tasks
reward opposite codes, so the same architecture trained on each should learn:

  torus (a MAP)     Delta signed, opposite actions cancelling -> theta measures
                    NET DISPLACEMENT. Diffusive: range(theta) ~ T^0.5.
  recency (a CLOCK) Delta one-signed on the counted tokens -> theta measures
                    ELAPSED COUNT. Ballistic: range(theta) ~ T^1.

If instead the same accumulator appears on both, alpha is descriptive only and a
claim in positional_review.tex is retracted.

`probe_sign.py` cannot be reused: it scores OPPOSITION between opposite actions,
and the recency task has no actions at all. The two task-independent quantities
are measured here instead -- the negative fraction of Delta, and the growth
exponent alpha of range(cumsum Delta) against T.
"""
import argparse, glob, json, os
import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.environment_recency import RecencyWorld
from mapformer.train_variant import VARIANT_MAP


def _load(ck, dev):
    b = torch.load(ck, map_location="cpu", weights_only=False)
    if "config" in b:                       # torus trainer's layout
        c = b["config"]; state = b["model_state_dict"]
        v = os.path.basename(ck)[:-3]
        kw = dict(vocab_size=c["vocab_size"], d_model=c["d_model"],
                  n_heads=c["n_heads"], n_layers=c["n_layers"],
                  grid_size=c["grid_size"])
    else:                                   # train_recency's flat layout
        state = b["model_state"]; v = b["variant"]
        kw = dict(vocab_size=b["vocab_size"], d_model=b["d_model"],
                  n_heads=b["n_heads"], n_layers=b["n_layers"],
                  grid_size=b["grid_size"])
    m = VARIANT_MAP[v](**kw); m.load_state_dict(state)
    return m.to(dev).eval(), kw


@torch.no_grad()
def measure(ck, task, dev, lengths=(64, 128, 256, 512, 1024)):
    m, kw = _load(ck, dev)
    if not hasattr(m, "action_to_lie"):
        return None
    rngs, negs = [], []
    for T in lengths:
        if task == "torus":
            env = GridWorld(size=kw["grid_size"], n_obs_types=16,
                            p_empty=0.5, seed=10000)
            np.random.seed(0)
            tok, _o, _r = env.generate_trajectory(T)
            tok = tok.unsqueeze(0)
        else:
            env = RecencyWorld(k_max=64, seed=10000)
            tok, _sp, _a, _i = env.generate_episode(T, np.random.RandomState(0))
            tok = tok.unsqueeze(0)
        d = m.action_to_lie(m.token_emb(tok.to(dev)))        # (1,T,H,nb)
        d = d.float().cpu().numpy()
        negs.append(float((d < 0).mean()))
        theta = np.cumsum(d, axis=1)
        # range of the accumulator, averaged over heads and frequency blocks
        rngs.append(float((theta.max(axis=1) - theta.min(axis=1)).mean()))
    T = np.array(lengths, dtype=float); R = np.array(rngs)
    ok = R > 0
    alpha = float(np.polyfit(np.log(T[ok]), np.log(R[ok]), 1)[0]) if ok.sum() > 1 else float("nan")
    return {"alpha": alpha, "neg_frac": float(np.mean(negs)),
            "neg_frac_at_max_T": negs[-1], "ranges": rngs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torus-dir", default="runs/sign/p0")
    ap.add_argument("--recency-dir", default="runs/recency")
    ap.add_argument("--variants", nargs="+",
                    default=["Signed_r4", "Abs_r4", "Pos_r4", "CARoPE_r4"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="RECENCY_H2.md")
    a = ap.parse_args()
    dev = torch.device(a.device)

    res = {}
    for v in a.variants:
        for task, pat in (("torus", f"{a.torus_dir}/{v}_s*/{v}.pt"),
                          ("recency", f"{a.recency_dir}/{v}_s*/{v}_recency.pt")):
            vals = []
            for ck in sorted(glob.glob(pat)):
                try:
                    r = measure(ck, task, dev)
                except Exception as e:                       # noqa: BLE001
                    print(f"  skip {ck}: {e}", flush=True); continue
                if r:
                    vals.append(r)
            if vals:
                res[(v, task)] = vals
                al = np.array([x["alpha"] for x in vals])
                nf = np.array([x["neg_frac"] for x in vals])
                print(f"{v:11s} {task:8s} n={len(vals):2d}  "
                      f"alpha {al.mean():.3f}+/-{al.std(ddof=1) if len(al)>1 else 0:.3f}  "
                      f"neg_frac {nf.mean():.3f}+/-{nf.std(ddof=1) if len(nf)>1 else 0:.3f}",
                      flush=True)

    L = ["# H2 -- does the unconstrained arm learn a different accumulator per task?", "",
         "`Signed_r4` is unconstrained and CONTAINS the monotone solution, so it can "
         "adopt whichever code a task rewards. The clock/map dichotomy predicts it "
         "adopts a cancelling code on the torus (a map, alpha ~ 0.5) and a "
         "one-signed counter on recency (a clock, alpha ~ 1.0). If the same "
         "accumulator appears on both, alpha is descriptive only.", "",
         "| arm | task | n | alpha | negative fraction of Delta |",
         "|---|---|---|---|---|"]
    for (v, task), vals in res.items():
        al = np.array([x["alpha"] for x in vals]); nf = np.array([x["neg_frac"] for x in vals])
        sd = lambda z: z.std(ddof=1) if len(z) > 1 else 0.0
        L.append(f"| `{v}` | {task} | {len(vals)} | {al.mean():.3f} +/- {sd(al):.3f} | "
                 f"{nf.mean():.3f} +/- {sd(nf):.3f} |")
    for v in a.variants:
        if (v, "torus") in res and (v, "recency") in res:
            at = np.array([x["alpha"] for x in res[(v, "torus")]])
            ar = np.array([x["alpha"] for x in res[(v, "recency")]])
            nt = np.array([x["neg_frac"] for x in res[(v, "torus")]])
            nr = np.array([x["neg_frac"] for x in res[(v, "recency")]])
            pooled = np.sqrt(at.var(ddof=1) / len(at) + ar.var(ddof=1) / len(ar))
            L += ["", f"**`{v}`: alpha {at.mean():.3f} (torus) -> {ar.mean():.3f} "
                      f"(recency), delta {ar.mean()-at.mean():+.3f}, se {pooled:.3f}; "
                      f"negative fraction {nt.mean():.3f} -> {nr.mean():.3f}.**"]
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()},
              open(a.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(L)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
