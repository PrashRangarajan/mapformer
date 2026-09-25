"""P3: does the learned gate actually separate action tokens from observations?

The pre-registered discriminator in GATED_PREREG.md, and the one that decides
whether an accuracy gain means what the mechanism claims. A gain with no gate
separation would mean the module helped for some other reason, and that reason
would have to be named rather than assumed.

The floor is set from existing data, not chosen. `GATE_PROBE.md` measured Selective
RoPE's sigmoid gate on this exact contrast at **1.35x** on the torus and judged that
NOT to be suppression -- 0.560 on actions against 0.415 on observations is nowhere
near a gate that closes. So 1.35x is what this must CLEAR to mean anything.

Reports the ratio per seed rather than a mean over seeds: the earlier version of the
recency gate probe averaged SIGNED quantities across seeds, where a global sign is
arbitrary per seed, and reported noise. Ratios of magnitudes are safe to pool, but
the per-seed column is printed anyway so the spread is visible -- the recency gate
ratio spanned 0.03x to 162x, and a median hid that.
"""
import argparse, glob, json, os
import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


@torch.no_grad()
def probe(ck, dev, T=512, env_seed=10000):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = blob["config"]; v = os.path.basename(ck)[:-3]
    m = VARIANT_MAP[v](vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                       n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                       grid_size=cfg["grid_size"])
    m.load_state_dict(blob["model_state_dict"]); m = m.to(dev).eval()
    if not hasattr(m, "gate_of"):
        return None
    env = GridWorld(size=cfg["grid_size"], n_obs_types=cfg.get("n_obs_types", 16),
                    p_empty=cfg.get("p_empty", 0.5), seed=env_seed)
    np.random.seed(0)
    tok, _om, _rv = env.generate_trajectory(T)
    g = m.gate_of(tok.unsqueeze(0).to(dev)).float().cpu().numpy()[0]   # (T,H)
    # the token stream is interleaved [a, o, a, o, ...]; actions sit at even slots
    act = g[0::2].mean(); obs = g[1::2].mean()
    return {"gate_action": float(act), "gate_obs": float(obs),
            "ratio": float(act / max(obs, 1e-9)),
            "per_head_ratio": [float(g[0::2, h].mean() / max(g[1::2, h].mean(), 1e-9))
                               for h in range(g.shape[1])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/home/prashr/mapformer/runs/gated_torus")
    ap.add_argument("--variants", nargs="+", default=["Gated_r4", "Gated_r2", "Gated_r4_frozen"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/home/prashr/mapformer/GATED_SEPARATION.md")
    a = ap.parse_args()
    dev = torch.device(a.device)

    res = {}
    for v in a.variants:
        rows = []
        for ck in sorted(glob.glob(f"{a.runs_dir}/{v}_s*/{v}.pt")):
            r = probe(ck, dev)
            if r:
                rows.append(r); print(f"{v:16s} {os.path.basename(os.path.dirname(ck)):22s} "
                                      f"act {r['gate_action']:.4f} obs {r['gate_obs']:.4f} "
                                      f"ratio {r['ratio']:.2f}x", flush=True)
        if rows:
            res[v] = rows

    L = ["# P3: does the gate separate actions from observations?", "",
         "The pre-registered discriminator (`GATED_PREREG.md`). **The floor is "
         "1.35x**, taken from `GATE_PROBE.md`: Selective RoPE's gate measured "
         "exactly that on this contrast (0.560 on actions, 0.415 on observations) "
         "and was judged NOT to be suppression. Clearing 1.35x is the requirement, "
         "not the target.", "",
         "| arm | gate on ACTIONS | gate on OBS | ratio | per-seed ratios | n |",
         "|---|---|---|---|---|---|"]
    for v, rows in res.items():
        A = np.array([r["gate_action"] for r in rows])
        O = np.array([r["gate_obs"] for r in rows])
        R = np.array([r["ratio"] for r in rows])
        L.append(f"| `{v}` | {A.mean():.4f} ± {A.std(ddof=1):.4f} | "
                 f"{O.mean():.4f} ± {O.std(ddof=1):.4f} | **{R.mean():.2f}x** | "
                 + " ".join(f"{x:.2f}" for x in R) + f" | {len(R)} |")
    L += ["", "`Gated_r4_frozen` is the control: its gate cannot learn, so its "
          "ratio must be **1.00x** by construction. Any other value there means "
          "the probe is wrong, not that the frozen gate separated anything."]
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(res, open(a.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
