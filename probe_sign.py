"""Post-hoc gate: did the sign constraint survive training, and what did the
SIGNED arm actually learn?

Two questions the accuracy table cannot answer.

1. CONSTRAINT INTEGRITY. Every constrained arm must have Delta >= 0 on real token
   streams after 300 epochs. It is enforced by construction, so a violation means
   the wrong checkpoint was loaded -- this is a wiring check, not a hypothesis.

2. DOES THE SIGNED ARM USE THE SIGN? The mechanism claim is that east and west
   must cancel. If it is right, the signed model's Delta for opposite actions
   should be near-opposite, and the constrained model's cannot be. Measured as
   the opposition score N+S ~= 0 used in ACTION_GEOMETRY.md, plus the fraction of
   Delta mass that is negative.

   This is the diagnostic that decides whether a null in the accuracy table means
   "sign is unnecessary" or "the signed model never used the sign either".
"""
import argparse, json, os
import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


@torch.no_grad()
def probe(ck, dev):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = blob["config"]; v = os.path.basename(ck)[:-3]
    m = VARIANT_MAP[v](vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                       n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                       grid_size=cfg["grid_size"])
    m.load_state_dict(blob["model_state_dict"]); m = m.to(dev).eval()
    if not hasattr(m, "action_to_lie"):
        return None

    env = GridWorld(size=cfg["grid_size"], n_obs_types=cfg.get("n_obs_types", 16),
                    p_empty=cfg.get("p_empty", 0.5), seed=10000)
    np.random.seed(0)
    tok, _om, _rv = env.generate_trajectory(512)
    d = m.action_to_lie(m.token_emb(tok.unsqueeze(0).to(dev)))     # (1,T,H,nb)

    # per-action mean Delta vector, actions are ids 0..N_ACTIONS-1 at even slots
    A = env.N_ACTIONS
    ids = tok.to(dev)
    vecs = {}
    for a in range(A):
        sel = (ids == a)
        if sel.sum() > 4:
            vecs[a] = d[0][sel].mean(0).flatten().cpu().numpy()
    out = dict(variant=v,
               delta_min=float(d.min()), delta_max=float(d.max()),
               frac_negative=float((d < 0).float().mean()),
               nonneg=bool(d.min() >= 0))
    # opposition: actions 0/1 and 2/3 are the +/- pairs on the torus
    for lab, (i, j) in (("oppose_x", (0, 1)), ("oppose_y", (2, 3))):
        if i in vecs and j in vecs:
            s = vecs[i] + vecs[j]
            scale = 0.5 * (np.linalg.norm(vecs[i]) + np.linalg.norm(vecs[j]))
            out[lab] = float(np.linalg.norm(s) / scale) if scale > 0 else float("nan")
    if 0 in vecs and 2 in vecs:
        n0, n2 = np.linalg.norm(vecs[0]), np.linalg.norm(vecs[2])
        out["cos_xy"] = float(abs(vecs[0] @ vecs[2] / (n0 * n2))) if n0 * n2 > 0 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)

    rows = []
    for v in a.variants:
        for s in a.seeds:
            ck = os.path.join(a.runs_dir, "p0", f"{v}_s{s}", f"{v}.pt")
            if os.path.exists(ck):
                r = probe(ck, dev)
                if r:
                    r["seed"] = s; rows.append(r); print(r, flush=True)

    o = ["# Sign-ablation probe: constraint integrity and action geometry", "",
         "`nonneg` is a wiring check -- the constraint is enforced by construction, so",
         "False means the wrong checkpoint was loaded. `oppose` is the ACTION_GEOMETRY",
         "opposition score `||Delta(+x) + Delta(-x)|| / mean||Delta||`: 0 means opposite",
         "actions cancel exactly, 2 means they are identical. A monotone code CANNOT",
         "reach 0. `frac_neg` is the fraction of Delta entries below zero on a real",
         "512-step stream.", "",
         "| arm | nonneg | frac_neg | Delta range | oppose_x | oppose_y | \\|cos(x,y)\\| |",
         "|---|---|---|---|---|---|---|"]
    for v in a.variants:
        rs = [r for r in rows if r["variant"] == v]
        if not rs:
            continue
        g = lambda k: np.mean([r[k] for r in rs if k in r and np.isfinite(r[k])]) \
            if any(k in r for r in rs) else float("nan")
        o.append(f"| `{v}` | {all(r['nonneg'] for r in rs)} | {g('frac_negative'):.3f} | "
                 f"{min(r['delta_min'] for r in rs):+.3f} .. {max(r['delta_max'] for r in rs):+.3f} | "
                 f"{g('oppose_x'):.3f} | {g('oppose_y'):.3f} | {g('cos_xy'):.3f} |")
    o += ["", "**Reading it.** If the signed arm's opposition scores are near 0 and the",
          "constrained arms' are near 2, the mechanism is confirmed at the level of the",
          "learned code, independently of accuracy. If the SIGNED arm's scores are also",
          "far from 0, it never used the sign either, and any null in the accuracy table",
          "says nothing about whether sign matters in principle."]
    open(a.out, "w").write("\n".join(o) + "\n")
    json.dump(rows, open(a.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(o))


if __name__ == "__main__":
    main()
