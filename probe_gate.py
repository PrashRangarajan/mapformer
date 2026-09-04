"""Does the sigmoid gate learn to suppress OBSERVATION tokens?

MapFormer's action_to_lie reads every token and the model must LEARN that
Delta ~= 0 on observations (model.py docstring). A sigmoid gate is the natural
shape for that, which would explain why the gate helps on the torus -- where half
the tokens are observations that should contribute nothing to the path integral --
and hurts on parity, where every bit must contribute.

Prediction, stated before looking: gate(action) >> gate(observation) on the torus,
and no such split on parity.

Inference only, on checkpoints that already exist.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


@torch.no_grad()
def torus_gate(cp, dev, n_traj=24, T=128):
    blob = torch.load(cp, map_location="cpu", weights_only=False)
    c = blob["config"]
    m = VARIANT_MAP["GateAngle"](vocab_size=c["vocab_size"], d_model=c["d_model"],
                                 n_heads=c["n_heads"], n_layers=c["n_layers"],
                                 grid_size=c["grid_size"]).to(dev).eval()
    m.load_state_dict(blob["model_state_dict"])
    env = GridWorld(size=c["grid_size"], n_obs_types=c.get("n_obs_types", 16),
                    p_empty=c.get("p_empty", 0.5), seed=777)
    np.random.seed(777)
    act, obs = [], []
    for _ in range(n_traj):
        tok, _om, _rev = env.generate_trajectory(T)
        x = m.token_emb(tok.unsqueeze(0).to(dev))
        g = torch.sigmoid(m.angle.gate(x))[0]          # (L, H*nb)
        act.append(g[0::2].mean().item())              # actions at EVEN positions
        obs.append(g[1::2].mean().item())              # observations at ODD
    return float(np.mean(act)), float(np.mean(obs))


@torch.no_grad()
def parity_gate(cp, dev, n=24, L=64):
    blob = torch.load(cp, map_location="cpu", weights_only=False)
    m = VARIANT_MAP["GateAngle"](vocab_size=blob["vocab_size"], d_model=blob["d_model"],
                                 n_heads=blob["n_heads"], n_layers=blob["n_layers"],
                                 grid_size=64).to(dev).eval()
    m.load_state_dict(blob["model_state"])
    rng = np.random.RandomState(777)
    ones, zeros = [], []
    for _ in range(n):
        bits = torch.from_numpy(rng.randint(0, 2, size=(1, L))).long().to(dev)
        g = torch.sigmoid(m.angle.gate(m.token_emb(bits)))[0]
        msk = bits[0].bool()
        ones.append(g[msk].mean().item()); zeros.append(g[~msk].mean().item())
    return float(np.mean(ones)), float(np.mean(zeros))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="GATE_PROBE.md")
    a = ap.parse_args()
    dev = torch.device(a.device)

    o = ["# Does the sigmoid gate suppress observation tokens?", "",
         "MapFormer's `action_to_lie` reads every token and must LEARN that",
         "`Delta ~= 0` on observations. A sigmoid gate is the natural shape for",
         "that. If it is what the gate is doing, it explains the sign flip: the",
         "torus is half observation tokens that should contribute nothing, while on",
         "parity every bit must contribute.", "",
         "Prediction, written before looking: **gate(action) >> gate(observation)**",
         "on the torus, and no such split on parity.", "",
         "## Torus", "", "| seed | gate on ACTION tokens | gate on OBSERVATION tokens | ratio |",
         "|---|---|---|---|"]
    ta, to = [], []
    for s in range(8):
        cp = Path(f"mapformer/runs/selective/torus/p0/GateAngle_s{s}/GateAngle.pt")
        if not cp.exists():
            continue
        x, y = torus_gate(cp, dev)
        ta.append(x); to.append(y)
        o.append(f"| {s} | {x:.4f} | {y:.4f} | {x/max(y,1e-9):.2f}x |")
    if ta:
        o.append(f"| **mean** | **{np.mean(ta):.4f}** | **{np.mean(to):.4f}** | "
                 f"**{np.mean(ta)/max(np.mean(to),1e-9):.2f}x** |")
    o += ["", "## Parity", "",
          "| seed | gate on bit=1 | gate on bit=0 | ratio |", "|---|---|---|---|"]
    pa, pb = [], []
    for s in range(6):
        cp = Path(f"mapformer/runs/selective/parity/GateAngle_s{s}/GateAngle_parity.pt")
        if not cp.exists():
            continue
        x, y = parity_gate(cp, dev)
        pa.append(x); pb.append(y)
        o.append(f"| {s} | {x:.4f} | {y:.4f} | {x/max(y,1e-9):.2f}x |")
    if pa:
        o.append(f"| **mean** | **{np.mean(pa):.4f}** | **{np.mean(pb):.4f}** | "
                 f"**{np.mean(pa)/max(np.mean(pb),1e-9):.2f}x** |")
    o += ["", "## Verdict", ""]
    if ta and to:
        r = np.mean(ta) / max(np.mean(to), 1e-9)
        if r > 1.5:
            o += [f"**Confirmed on the torus**: the gate is {r:.2f}x larger on "
                  f"action tokens than on observations. It has learned the "
                  f"suppression the low-rank map would otherwise have to arrange, "
                  f"which is a MECHANISM for the torus win rather than capacity — "
                  f"and it predicts the parity loss, since parity has no tokens "
                  f"that should be suppressed.", ""]
        else:
            o += [f"**NOT confirmed**: the gate is only {r:.2f}x larger on actions "
                  f"than on observations. The token-suppression story does not "
                  f"explain the torus win, and the capacity reading stands — "
                  f"GateAngle and NoBottleneck buy the same thing for the same "
                  f"~8k parameters.", ""]
    o += ["Inference only, on existing checkpoints. n=8 torus seeds, 24 "
          "trajectories each."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
