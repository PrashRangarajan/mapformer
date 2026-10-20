"""SEARCH_THEORY A1 (eval-only, CPU, post hoc): latent geometry of the two move tokens in observation-step-free ring models.

For each checkpoint (and its reconstructed initialisation) report, per run:
  cos(z1, z2), |z2| / |z1|                      move latents z = W_in e(move)
  rel residual of the relation in latent space  +1/-1: |z1 + z2| / (|z1| + |z2|);  +1/+2: min over the two |.|-mirrors
                                                  |z2 -+ 2 z1| / (|z2| + 2 |z1|)
  pre-|.| sign coherence                        share of channels (both |angle| > 0.05) whose PRE-|.| steps w_c.z1, w_c.z2
                                                  have the same sign; coherence = max(share, 1 - share)
  per-channel collapse pattern                  channels with 0 / 1 / 2 move angles within 0.05 rad of 0 (mod 2 pi)
  cone tightness (|.| only, ring +1/-1)         share of channels with sigma < pi (raw angle sum) whose map part
                                                  |delta| = |a1 - a2| / 2 is within 10% of its bound sigma / 2
Inits are rebuilt exactly as the trainers do (torch.manual_seed(seed); np.random.seed(seed); env; model) and checked
against the stored default angles where the run saved them (MECHANISM eval.json 'default_angles').
  python3 latent_geometry.py > latent_geometry_out.txt
"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, "/home/prashr")
import numpy as np
import torch

from mapformer.environment_cancel import GridWorldCancel
from mapformer.environment_steps import GridWorldSteps
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4

torch.set_num_threads(2)
R = Path("/home/prashr/mapformer/runs")


def wrapa(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def build(signed, seed, deltas=None):
    torch.manual_seed(seed); np.random.seed(seed)
    env = GridWorldCancel(size=32, seed=seed, p_plus=0.5) if deltas is None else \
        GridWorldSteps(size=32, seed=seed, p_first=0.5, deltas=deltas)
    kw = dict(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=32,
              action_ids=(env.action_offset, env.action_offset + 1))
    return (MapFormerWM_SignedAO_r4 if signed else MapFormerWM_AbsAO_r4)(**kw)


@torch.no_grad()
def stats(m, rel, signed):
    e = m.token_emb.weight.double(); Win = m.action_to_lie.w_in.weight.double(); Wout = m.action_to_lie.w_out.weight.double()
    z = e[m.act] @ Win.T                                    # (2, r)
    pre = z @ Wout.T                                        # (2, C) pre-|.| steps
    om = m.path_integrator.omega.double().reshape(-1)
    ang = (pre if signed else pre.abs()) * om               # (2, C)
    z1, z2 = z[0], z[1]
    cos = float(z1 @ z2 / (z1.norm() * z2.norm()))
    if rel == "m":
        res = float((z1 + z2).norm() / (z1.norm() + z2.norm()))
    else:
        res = float(min((z2 - 2 * z1).norm(), (z2 + 2 * z1).norm()) / (z2.norm() + 2 * z1.norm()))
    big = (wrapa(ang[0]).abs() > 0.05) & (wrapa(ang[1]).abs() > 0.05)
    same = (torch.sign(pre[0]) == torch.sign(pre[1]))[big].double()
    share = float(same.mean()) if same.numel() else float("nan")
    col = (wrapa(ang).abs() <= 0.05).sum(0)                 # per channel: number of collapsed move angles
    pat = [int((col == k).sum()) for k in (0, 1, 2)]
    out = {"cos": cos, "ratio": float(z2.norm() / z1.norm()), "res": res, "same_share": share,
           "coherence": max(share, 1 - share), "n_big": int(big.sum()), "collapse_0_1_2": pat}
    if rel == "m" and not signed:
        sig = ang.sum(0); dl = (ang[0] - ang[1]).abs() / 2
        sel = (sig < math.pi) & (sig > 0.05)
        tight = (dl[sel] >= 0.9 * sig[sel] / 2).double()
        out["cone_tight"] = float(tight.mean()) if tight.numel() else float("nan"); out["n_cone"] = int(sel.sum())
    return out


def load(path, signed, seed, deltas=None):
    m = build(signed, seed, deltas)
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    m.load_state_dict(sd, strict=False)
    return m


def fmt(d):
    s = f"cos {d['cos']:+.3f} ratio {d['ratio']:.2f} res {d['res']:.3f} same {d['same_share']:.2f} coh {d['coherence']:.2f} " \
        f"(n {d['n_big']:2d}) collapse0/1/2 {d['collapse_0_1_2']}"
    if "cone_tight" in d:
        s += f" cone_tight {d['cone_tight']:.2f} (n {d['n_cone']})"
    return s


def main():
    rows = []
    print("== reconstruction check (MECHANISM default angles) ==")
    for seed in range(800, 808):
        m = build(True, seed)
        with torch.no_grad():
            st = m.action_to_lie.w_out(m.action_to_lie.w_in(m.token_emb.weight[m.act])) * m.path_integrator.omega.reshape(-1)
        ref = torch.tensor(json.load(open(R / f"wrap_mech/p0/SO_s{seed}/eval.json"))["default_angles"])
        print(f"seed {seed}: max |rebuilt - stored default angle| = {float((st - ref).abs().max()):.2e}")

    print("\n== GAUGE (seeds 1300-1307): init and end ==")
    for cell, signed, deltas, rel in (("SO_m", True, (1, -1), "m"), ("AO_m", False, (1, -1), "m"),
                                       ("SO_p", True, (1, 2), "p"), ("AO_p", False, (1, 2), "p")):
        print(f"-- {cell}")
        for seed in range(1300, 1308):
            d0 = stats(build(signed, seed, deltas), rel, signed)
            p = R / f"gauge/p0/{cell}_s{seed}"
            ev = json.load(open(p / "eval.json"))
            m = load(p / f"{cell[:2]}.pt", signed, seed, deltas)
            d1 = stats(m, rel, signed)
            solved = ev["final_loss"] < 0.05
            print(f"  s{seed} {'SOLV' if solved else '----'} acc {ev['eval']['128']['acc']:.3f} loss {ev['final_loss']:.3f}")
            print(f"     init: {fmt(d0)}")
            print(f"     end : {fmt(d1)}")
            rows.append({"batch": "gauge", "cell": cell, "seed": seed, "solved": bool(solved),
                         "acc": ev["eval"]["128"]["acc"], "final_loss": ev["final_loss"], "init": d0, "end": d1})

    print("\n== MECHANISM (seeds 800-807): end (init = the arm's start, rebuilt only for SO/AO) ==")
    for cell in ("SO", "SO_absmag", "SO_high", "AO", "AO_absmag", "AO_mid", "AO_high"):
        signed = cell.startswith("SO")
        print(f"-- {cell}")
        for seed in range(800, 808):
            p = R / f"wrap_mech/p0/{cell}_s{seed}"
            if not (p / f"{cell}.pt").exists():
                continue
            ev = json.load(open(p / "eval.json"))
            d1 = stats(load(p / f"{cell}.pt", signed, seed), "m", signed)
            solved = ev["final_loss"] < 0.05
            print(f"  s{seed} {'SOLV' if solved else '----'} acc {ev['eval']['128']['acc']:.3f} end: {fmt(d1)}")
            rows.append({"batch": "mech", "cell": cell, "seed": seed, "solved": bool(solved),
                         "acc": ev["eval"]["128"]["acc"], "final_loss": ev["final_loss"], "end": d1})
    json.dump(rows, open(Path(__file__).with_name("latent_geometry_rows.json"), "w"), indent=1)

    print("\n== summary (means over seeds) ==")
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["batch"], r["cell"])].append(r)
    for k, rs in g.items():
        def mean(f):
            v = [f(r) for r in rs if not (isinstance(f(r), float) and math.isnan(f(r)))]
            return sum(v) / len(v) if v else float("nan")
        line = f"{k[0]:6s} {k[1]:10s} solved {sum(r['solved'] for r in rs)}/{len(rs)} | end: res {mean(lambda r: r['end']['res']):.3f} " \
               f"cos {mean(lambda r: r['end']['cos']):+.3f} coh {mean(lambda r: r['end']['coherence']):.2f} " \
               f"1-collapsed ch {mean(lambda r: r['end']['collapse_0_1_2'][1]):.1f} 2-collapsed {mean(lambda r: r['end']['collapse_0_1_2'][2]):.1f}"
        if "init" in rs[0]:
            line += f" | init: res {mean(lambda r: r['init']['res']):.3f} coh {mean(lambda r: r['init']['coherence']):.2f}"
        if "cone_tight" in rs[0]["end"]:
            line += f" | cone_tight {mean(lambda r: r['end']['cone_tight']):.2f}"
        print(line)


if __name__ == "__main__":
    main()
