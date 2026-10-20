"""SEARCH_THEORY A5 (eval-only, CPU, post hoc): where does the common mode (clock) of the move steps go in the rank arms?

Torus T = 1024 (RANK_MI / RANK_SEP / RANK3 checkpoints, full MapWM: observations step too). Moves 0..3 = (-x, +x, -y, +y).
Per run and head h, with latents z_t = W_in^h e_t (per head) or the shared latent (read by head h through W_out^h):
  common mode (clock) latents  c_x = z_0 + z_1, c_y = z_2 + z_3; map latents d_x = (z_1 - z_0)/2, d_y = (z_3 - z_2)/2
  latent clock ratio           (|W c_x| + |W c_y|) / (|W d_x| + |W d_y|)  -- with W = I (raw latent) and W = W_out^h
  channel clock ratio          usage-weighted rms of the wrapped clock angles omega * W_out^h c over rms of the map angles
  clock-in-map-span            share of the (usage-weighted) clock angle vector inside span{map_x, map_y} (per head)
  obs clock                    usage-weighted rms of the mean observation step angle over the map's
Usage per channel = mean |q_c| over move tokens x mean |k_c| over observation tokens (layer-0 inputs are the embeddings, so
this is exact per token; `model_wrap_exist.usage_weights` up to token frequencies).
  python3 rank_clock.py > rank_clock_out.txt
"""
import sys, math, collections
from pathlib import Path
sys.path.insert(0, "/home/prashr")
import numpy as np
import torch

R = Path("/home/prashr/mapformer/runs")
ARMS = [("shared r2", "rank_mi", "Vanilla"), ("per-head r2", "rank_mi", "Vanilla_r2ph"),
        ("blockdiag r4 (2/head)", "rank_sep", "Vanilla_r4mibd"), ("per-head r3", "rank3", "Vanilla_r3ph"),
        ("shared r4", "rank_mi", "Vanilla_r4mi"), ("per-head r4", "rank_sep", "Vanilla_r4ph")]


def wrapa(x):
    return np.arctan2(np.sin(x), np.cos(x))


def head_latents(sd, h, nb=32):
    """(V, r_h) latents seen by head h and (nb, r_h) readout W_h."""
    e = sd["token_emb.weight"].double().numpy()
    if "action_to_lie.w_out.weight" in sd:                       # shared latent (masked block-diagonal if a mask exists)
        Wi = sd["action_to_lie.w_in.weight"].double().numpy(); Wo = sd["action_to_lie.w_out.weight"].double().numpy()
        if "action_to_lie.mask" in sd:
            Wo = Wo * sd["action_to_lie.mask"].double().numpy()
        Wh = Wo[h * nb:(h + 1) * nb]
        keep = np.abs(Wh).sum(0) > 0                             # latent dims this head reads
        return e @ Wi[keep].T, Wh[:, keep]
    Wi = sd["action_to_lie.w_in.weight"].double().numpy(); Wh = sd[f"action_to_lie.w_out.{h}.weight"].double().numpy()
    r = Wh.shape[1]
    return e @ Wi[h * r:(h + 1) * r].T, Wh


def usage(sd, h, nb=32, d_head=64):
    e = sd["token_emb.weight"].double()
    g, b = sd["layers.0.norm1.weight"].double(), sd["layers.0.norm1.bias"].double()
    x = torch.nn.functional.layer_norm(e, (e.shape[1],), g, b)
    q = (x @ sd["layers.0.q_proj.weight"].double().T + sd["layers.0.q_proj.bias"].double())[:, h * d_head:(h + 1) * d_head]
    k = (x @ sd["layers.0.k_proj.weight"].double().T + sd["layers.0.k_proj.bias"].double())[:, h * d_head:(h + 1) * d_head]
    qn = torch.sqrt(q[:, 0::2] ** 2 + q[:, 1::2] ** 2); kn = torch.sqrt(k[:, 0::2] ** 2 + k[:, 1::2] ** 2)
    return (qn[:4].mean(0) * kn[4:].mean(0)).numpy()             # (nb,)


def main():
    rows = collections.defaultdict(list)
    for label, batch, var in ARMS:
        print(f"== {label} ({batch}/{var})")
        for s in range(8):
            p = R / f"{batch}/p0/{var}_s{s}/{var}.pt"
            if not p.exists():
                continue
            d = torch.load(p, map_location="cpu", weights_only=False); sd = d["model_state_dict"]
            L = d["losses"]; final = float(np.mean(L[-max(1, len(L) // 20):])); solved = final < 0.05
            om = sd["path_integrator.omega"].double().numpy()
            per = []
            for h in range(2):
                Z, Wh = head_latents(sd, h); u = usage(sd, h)
                cx, cy = Z[0] + Z[1], Z[2] + Z[3]; dx, dy = (Z[1] - Z[0]) / 2, (Z[3] - Z[2]) / 2
                zo = Z[4:].mean(0)
                lat = (np.linalg.norm(cx) + np.linalg.norm(cy)) / (np.linalg.norm(dx) + np.linalg.norm(dy))
                A = lambda v: om[h] * (Wh @ v)                   # angles per channel
                Cx, Cy, Mx, My, O = wrapa(A(cx)), wrapa(A(cy)), A(dx), A(dy), wrapa(A(zo))
                w = u / u.sum()
                clock = math.sqrt(np.sum(w * (Cx ** 2 + Cy ** 2))); mapn = math.sqrt(np.sum(w * (Mx ** 2 + My ** 2)))
                ob = math.sqrt(np.sum(w * O ** 2))
                sw = np.sqrt(w); B = np.stack([sw * Mx, sw * My], 1)
                Q, _ = np.linalg.qr(B)
                ins = []
                for C in (Cx, Cy):
                    v = sw * C; nv = np.linalg.norm(v)
                    ins.append(float(np.linalg.norm(Q.T @ v) ** 2 / nv ** 2) if nv > 1e-12 else float("nan"))
                per.append({"lat": lat, "ch_clock": clock / mapn, "in_span": float(np.nanmean(ins)), "obs": ob / mapn,
                            "r_h": Z.shape[1]})
            print(f"  s{s} {'SOLV' if solved else '----'} loss {final:.3f} | " + " | ".join(
                f"h{h} r{q['r_h']} lat-clock {q['lat']:.3f} ch-clock {q['ch_clock']:.3f} in-span {q['in_span']:.2f} obs {q['obs']:.3f}"
                for h, q in enumerate(per)))
            for q in per:
                rows[label].append((solved, q))
    print("\n== summary over heads (mean; SOLVED runs / unsolved runs) ==")
    for label, rs in rows.items():
        def m(sel, k):
            v = [q[k] for sv, q in rs if sv == sel and not math.isnan(q[k])]
            return (f"{np.mean(v):.3f}" if v else "  -  ") + f" (n {len(v)})"
        print(f"{label:24s} lat-clock S {m(True, 'lat')} U {m(False, 'lat')} | ch-clock S {m(True, 'ch_clock')} U {m(False, 'ch_clock')}"
              f" | in-span S {m(True, 'in_span')} U {m(False, 'in_span')} | obs S {m(True, 'obs')} U {m(False, 'obs')}")


if __name__ == "__main__":
    main()
