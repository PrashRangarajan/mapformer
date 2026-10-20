"""SEARCH_THEORY A6 (eval-only, CPU, post hoc): the clock left in the move code of the sign arms, in the same units as
rank_clock.py, and the |.| cone (Proposition C1: sigma >= 2 |delta| in every unwrapped |.| channel).

Per inverse pair (x, y) and channel: clock s = wrap(a_x + a_y) (the phase a backward move adds beyond the displacement),
map m = wrap(a_y) (the phase per unit displacement). Clock ratio = usage-weighted rms(s) / rms(m), usage = mean |q_c| over
move tokens x mean |k_c| over observation tokens (layer-0 input = the embedding, exact per token).
Also: share of usage on channels that are (i) cancelling (|s| < 0.1, |m| > 0.05), (ii) blind (|m| <= 0.05), (iii) clock-coupled.
  python3 sign_clock.py > sign_clock_out.txt
"""
import sys, math, json, collections
from pathlib import Path
sys.path.insert(0, "/home/prashr")
import numpy as np
import torch

R = Path("/home/prashr/mapformer/runs")


def wrapa(x):
    return np.arctan2(np.sin(x), np.cos(x))


def usage(sd, nb=32, d_head=64, n_act=2):
    e = sd["token_emb.weight"].double()
    x = torch.nn.functional.layer_norm(e, (e.shape[1],), sd["layers.0.norm1.weight"].double(), sd["layers.0.norm1.bias"].double())
    q = x @ sd["layers.0.q_proj.weight"].double().T + sd["layers.0.q_proj.bias"].double()
    k = x @ sd["layers.0.k_proj.weight"].double().T + sd["layers.0.k_proj.bias"].double()
    qn = torch.sqrt(q[:, 0::2] ** 2 + q[:, 1::2] ** 2); kn = torch.sqrt(k[:, 0::2] ** 2 + k[:, 1::2] ** 2)   # (V, H*nb), order h*nb+b
    return (qn[:n_act].mean(0) * kn[n_act:].mean(0)).numpy()


def angles(sd, signed, n_act):
    e = sd["token_emb.weight"].double(); Wi = sd["action_to_lie.w_in.weight"].double(); Wo = sd["action_to_lie.w_out.weight"].double()
    if "frozen_act_emb" in sd:
        e = e.clone(); e[:n_act] = sd["frozen_act_emb"].double()
    st = (e[:n_act] @ Wi.T) @ Wo.T
    if not signed:
        st = st.abs()
    return (st * sd["path_integrator.omega"].double().reshape(-1)).numpy()       # (A, C)


def readout(sd, signed, pairs):
    n_act = 2 * len(pairs)
    A = angles(sd, signed, n_act); u = usage(sd, n_act=n_act); w = u / u.sum()
    num = den = 0.0; canc = blind = 0.0; cone_viol = cone_n = 0
    for x, y in pairs:
        s = wrapa(A[x] + A[y]); m = wrapa(A[y])
        num += np.sum(w * s ** 2); den += np.sum(w * m ** 2)
        canc += np.sum(w * ((np.abs(s) < 0.1) & (np.abs(m) > 0.05))); blind += np.sum(w * (np.abs(m) <= 0.05))
        if not signed:                                   # Proposition C1 on raw (unwrapped) angles below pi
            sig = np.abs(A[x] + A[y]); dl = np.abs(A[x] - A[y]) / 2; sel = sig < math.pi
            cone_viol += int(np.sum(sig[sel] < 2 * dl[sel] - 1e-9)); cone_n += int(sel.sum())
    k = len(pairs)
    return {"clock": math.sqrt(num / den), "cancel": canc / k, "blind": blind / k, "cone_viol": cone_viol, "cone_n": cone_n}


def main():
    sets = []
    for cell in ("SO", "SO_absmag", "SO_high", "AO", "AO_absmag", "AO_mid", "AO_high"):
        sets.append(("ring MECH", cell, [R / f"wrap_mech/p0/{cell}_s{s}/{cell}.pt" for s in range(800, 808)], cell.startswith("SO"), ((0, 1),)))
    for cell in ("SO_absmag_fz", "SO_same_fz"):
        sets.append(("ring ACCOUNTS T2", cell, [R / f"accounts/t2/{cell}_s{s}/{cell}.pt" for s in range(1000, 1008)], True, ((0, 1),)))
    for cell in ("SO_m", "AO_m"):
        sets.append(("ring GAUGE", cell, [R / f"gauge/p0/{cell}_s{s}/{cell[:2]}.pt" for s in range(1300, 1308)], cell.startswith("SO"), ((0, 1),)))
    for cell in ("SO4", "AO4", "SO4_high", "AO4_high", "AO4_mid"):
        sets.append(("torus TORUS_START", cell, [R / f"torus_start/p0/{cell}_s{s}/{cell}.pt" for s in range(1200, 1208)], cell.startswith("SO"),
                     ((0, 1), (2, 3))))
    out = []
    for batch, cell, paths, signed, pairs in sets:
        rs = []
        for p in paths:
            if not p.exists():
                continue
            d = torch.load(p, map_location="cpu", weights_only=False); sd = d["model_state_dict"]
            L = d["losses"]; final = float(np.mean(L[-max(1, len(L) // 20):]))
            r = readout(sd, signed, pairs); r["solved"] = final < 0.05; r["final"] = final; rs.append(r)
        if not rs:
            print(f"{batch:18s} {cell:12s} no checkpoints"); continue
        f = lambda k, sel=None: np.mean([r[k] for r in rs if sel is None or r["solved"] == sel]) if any(sel is None or r["solved"] == sel for r in rs) else float("nan")
        cv = sum(r["cone_viol"] for r in rs); cn = sum(r["cone_n"] for r in rs)
        print(f"{batch:18s} {cell:12s} SOLVED {sum(r['solved'] for r in rs)}/{len(rs)} | clock ratio mean {f('clock'):.3f} "
              f"(solved {f('clock', True):.3f}, unsolved {f('clock', False):.3f}) | usage on cancelling {f('cancel'):.2f}, blind {f('blind'):.2f}"
              + (f" | cone violations {cv}/{cn}" if not signed else ""))
        print("      per run clock: " + " ".join(f"{r['clock']:.2f}{'*' if r['solved'] else ''}" for r in rs))
        out.append({"batch": batch, "cell": cell, "runs": rs})
    json.dump(out, open(Path(__file__).with_name("sign_clock_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
