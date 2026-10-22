"""One clock / map definition for every row of MAIN_DRAFT2 Table 4 (POST HOC, eval only, CPU, ~1 min).

Written to apply docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md B3 and M-SG1: Table 4 of draft 2 put sign_clock.py's
ratio (map = wrap(a_y), heads pooled) and rank_clock.py's (map = unwrapped half-difference, per head) in one column, and
sign_clock's map used one move of each inverse pair only (rule 8: it depends on which move is called y). This script
computes, for the sign arms of sign_clock.py and the rank arms of rank_clock.py alike:

  per inverse pair (x, y) and rotary channel c:  clock s_c = wrap(a_c(x) + a_c(y)),
                                                 map   m_c^2 = (wrap(a_c(x))^2 + wrap(a_c(y))^2) / 2   (symmetric)
  clock / map = sqrt( sum_pairs sum_c w_c s_c^2 / sum_pairs sum_c w_c m_c^2 )

with w = usage (mean |q_c| over move tokens x mean |k_c| over observation tokens; layer-0 inputs are the embeddings, so
this is exact per token), normalised over all channels of both heads: one ratio per run ("run"). The same ratio with w
normalised within one head gives a per-head value ("head"). Angles a_c(x) = omega_c * step_c(x) for the move tokens, with
|.| applied per channel after W_out for the |.| arms, `frozen_act_emb` for ACCOUNTS T2, and the block-diagonal mask for
RANK_SEP's Vanilla_r4mibd. As a check the script also prints the run ratio with sign_clock.py's one-sided map wrap(a_y)
("ay"), which reproduces sign_clock_out.txt's arm means for the sign arms, and per run the usage share of channels whose
per-pair clock is near pi (|wrap(k)| > 2.6 rad), where Proposition 2(a)'s bound says nothing.
SOLVED = mean training loss over the final 5% of epochs < 0.05 (stats_core.classify_run's registered rule).
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/expansion/scripts/sign/clock_table.py \
      > mapformer/docs/paper/expansion/scripts/sign/clock_table_out.txt
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
R = Path("/home/prashr/mapformer/runs")
NB = 32
PI_CUT = 2.6          # rad: a per-pair clock 'near pi' (|wrap(k)| > 2.6), reported as a usage share per run


def wrapa(x):
    return np.arctan2(np.sin(x), np.cos(x))


def usage(sd, n_act):
    """(H*NB,) usage per channel, order h*NB + b."""
    e = sd["token_emb.weight"].double()
    x = torch.nn.functional.layer_norm(e, (e.shape[1],), sd["layers.0.norm1.weight"].double(), sd["layers.0.norm1.bias"].double())
    q = x @ sd["layers.0.q_proj.weight"].double().T + sd["layers.0.q_proj.bias"].double()
    k = x @ sd["layers.0.k_proj.weight"].double().T + sd["layers.0.k_proj.bias"].double()
    qn = torch.sqrt(q[:, 0::2] ** 2 + q[:, 1::2] ** 2); kn = torch.sqrt(k[:, 0::2] ** 2 + k[:, 1::2] ** 2)
    return (qn[:n_act].mean(0) * kn[n_act:].mean(0)).numpy()


def move_angles(sd, signed, n_act):
    """(n_act, H*NB) omega-scaled angles of the move tokens."""
    e = sd["token_emb.weight"].double().numpy()
    if "frozen_act_emb" in sd:
        e = e.copy(); e[:n_act] = sd["frozen_act_emb"].double().numpy()
    om = sd["path_integrator.omega"].double().numpy().reshape(-1)
    Wi = sd["action_to_lie.w_in.weight"].double().numpy()
    if "action_to_lie.w_out.weight" in sd:                                   # shared latent (optionally masked)
        Wo = sd["action_to_lie.w_out.weight"].double().numpy()
        if "action_to_lie.mask" in sd:
            Wo = Wo * sd["action_to_lie.mask"].double().numpy()
        st = (e[:n_act] @ Wi.T) @ Wo.T
    else:                                                                    # per-head latents
        H = len([k for k in sd if k.startswith("action_to_lie.w_out.")])
        parts = []
        for h in range(H):
            Wh = sd[f"action_to_lie.w_out.{h}.weight"].double().numpy(); r = Wh.shape[1]
            parts.append((e[:n_act] @ Wi[h * r:(h + 1) * r].T) @ Wh.T)
        st = np.concatenate(parts, 1)
    if not signed:
        st = np.abs(st)
    return st * om


def ratios(sd, signed, pairs):
    n_act = 2 * len(pairs)
    A = move_angles(sd, signed, n_act); u = usage(sd, n_act)
    H = A.shape[1] // NB

    def ratio(w, sym=True):
        num = den = 0.0
        for x, y in pairs:
            s = wrapa(A[x] + A[y])
            m2 = (wrapa(A[x]) ** 2 + wrapa(A[y]) ** 2) / 2 if sym else wrapa(A[y]) ** 2
            num += np.sum(w * s ** 2); den += np.sum(w * m2)
        return math.sqrt(num / den) if den > 0 else float("nan")

    run = ratio(u / u.sum()); ay = ratio(u / u.sum(), sym=False)
    w = u / u.sum()
    pi_share = float(np.mean([np.sum(w * (np.abs(wrapa(A[x] + A[y])) > PI_CUT)) for x, y in pairs]))
    heads = []
    for h in range(H):
        w = np.zeros_like(u); w[h * NB:(h + 1) * NB] = u[h * NB:(h + 1) * NB]
        heads.append(ratio(w / w.sum()))
    return run, ay, heads, pi_share


def sets():
    out = []
    for cell in ("SO", "SO_absmag", "SO_high", "AO", "AO_absmag", "AO_mid", "AO_high"):
        out.append(("ring MECHANISM", cell, [R / f"wrap_mech/p0/{cell}_s{s}/{cell}.pt" for s in range(800, 808)], cell.startswith("SO"), ((0, 1),)))
    for cell in ("SO_absmag_fz", "SO_same_fz"):
        out.append(("ring ACCOUNTS T2", cell, [R / f"accounts/t2/{cell}_s{s}/{cell}.pt" for s in range(1000, 1008)], True, ((0, 1),)))
    for cell in ("SO_m", "AO_m"):
        out.append(("ring GAUGE", cell, [R / f"gauge/p0/{cell}_s{s}/{cell[:2]}.pt" for s in range(1300, 1308)], cell.startswith("SO"), ((0, 1),)))
    for cell in ("SO4", "AO4", "SO4_high", "AO4_high", "AO4_mid"):
        out.append(("torus TORUS_START", cell, [R / f"torus_start/p0/{cell}_s{s}/{cell}.pt" for s in range(1200, 1208)], cell.startswith("SO"),
                    ((0, 1), (2, 3))))
    for label, batch, var in (("shared r2", "rank_mi", "Vanilla"), ("per-head r2", "rank_mi", "Vanilla_r2ph"),
                              ("blockdiag r4 (2/head)", "rank_sep", "Vanilla_r4mibd"), ("per-head r3", "rank3", "Vanilla_r3ph"),
                              ("shared r4", "rank_mi", "Vanilla_r4mi"), ("per-head r4", "rank_sep", "Vanilla_r4ph")):
        out.append((f"torus rank", label, [R / f"{batch}/p0/{var}_s{s}/{var}.pt" for s in range(8)], True, ((0, 1), (2, 3))))
    return out


def main():
    print(__doc__.split("\n")[0])
    print("definition: symmetric map, usage pooled over heads (run) / within a head (head); 'ay' = sign_clock.py's map\n")
    for batch, cell, paths, signed, pairs in sets():
        rs = []
        for p in paths:
            if not p.exists():
                continue
            d = torch.load(p, map_location="cpu", weights_only=False); sd = d["model_state_dict"]
            L = d["losses"]; final = float(np.mean(L[-max(1, len(L) // 20):]))
            run, ay, heads, pis = ratios(sd, signed, pairs)
            rs.append(dict(solved=final < 0.05, run=run, ay=ay, heads=heads, pis=pis))
        if not rs:
            print(f"{batch:18s} {cell:22s} no checkpoints"); continue
        ns = sum(r["solved"] for r in rs)

        def agg(sel):
            v = [r["run"] for r in rs if sel is None or r["solved"] == sel]
            hv = [h for r in rs if sel is None or r["solved"] == sel for h in r["heads"]]
            if not v:
                return "  --  "
            return (f"run mean {np.mean(v):.3f} median {np.median(v):.3f} range {min(v):.3f}-{max(v):.3f}; "
                    f"heads median {np.median(hv):.3f} range {min(hv):.3f}-{max(hv):.3f}")
        print(f"{batch:18s} {cell:22s} SOLVED {ns}/{len(rs)} | ay-map run mean {np.mean([r['ay'] for r in rs]):.3f}")
        print(f"      all      {agg(None)}")
        print(f"      solved   {agg(True)}")
        print(f"      unsolved {agg(False)}")
        uns = [r for r in rs if not r["solved"]]
        if uns:
            print(f"      unsolved runs with every head < 0.15: {sum(all(h < 0.15 for h in r['heads']) for r in uns)}/{len(uns)}; "
                  f"heads < 0.15: {sum(h < 0.15 for r in uns for h in r['heads'])}/{sum(len(r['heads']) for r in uns)}")
        print("      per run (run ratio [heads]; * = SOLVED): " + "  ".join(
            f"{r['run']:.3f}[{'/'.join(f'{h:.2f}' for h in r['heads'])}]{'*' if r['solved'] else ''}" for r in rs))
        print("      per run, usage share of channels with a per-pair clock near pi (|wrap(k)| > 2.6 rad): " + "  ".join(
            f"{r['pis']:.2f}{'*' if r['solved'] else ''}" for r in rs))


if __name__ == "__main__":
    main()
