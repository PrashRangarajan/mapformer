"""Per-stratum map-free floors for fig_ring_strata (post hoc, CPU, no model). Stream and strata exactly as ring_strata.py
(GridWorldCancel(32, seed 10000, p_plus 0.5), np seed 10**6, 400 walks of T = 128; its stream() is reused). Retrace = the
one-level rule of docs/audits/2026-10-16/ring_floors.py (while a run of actions reverses the previous run, step j copies
the observation 2j tokens back; else blank). Check: the overall retrace on this stream is printed beside ring_floors'
0.8596 (another stream of the same task: 2000 walks, np seed 4242).
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/ring_strata_floors.py"""
import json, os
import numpy as np
from ring_strata import stream, GAP_BINS

HERE = os.path.dirname(os.path.abspath(__file__))
toks, S, te = stream()
BL = te.blank_token + te.obs_offset
rev = np.stack([s[0] for s in S]); gap = np.stack([s[1] for s in S]); wo = np.stack([s[2] for s in S])
tok = toks.numpy(); pred = np.full(rev.shape, BL)
for w in range(tok.shape[0]):
    a = tok[w, 0::2]; o = tok[w, 1::2]; prev_run = cur = 0
    for t in range(len(a)):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_run = cur if t > 0 else 0
            cur = 1
        if prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0:
            pred[w, t] = o[t - 2 * cur]
tgt = tok[:, 1::2]; hit = pred == tgt; blank = tgt == BL
masks = {f"plain gap {lo}-{hi - 1}": rev & ~wo & (gap >= lo) & (gap < hi) for lo, hi in GAP_BINS}
masks["wrap-only"] = rev & wo; masks["all revisits"] = rev
res = {k: {"n": int(m.sum()), "retrace": float(hit[m].mean()), "blank": float(blank[m].mean())} for k, m in masks.items()}
lines = [f"{k:18s} n {v['n']:6d}  blank {v['blank']:.4f}  retrace {v['retrace']:.4f}" for k, v in res.items()]
lines.append("(ring_floors_out.txt, other stream, p_plus 0.5: blank 0.4967, retrace 0.8596)")
print("\n".join(lines)); open(f"{HERE}/ring_strata_floors_out.txt", "w").write("\n".join(lines) + "\n")
json.dump(res, open(f"{HERE}/ring_strata_floors.json", "w"), indent=1)
