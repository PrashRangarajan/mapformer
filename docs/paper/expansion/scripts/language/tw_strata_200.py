"""Text-world accuracy by retrace stratum on ALL 200 registered evaluation walks (POST HOC, eval only, CPU).

Successor of tw_strata.py (which used the first 100 of the 200 walks), written to apply the analysis-script audit
(docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md, B1 and M-LG1). tw_strata.py and its output are left unchanged.

Stream: held-out map 10000, np seed 0, the first N_WALKS walks (default 200 = the registered TEXTWORLD stream on which
tw_retrace_floor.py gives retrace 0.9337). Per revisit target the retrace rule of tw_retrace_floor.py either APPLIES
(the current run undoes the previous one and has not overshot it: predict the observation 2j steps back) or does not
(predict 'nothing'). The rule is never wrong when it applies (checked and printed). Strata:
  NON-RETRACE  the rule does not apply: places reached by another route than undoing the last run (outcome-free).
  NEEDS-MAP    NON-RETRACE and the target is an object (= the revisits retrace gets wrong; on blank NON-RETRACE
               targets retrace's default 'nothing' is right by construction), the tw_strata.py stratum.
  RETRACE-OK   the complement of NEEDS-MAP.
  applies      the rule applies (the complement of NON-RETRACE).
Also prints the best constant on each stratum (the third line of tw_retrace_floor_out.txt, which had no generator).
Path models are read with the x1.111 attention-dropout re-score and in eval mode; RoPE models in eval mode.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 \
    mapformer/docs/paper/expansion/scripts/language/tw_strata_200.py > .../language/tw_strata_200walks_out.txt
  (N_WALKS=100 reproduces tw_strata_out.txt's per-model RETRACE-OK / NEEDS-MAP columns.)
"""
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/language")
from tw_lib import TextWorld, gen, load, ck_tw, delta_of, forward, SCALE, TW  # noqa
from mapformer.train_variant import VARIANT_MAP  # noqa

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
N_WALKS = int(os.environ.get("N_WALKS", "200"))


def labels(g, nothing):
    """slot index -> (applies, target, retrace prediction)."""
    slots = np.nonzero(g["obs"])[0]
    a = g["dirs"][:len(slots)]; o = g["tok"][slots]; r = g["rev"][slots]
    out = {}
    prev_run = cur = 0
    for t in range(len(slots)):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
            cur = 1
        if r[t]:
            applies = prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0
            pr = o[t - 2 * cur] if applies else nothing
            out[int(slots[t])] = (bool(applies), int(o[t]), int(pr))
    return out


def load_rope(L, s, V):
    b = torch.load(f"{TW}/RoPE_L{L}_s{s}/RoPE.pt", map_location="cpu", weights_only=False)
    m = VARIANT_MAP["RoPE"](vocab_size=V, d_model=128, n_heads=2, n_layers=L, grid_size=64)
    m.load_state_dict(b["model_state_dict"]); return m.eval()


te = TextWorld(size=64, seed=10000); np.random.seed(0)
NOTHING = te.idx["nothing"]
G = [gen(te, 1024) for _ in range(N_WALKS)]
LB = [labels(g, NOTHING) for g in G]
rows = [v for lb in LB for v in lb.values()]
n = len(rows)
app = [v for v in rows if v[0]]
non = [v for v in rows if not v[0]]
nmap = [v for v in non if v[1] != NOTHING]
wrong_when_applies = sum(v[2] != v[1] for v in app)
ret = sum(v[2] == v[1] for v in rows) / n
print(f"stream: first {N_WALKS} walks of the TEXTWORLD eval stream (map 10000, np seed 0): {n} revisit targets; "
      f"retrace correct {ret:.4f}")
print(f"  rule applies {len(app)} ({len(app) / n:.3f}), wrong when it applies {wrong_when_applies}; "
      f"NON-RETRACE {len(non)} ({len(non) / n:.3f}; blank share {np.mean([v[1] == NOTHING for v in non]):.3f}); "
      f"NEEDS-MAP = NON-RETRACE and object target {len(nmap)} ({len(nmap) / n:.3f}); RETRACE-OK {n - len(nmap)}")
inv = {i: w for w, i in te.idx.items()}
for name, sub in (("all revisits", rows), ("NON-RETRACE", non), ("NEEDS-MAP", nmap)):
    c = collections.Counter(v[1] for v in sub)
    tok, cnt = c.most_common(1)[0]
    print(f"  best constant on {name}: {cnt / len(sub):.4f} ('{inv[tok]}'); share 'nothing' {c[NOTHING] / len(sub):.4f}")


def score(pred_fn):
    """accuracy on: all, retrace applies, RETRACE-OK, NEEDS-MAP, NON-RETRACE, NON-RETRACE blank targets."""
    acc = collections.defaultdict(list)
    for g, lb in zip(G, LB):
        pred = pred_fn(g).argmax(-1).numpy()
        for i, (applies, tgt, pr) in lb.items():
            hit = bool(pred[i - 1] == g["tok"][i])
            acc["all"].append(hit)
            needs = (not applies) and tgt != NOTHING
            acc["needs_map" if needs else "retrace_ok"].append(hit)
            if applies:
                acc["applies"].append(hit)
            else:
                acc["non_retrace"].append(hit)
                if tgt == NOTHING:
                    acc["non_retrace_blank"].append(hit)
    return {k: float(np.mean(v)) for k, v in acc.items()}


KEYS = ("all", "applies", "retrace_ok", "needs_map", "non_retrace", "non_retrace_blank")
V = len(te.vocab)
res = collections.defaultdict(list)
for s in range(8):
    m, _ = load(ck_tw(s))
    r1 = score(lambda g: forward(m, g["tok"], delta_of(m, g["tok"]), scale=SCALE))
    r0 = score(lambda g: forward(m, g["tok"], delta_of(m, g["tok"]), scale=1.0))
    res["path"].append(r1); res["path_x1"].append(r0)
    print(f"path s{s} re-scored: " + " ".join(f"{k} {r1[k]:.4f}" for k in KEYS), flush=True)
    print(f"path s{s} eval mode: " + " ".join(f"{k} {r0[k]:.4f}" for k in KEYS), flush=True)
    for L in (1, 2):
        mr = load_rope(L, s, V)
        with torch.no_grad():
            rr = score(lambda g: mr(torch.as_tensor(g["tok"])[None])[0])
        res[f"rope{L}"].append(rr)
        print(f"  RoPE {L}L s{s} eval mode: " + " ".join(f"{k} {rr[k]:.4f}" for k in KEYS), flush=True)
print()
for arm, v in res.items():
    for k in KEYS:
        x = np.array([r[k] for r in v])
        print(f"MEAN {arm:8s} {k:18s}: all seeds {x.mean():.4f} (min {x.min():.4f}, max {x.max():.4f}); "
              f"fresh seeds 2-7 {x[2:].mean():.4f} (min {x[2:].min():.4f})")
