"""Text-world accuracy by retrace stratum (POST HOC, eval only, CPU).

The text world's walks are ~132 moves long, the regime in which the main paper finds that a map-free retrace rule
already scores 0.934 on the torus (Sec. 4). tw_retrace_floor.py gives retrace 0.934 on the text world's registered eval
stream too. Here every model of the registered batch (runs/textworld/p0: path r4 1 layer, RoPE 1 and 2 layers; 8 seeds
each) is scored on that stream (held-out map 10000, np seed 0; first 100 of its 200 walks) split into
  RETRACE-OK   revisits the retrace rule answers correctly;
  NEEDS-MAP    revisits the retrace rule gets wrong (the place was reached by a route other than undoing the last run).
Path models are read with the x1.111 attention-dropout re-score (registered for TW_NORMSTEP / TW_STATECHANGE) and in
eval mode; RoPE models in eval mode (the re-score moves them by <= 0.001, DROPOUT_RESCORE.md).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/language")
from tw_lib import TextWorld, gen, load, ck_tw, delta_of, forward, SCALE, TW  # noqa
from mapformer.train_variant import VARIANT_MAP  # noqa

torch.set_num_threads(4)


def retrace_ok(g):
    nothing = None
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
            pr = NOTHING
            if prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0:
                pr = o[t - 2 * cur]
            out[int(slots[t])] = bool(pr == o[t])
    return out


def load_rope(L, s, V):
    b = torch.load(f"{TW}/RoPE_L{L}_s{s}/RoPE.pt", map_location="cpu", weights_only=False)
    m = VARIANT_MAP["RoPE"](vocab_size=V, d_model=128, n_heads=2, n_layers=L, grid_size=64)
    m.load_state_dict(b["model_state_dict"]); return m.eval()


te = TextWorld(size=64, seed=10000); np.random.seed(0)
NOTHING = te.idx["nothing"]
G = [gen(te, 1024) for _ in range(100)]
RT = [retrace_ok(g) for g in G]
nr = sum(len(x) for x in RT); nok = sum(sum(x.values()) for x in RT)
print(f"stream: {nr} revisit targets; retrace correct on {nok / nr:.4f} (RETRACE-OK n {nok}, NEEDS-MAP n {nr - nok})")


def score(pred_fn):
    a_ok, a_map = [], []
    for g, rt in zip(G, RT):
        pred = pred_fn(g).argmax(-1).numpy()
        for i, ok in rt.items():
            (a_ok if ok else a_map).append(pred[i - 1] == g["tok"][i])
    return float(np.mean(a_ok + a_map)), float(np.mean(a_ok)), float(np.mean(a_map))


V = len(te.vocab)
rows = {}
for s in range(8):
    m, _ = load(ck_tw(s))
    r1 = score(lambda g: forward(m, g["tok"], delta_of(m, g["tok"]), scale=SCALE))
    r0 = score(lambda g: forward(m, g["tok"], delta_of(m, g["tok"]), scale=1.0))
    rows.setdefault("path", []).append(r1); rows.setdefault("path_x1", []).append(r0)
    print(f"path s{s}: re-scored all {r1[0]:.4f} RETRACE-OK {r1[1]:.4f} NEEDS-MAP {r1[2]:.4f} | eval mode {r0[0]:.4f} / {r0[1]:.4f} / {r0[2]:.4f}", flush=True)
    for L in (1, 2):
        mr = load_rope(L, s, V)
        with torch.no_grad():
            rr = score(lambda g: mr(torch.as_tensor(g["tok"])[None])[0])
        rows.setdefault(f"rope{L}", []).append(rr)
        print(f"  RoPE {L}L s{s}: all {rr[0]:.4f} RETRACE-OK {rr[1]:.4f} NEEDS-MAP {rr[2]:.4f}", flush=True)
for k, v in rows.items():
    v = np.array(v)
    print(f"MEAN {k:8s}: all {v[:, 0].mean():.4f} | RETRACE-OK {v[:, 1].mean():.4f} | NEEDS-MAP {v[:, 2].mean():.4f} "
          f"(min {v[:, 2].min():.4f}, max {v[:, 2].max():.4f}); fresh seeds 2-7 NEEDS-MAP {v[2:, 2].mean():.4f}")
