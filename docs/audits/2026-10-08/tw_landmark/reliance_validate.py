"""TW_LANDMARK: validate the theta-reliance readout (tw_landmark_eval.DirMean, mode 'dir') on committed text-world
path models and on untrained models, on the batch's own eval stream (held-out map 10000, np seed 10**6, 200 walks,
the common-move targets of the rate-0 condition = names stripped). CPU only. Committed models have the 58-word
vocabulary; the rate-0 rendering uses only those ids (TextWorldLandmark at rate 0 = TextWorld byte for byte).
Output: reliance_validate_out.txt (+ .json: per-seed acc / reliance, input to tw_landmark_power.py)."""
import json
import sys

import numpy as np
import torch

from mapformer.analyze_textworld_secondary import load as load_tw
from mapformer.tw_normstep_readouts import load as load_ns
from mapformer.tw_landmark_eval import build_eval, DirMean, predict, acc_of
from mapformer.environment_tw_landmark import TextWorldLandmark
from mapformer.train_tw_landmark import build

torch.set_num_threads(16)
OUT = "/home/prashr/mapformer/docs/audits/2026-10-08/tw_landmark/reliance_validate_out"
E = build_eval(0.0, 200)
toks, tg = E["strip"]
assert int(toks.max()) < 58
env = TextWorldLandmark(seed=10000)
dirs = [i for a in range(4) for i in env.dir_ids[a]]
print(f"eval: {len(toks)} walks, {len(tg)} common-move revisit targets (names stripped)")


def rel(m, names=()):
    h = DirMean(m, dirs, list(names) or [dirs[0]])
    p = predict(m, toks, "cpu", hook=h); a = acc_of(p, tg)[0]
    h.mode = "dir"; q = predict(m, toks, "cpu", hook=h); b = acc_of(q, tg)[0]; h.mode = None
    # identity check: every direction word replaced by ITS OWN step must leave the logits unchanged (up to float
    # rounding: the step is recomputed outside the sequence; argmax can flip on exact near-ties, so logits are compared)
    with torch.no_grad():
        h.groups["self"] = [(torch.tensor([i]), m.action_to_lie(m.token_emb(torch.tensor([[i]])))[0, 0]) for i in dirs]
        x = toks[:4, :-1]; h.tok = x; l0 = m(x); h.mode = "self"; l1 = m(x); h.mode = None
    h.h.remove()
    same = float((l0 - l1).abs().max())
    return a, a - b, same


res = {"tw_path1L": {}, "ns_MapWM": {}, "untrained": {}}
for s in range(8):
    m = load_tw(f"/home/prashr/mapformer/runs/textworld/p0/Vanilla_r4_L1_s{s}/Vanilla_r4.pt", "cpu")
    a, r, same = rel(m); res["tw_path1L"][s] = {"acc": a, "rel": r}
    print(f"text world path 1L s{s}: acc {a:.4f}  reliance {r:+.4f}  (identity hook max |logit diff| {same:.1e})", flush=True)
for s in range(10, 18):
    m, _ = load_ns(f"/home/prashr/mapformer/runs/tw_normstep/p0/MapWM_s{s}/MapWM.pt")
    a, r, same = rel(m); res["ns_MapWM"][s] = {"acc": a, "rel": r}
    print(f"TW_NORMSTEP MapWM s{s}: acc {a:.4f}  reliance {r:+.4f}  (identity hook max |logit diff| {same:.1e})", flush=True)
for L in (1, 2):
    for s in (150, 151):
        torch.manual_seed(s); m = build("MapWM", TextWorldLandmark(seed=s), L).eval()
        a, r, same = rel(m); res["untrained"][f"MapWM_L{L}_s{s}"] = {"acc": a, "rel": r}
        print(f"untrained MapWM {L}L s{s}: acc {a:.4f}  reliance {r:+.4f}  (identity hook max |logit diff| {same:.1e})", flush=True)
for k, v in res.items():
    if v:
        x = np.array([d["rel"] for d in v.values()])
        print(f"{k}: reliance min {x.min():+.4f} median {np.median(x):+.4f} max {x.max():+.4f}")
json.dump(res, open(OUT + ".json", "w"), indent=1)
