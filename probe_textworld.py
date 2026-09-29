"""Which words move the phase? Per-word step probe for TEXTWORLD_PREREG.md.

MapFormer's step is a function of the token alone, Delta_w = W_out W_in emb(w), so the whole map from
words to steps is a table read from the weights (no data needed). All heads' angle increments are
concatenated; omega is not applied (as in probe_action_geometry --space delta). Readouts, all
scale-free:
  move ratio   mean ||Delta|| over NON-direction words / mean over the 12 direction words
               (0 = only direction words move the phase; the torus analogue is obs norm / action norm)
  opposition   ||Delta(a) + Delta(b)|| / mean(||Delta(a)||, ||Delta(b)||) over all synonym pairs of
               north x south and west x east (0 = opposite directions cancel, 2 = identical)
  synonym cos  mean pairwise cosine within each direction's three synonyms (1 = the same step)
  synonym norm min/max norm ratio within each synonym set (1 = the same size)
  |cos(N,E)|   between the synonym-averaged north and east steps (0 = orthogonal axes)
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from mapformer.environment_textworld import TextWorld, DIRS
from mapformer.train_variant import VARIANT_MAP


@torch.no_grad()
def table(ck):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    a = blob["config"]["args"]
    env = TextWorld(size=a["size"], seed=0)
    m = VARIANT_MAP[a["variant"]](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2,
                                  n_layers=a["n_layers"], grid_size=a["size"])
    m.load_state_dict(blob["model_state_dict"]); m.eval()
    d = m.action_to_lie(m.token_emb.weight[None])[0]          # (V, H, n_blocks)
    return env, d.reshape(d.shape[0], -1).numpy()


def readouts(env, D):
    n = np.linalg.norm(D, axis=1)
    dir_ids = [i for a in range(4) for i in env.dir_ids[a]]
    other = [i for i in range(len(env.vocab)) if i not in dir_ids]
    move = n[other].mean() / n[dir_ids].mean()
    opp = []
    for a, b in ((0, 1), (2, 3)):
        for i in env.dir_ids[a]:
            for j in env.dir_ids[b]:
                opp.append(np.linalg.norm(D[i] + D[j]) / ((n[i] + n[j]) / 2))
    cos, nr = [], []
    for a in range(4):
        ids = env.dir_ids[a]
        for x in range(3):
            for y in range(x + 1, 3):
                cos.append(D[ids[x]] @ D[ids[y]] / (n[ids[x]] * n[ids[y]]))
        nr.append(n[ids].min() / n[ids].max())
    avg = {a: D[env.dir_ids[a]].mean(0) for a in range(4)}
    cne = abs(avg[0] @ avg[3]) / (np.linalg.norm(avg[0]) * np.linalg.norm(avg[3]))
    top = [env.vocab[i] for i in np.argsort(-n)[:12]]
    return {"move_ratio": float(move), "opposition": float(np.mean(opp)), "synonym_cos": float(np.mean(cos)),
            "synonym_norm": float(np.mean(nr)), "cos_NE": float(cne), "top_words": top,
            "norm_by_word": {env.vocab[i]: float(n[i]) for i in range(len(env.vocab))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="holds <arm>_L<n>_s<seed>/Vanilla_r4.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for ck in sorted(glob.glob(os.path.join(a.runs_dir, "Vanilla_r4_L*_s*", "Vanilla_r4.pt"))):
        env, D = table(ck)
        r = readouts(env, D); res[os.path.basename(os.path.dirname(ck))] = r
        print(f"{os.path.basename(os.path.dirname(ck)):20s} move {r['move_ratio']:.3f}  opp {r['opposition']:.3f}  "
              f"syn cos {r['synonym_cos']:.3f}  syn norm {r['synonym_norm']:.3f}  |cos NE| {r['cos_NE']:.3f}  "
              f"top: {' '.join(r['top_words'][:8])}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
