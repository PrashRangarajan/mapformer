"""Task gate, floors and perception ceiling for the image-observation torus (IMAGE_TORUS_PREREG.md; rules 1, 11).
CPU only, no trained model. Writes IMAGE_TORUS_GATE.json / .md (md5-guarded by the driver; the analysis reads them).

1. Stream checks: train_image's environment is GridWorld(64, 16, 0.5, seed) -- the token task's -- so the class-label
   stream and revisit mask are the token task's by construction; checked here by CALLING train_image.make_env and
   check_interleaving (rule 11) against a fresh GridWorld for each batch seed; token ids inside the 21-token vocabulary.
   The evaluation walks are eval_image_torus.walks itself (Amendment 1, m5).
2. Floors on the evaluation walks of each batch seed (held-out map, env seed 10000, np seed 1234 + s, 100 walks,
   T = 1024; eval_noise_refine's stream): always-blank; retrace (analyze_factorial_n8.floors_walk, the ported
   score_rank_floors rule); best backoff token n-gram, orders 0-5, and action-stream-only n-gram, orders 1-5 (the n
   previous ACTION tokens), both fitted on revisit targets of 50 walks on each batch seed's TRAINING map (np seed
   70000 + s); per stratum (short-gap, long-gap, wrap-only) for blank and retrace.
3. Perception: a standalone copy of the image encoder (model_image.ImageEncoder + a linear 16-way head, seed 0, Adam
   1e-3, batch 256, 10 epochs on the train split) -- the recognition this encoder size reaches alone, per class, on
   the test split. Ceilings on the same evaluation walks for an agent with a PERFECT map whose only error is
   recognition by this classifier, every visit a fresh test-split instance (torch generator seed 4242 + s):
     ceil_last  -- names the class it recognised at the most recent earlier visit;
     ceil_vote  -- names the argmax of the summed log-posteriors over all earlier visits (Bayes, uniform prior);
   blank targets count as hits (the blank image is constant). ceil_vote is the reference ceiling.
4. Init scale (Amendment 1, M2): observation-embedding norms and the step map's object / action and within-class /
   action ratios at initialisation, image MapWM (with the fixed encoder gain) vs the token twin, per batch seed.

  cd /home/prashr && OMP_NUM_THREADS=4 python3 -m mapformer.gate_image_torus [--seeds 3100 ...] [--epochs 10]
"""
import argparse
import collections
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.analyze_factorial_n8 import floors_walk
from mapformer import model_image as MI
from mapformer import image_bank

REPO = "/home/prashr/mapformer"
SEEDS = list(range(3100, 3108))
STRATA = ("plain_lag<128", "plain_lag>=128", "wrap")


def eval_walks(s, T=1024, n=100, env_seed=10000):
    """The evaluator's own walk function (rule 11: called, not copied)."""
    from mapformer.eval_image_torus import walks
    return [tuple(x.numpy() for x in w) for w in walks(env_seed, 1234 + s, n, T)]


def stream_check(seeds):
    """train_image's env vs GridWorld(seed); the batch it would emit vs a fresh GridWorld's under the same np seed."""
    from mapformer.train_image import make_env, check_interleaving      # the trainer's construction (rule 11)
    out = {}
    for s in seeds:
        a = make_env(s); check_interleaving(a, 1024)
        b = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=s)
        np.random.seed(s); ta, oa, ra, _ = a.generate_batch(4, 256)
        np.random.seed(s); tb, ob, rb, _ = b.generate_batch(4, 256)
        out[s] = {"map_equal": bool(torch.equal(a.obs_map, b.obs_map)),
                  "batch_equal": bool(torch.equal(ta, tb) and torch.equal(ra, rb)),
                  "max_token": int(ta.max()), "vocab": a.unified_vocab_size,
                  "occupied_share": float((a.obs_map != 16).float().mean())}
        assert out[s]["map_equal"] and out[s]["batch_equal"] and out[s]["max_token"] < MI.VOCAB, out[s]
    return out


def fit_ngrams(s):
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=s)
    np.random.seed(70000 + s)
    tok_c = [collections.defaultdict(collections.Counter) for _ in range(6)]
    act_c = [collections.defaultdict(collections.Counter) for _ in range(6)]
    for _ in range(50):
        tok, rev = (x.numpy() for x in env.generate_trajectory(1024)[::2])
        acts = tok[0::2]
        for i in np.nonzero(rev)[0]:
            t = i // 2
            for n in range(6):
                tok_c[n][tuple(tok[i - n:i])][tok[i]] += 1
                act_c[n][tuple(acts[max(0, t + 1 - n):t + 1]) if n else ()][tok[i]] += 1
    best = lambda C: [{k: c.most_common(1)[0][0] for k, c in C[n].items()} for n in range(6)]
    return best(tok_c), best(act_c)


def ngram_hits(tok, rev, tb, ab):
    acts = tok[0::2]; h_tok = np.zeros(6); h_act = np.zeros(6); n = 0
    for i in np.nonzero(rev)[0]:
        t = i // 2; n += 1
        for k in range(6):
            p = None
            for m in range(k, -1, -1):
                p = tb[m].get(tuple(tok[i - m:i]))
                if p is not None:
                    break
            h_tok[k] += p == tok[i]
            p = None
            for m in range(k, -1, -1):
                p = ab[m].get(tuple(acts[max(0, t + 1 - m):t + 1]) if m else ())
                if p is not None:
                    break
            h_act[k] += p == tok[i]
    return h_tok, h_act, n


def floors(s, W):
    tb, ab = fit_ngrams(s)
    c = collections.Counter(); ht = np.zeros(6); ha = np.zeros(6); nn_ = 0
    for tok, rev in W:
        for b, rt, st in floors_walk(tok, rev, 64):
            for k in ("all", st):
                c[f"n|{k}"] += 1; c[f"blank|{k}"] += int(b); c[f"retrace|{k}"] += int(rt)
        a, b2, n = ngram_hits(tok, rev, tb, ab); ht += a; ha += b2; nn_ += n
    f = {k: {"n": c[f"n|{k}"], "blank": c[f"blank|{k}"] / max(1, c[f"n|{k}"]),
             "retrace": c[f"retrace|{k}"] / max(1, c[f"n|{k}"])} for k in ("all",) + STRATA}
    f["ngram_token_by_order"] = (ht / nn_).tolist(); f["ngram_token_best"] = float((ht / nn_).max())
    f["ngram_action_by_order"] = (ha / nn_).tolist(); f["ngram_action_best"] = float((ha / nn_)[1:].max())
    f["best_trivial"] = max(f["all"]["blank"], f["all"]["retrace"], f["ngram_token_best"], f["ngram_action_best"])
    return f


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = MI.ImageEncoder(128); self.head = nn.Linear(128, MI.N_CLS)

    def forward(self, x):
        return self.head(self.enc(x))


def train_classifier(banks, epochs, seed=0, bs=256):
    torch.manual_seed(seed)
    clf = Classifier(); opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    X = banks["train"]["images"]; y = banks["train"]["labels"]; g = torch.Generator().manual_seed(seed)
    for e in range(epochs):
        perm = torch.randperm(len(X), generator=g); t0 = time.time(); tot = 0.0
        for i in range(0, len(X), bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(clf(X[j].float()[:, None] / 255.0), y[j])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(j)
        print(f"  classifier epoch {e + 1}/{epochs} loss {tot / len(X):.4f} ({time.time() - t0:.0f} s)", flush=True)
    return clf.eval()


@torch.no_grad()
def logpost(clf, imgs, bs=4096):
    return torch.cat([F.log_softmax(clf(imgs[i:i + bs].float()[:, None] / 255.0), -1)
                      for i in range(0, len(imgs), bs)])


def ceilings(s, W, LP, test):
    """Perfect map, recognition by the standalone classifier; a fresh test instance at every visit."""
    g = torch.Generator().manual_seed(4242 + s)
    off, cnt = test["off"], test["cnt"]
    c = collections.Counter()
    for tok, rev in W:
        a = tok[0::2]; o = tok[1::2]; r = rev[1::2]
        pos = np.cumsum(MI_DELTA[a], 0) % 64
        cls = np.clip(o - MI.N_ACT, 0, MI.N_CLS - 1)
        u = torch.rand(len(o), generator=g)
        idx = off[cls] + torch.minimum((u * cnt[cls]).long(), cnt[cls] - 1)
        lp = LP[idx]                                            # (T, 16) posterior of the instance seen at step t
        last, summ = {}, {}
        for t in range(len(o)):
            key = (int(pos[t, 0]), int(pos[t, 1]))
            obj = o[t] != MI.BLANK
            if r[t]:
                c["n"] += 1
                if not obj:
                    c["last"] += 1; c["vote"] += 1; c["blank"] += 1
                else:
                    c["n_obj"] += 1
                    k = int(cls[t])
                    hl = int(last[key].argmax()) == k; hv = int(summ[key].argmax()) == k
                    c["last"] += hl; c["vote"] += hv; c["obj_last"] += hl; c["obj_vote"] += hv
            if obj:
                last[key] = lp[t]; summ[key] = summ.get(key, 0) + lp[t]
    return {"ceil_last": c["last"] / c["n"], "ceil_vote": c["vote"] / c["n"],
            "obj_last": c["obj_last"] / max(1, c["n_obj"]), "obj_vote": c["obj_vote"] / max(1, c["n_obj"]),
            "n": c["n"], "n_obj": c["n_obj"]}


MI_DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def init_scale(seeds, banks):
    """Amendment 1 (M2): at initialisation, observation-embedding norms and step statistics of image MapWM vs its
    token twin (eval_image_torus.step_stats on 10 evaluation walks), per batch seed."""
    from mapformer.eval_image_torus import walks, step_stats
    out = {}
    for s in seeds:
        W = walks(10000, 1234 + s, 10, 1024)
        tok = torch.stack([w[0] for w in W])[:, :-1]
        row = {}
        for obs in ("token", "image"):
            m = MI.build("MapWM", obs, s, banks=banks if obs == "image" else None).eval()
            if obs == "image":
                MI.image_emb(m).reseed(1)
            with torch.no_grad():
                x = m.token_emb(tok)
            st = step_stats(m, W, s, "cpu")
            row[obs] = {"emb_norm_action": float(x[tok < 4].norm(dim=1).mean()),
                        "emb_norm_object": float(x[(tok >= 4) & (tok < 20)].norm(dim=1).mean()),
                        "emb_norm_blank": float(x[tok == 20].norm(dim=1).mean()), **st}
        out[s] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default=f"{REPO}/IMAGE_TORUS_GATE")
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    banks = image_bank.load_banks()
    G = {"seeds": a.seeds, "bank_md5": image_bank.bank_md5(), "stream": stream_check(a.seeds), "floors": {},
         "ceilings": {}}
    W = {}
    for s in a.seeds:
        W[s] = eval_walks(s)
        G["floors"][s] = floors(s, W[s])
        f = G["floors"][s]
        print(f"s{s}: blank {f['all']['blank']:.4f} retrace {f['all']['retrace']:.4f} n-gram {f['ngram_token_best']:.4f} "
              f"action n-gram {f['ngram_action_best']:.4f}", flush=True)
    clf = train_classifier(banks, a.epochs)
    LP = logpost(clf, banks["test"]["images"])
    pred = LP.argmax(1); y = banks["test"]["labels"]
    G["classifier"] = {"epochs": a.epochs, "test_acc": float((pred == y).float().mean()),
                       "train_acc": float((logpost(clf, banks["train"]["images"]).argmax(1)
                                           == banks["train"]["labels"]).float().mean()),
                       "per_class_test": [float((pred[y == k] == k).float().mean()) for k in range(MI.N_CLS)],
                       "params_encoder": sum(p.numel() for p in clf.enc.parameters())}
    for s in a.seeds:
        G["ceilings"][s] = ceilings(s, W[s], LP, banks["test"])
        print(f"s{s}: ceil_last {G['ceilings'][s]['ceil_last']:.4f} ceil_vote {G['ceilings'][s]['ceil_vote']:.4f}",
              flush=True)
    G["enc_gain"] = MI.ENC_GAIN
    G["init_scale"] = {str(k): v for k, v in init_scale(a.seeds, banks).items()}
    G["floors"] = {str(k): v for k, v in G["floors"].items()}
    G["ceilings"] = {str(k): v for k, v in G["ceilings"].items()}
    G["stream"] = {str(k): v for k, v in G["stream"].items()}
    json.dump(G, open(a.out + ".json", "w"), indent=1)
    md = ["# Image torus: gate, floors, perception ceiling (gate_image_torus.py; no trained model)", "",
          f"Bank md5 `{G['bank_md5']}`. Stream check: train_image's environment equals GridWorld(64, 16, 0.5, seed) and "
          "emits the same batch under the same np seed for every seed below; max token < 21.", "",
          "| seed | blank | retrace | n-gram (token, best of 0-5) | n-gram (actions only, best of 1-5) | best trivial | "
          "retrace short / long / wrap | ceil_last | ceil_vote |", "|---" * 10 + "|"]
    for s in a.seeds:
        f, c = G["floors"][str(s)], G["ceilings"][str(s)]
        md.append(f"| {s} | {f['all']['blank']:.4f} | {f['all']['retrace']:.4f} | {f['ngram_token_best']:.4f} | "
                  f"{f['ngram_action_best']:.4f} | {f['best_trivial']:.4f} | {f['plain_lag<128']['retrace']:.3f} / "
                  f"{f['plain_lag>=128']['retrace']:.3f} / {f['wrap']['retrace']:.3f} | {c['ceil_last']:.4f} | "
                  f"{c['ceil_vote']:.4f} |")
    C = G["classifier"]
    md += ["", f"Standalone classifier (encoder {C['params_encoder']:,} params + linear head, {C['epochs']} epochs): "
           f"test {C['test_acc']:.4f}, train {C['train_acc']:.4f}. Per class (test): "
           + ", ".join(f"{image_bank.CLASS_NAMES[k]} {v:.3f}" for k, v in enumerate(C["per_class_test"])), ""]
    md += ["## Init scale (Amendment 1, M2): image MapWM (fixed encoder gain "
           f"{MI.ENC_GAIN}) vs token twin, at initialisation, 10 evaluation walks", "",
           "| seed | obs | emb norm action / object / blank | rms step object / action | within-class (instance) / action |",
           "|---|---|---|---|---|"]
    for s in a.seeds:
        for obs in ("token", "image"):
            r = G["init_scale"][str(s)][obs]
            md.append(f"| {s} | {obs} | {r['emb_norm_action']:.2f} / {r['emb_norm_object']:.2f} / {r['emb_norm_blank']:.2f} | "
                      f"{r['object_over_action']:.3f} | {r['instance_over_action']:.3f} |")
    for obs in ("token", "image"):
        v = [G["init_scale"][str(s)][obs]["object_over_action"] for s in a.seeds]
        u = [G["init_scale"][str(s)][obs]["instance_over_action"] for s in a.seeds]
        md.append(f"\n{obs}: object/action step ratio mean {np.mean(v):.3f} (range {min(v):.3f}-{max(v):.3f}); "
                  f"instance/action mean {np.mean(u):.3f}")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
