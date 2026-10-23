"""Task gate for G-3 (rule 11; G3_SSM_PREREG.md Sec. 3). CPU, calls the task code (GridWorldSteps, train_cancel's
evaluation stream). For each ring size m in {8, 32} at T = 128 (registered) and 512 (descriptive):

  vocabulary   every token id < unified_vocab_size; action ids are the model's act buffer
  chance       majority label among scored revisit targets (blank, p_empty 0.5)
  action-only  n-gram on the preceding ACTION tokens only, orders 1-5 (validate_nd.g2_action_ngram's rule: fit on 2000
               training walks, np seed 4242; scored on the registered evaluation stream)
  full-stream  n-gram on the preceding k TOKENS (actions and observations), orders 1-5, same split
  retrace      docs/audits/2026-10-16/ring_floors.py's one-level retrace rule (copy the observation 2j tokens back while
               the current run reverses the previous one; else blank) -- the best map-free rule on record for this walk
  revisits     share of steps that are revisits, mean per walk

The integration check (a model without path integration fails) is read from the pilot's trained `index` (every token
steps 1: an input-independent rotary SSM) and `none` (memoryless) arms when their eval.json exists.

  cd /home/prashr && python3 -m mapformer.gate_ssm_sign [--pilot-root runs/ssm_sign_pilot] [--json OUT]
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_steps import GridWorldSteps

REPO = "/home/prashr/mapformer"
EVAL_SEED, EVAL_ENV_SEED, N_TRIALS = 10 ** 6, 10000, 400


def walks(env, T, n, seed):
    np.random.seed(seed)
    out = []
    for _ in range(n):
        tok, _om, rev = env.generate_trajectory(T)
        out.append((tok.numpy(), rev.numpy()))
    return out


def scored(tok, rev):
    """(input position, target) pairs exactly as train_cancel.evaluate scores them: logits at i predict token i+1,
    scored where rev[i+1]."""
    return [(i, int(tok[i + 1])) for i in range(len(tok) - 1) if rev[i + 1]]


def ngram(train, test, order, actions_only, n_act):
    def ctx(tok, i):
        if actions_only:                       # the last `order` ACTION tokens at or before input position i
            a = [int(t) for t in tok[:i + 1] if t < n_act]
            return tuple(a[-order:])
        return tuple(int(t) for t in tok[max(0, i - order + 1):i + 1])
    table = defaultdict(Counter)
    for tok, rev in train:
        for i, y in scored(tok, rev):
            table[ctx(tok, i)][y] += 1
    glob = Counter()
    for c in table.values():
        glob.update(c)
    fallback = glob.most_common(1)[0][0]
    hit = tot = 0
    for tok, rev in test:
        for i, y in scored(tok, rev):
            c = table.get(ctx(tok, i))
            pred = c.most_common(1)[0][0] if c else fallback
            hit += pred == y
            tot += 1
    return hit / tot


def retrace(test, n_act, blank):
    hit = tot = 0
    for tok, rev in test:
        a = tok[0::2]
        o = tok[1::2]
        r = rev[1::2]
        prev_run = cur = 0
        for t in range(len(a)):
            if t > 0 and a[t] == a[t - 1]:
                cur += 1
            else:
                prev_run = cur if t > 0 else 0
                cur = 1
            if r[t]:
                tot += 1
                pr = o[t - 2 * cur] if (prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0) else blank
                hit += pr == o[t]
    return hit / tot


def gate(m, T, n_train=2000):
    env = GridWorldSteps(size=m, seed=EVAL_ENV_SEED, p_first=0.5, deltas=(1, -1))
    V, n_act = env.unified_vocab_size, 2
    blank = env.blank_token + env.obs_offset
    train = walks(env, T, n_train, 4242)
    test = walks(env, T, N_TRIALS, EVAL_SEED)       # the registered evaluation stream (train_cancel.evaluate)
    ids_ok = all(int(t.max()) < V and int(t.min()) >= 0 for t, _ in train + test)
    labels = [y for tok, rev in test for _, y in scored(tok, rev)]
    c = Counter(labels)
    rev_share = float(np.mean([r[1::2].mean() for _, r in test]))
    out = {"m": m, "T": T, "vocab": V, "token_ids_in_vocab": ids_ok, "n_scored": len(labels),
           "revisit_share": rev_share, "chance_majority": max(c.values()) / len(labels),
           "majority_is_blank": c.most_common(1)[0][0] == blank,
           "action_ngram": {k: ngram(train, test, k, True, n_act) for k in range(1, 6)},
           "stream_ngram": {k: ngram(train, test, k, False, n_act) for k in range(1, 6)},
           "retrace": retrace(test, n_act, blank)}
    out["best_floor"] = max([out["chance_majority"], out["retrace"], *out["action_ngram"].values(),
                             *out["stream_ngram"].values()])
    return out


def pilot_baselines(root):
    from mapformer.stats_core import classify_run
    rows = {}
    if not os.path.isdir(root):
        return rows
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "eval.json")
        if not os.path.exists(p):
            continue
        e = json.load(open(p))
        if e["arm"] in ("index", "none"):
            k = classify_run(e["losses"])
            rows[d] = {"arm": e["arm"], "m": e["size"], "acc128": e["eval"]["128"]["acc"],
                       "acc512": e["eval"]["512"]["acc"], "class": k["cls"], "final_loss": e["final_loss"],
                       "epochs": len(e["losses"])}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", default=f"{REPO}/runs/ssm_sign_pilot")
    ap.add_argument("--json", default=None)
    ap.add_argument("--n-train", type=int, default=2000)
    a = ap.parse_args()
    res = {"floors": [], "pilot_baselines": pilot_baselines(a.pilot_root)}
    for m in (8, 32):
        for T in (128, 512):
            g = gate(m, T, a.n_train if T == 128 else a.n_train // 4)
            res["floors"].append(g)
            an = " ".join(f"{v:.3f}" for v in g["action_ngram"].values())
            sn = " ".join(f"{v:.3f}" for v in g["stream_ngram"].values())
            print(f"m={m:2d} T={T}: ids ok {g['token_ids_in_vocab']} vocab {g['vocab']} | scored {g['n_scored']} "
                  f"revisit share {g['revisit_share']:.3f} | chance {g['chance_majority']:.4f} (blank "
                  f"{g['majority_is_blank']}) | action n-gram 1-5 {an} | stream n-gram 1-5 {sn} | retrace "
                  f"{g['retrace']:.4f} | BEST FLOOR {g['best_floor']:.4f}", flush=True)
    for d, r in res["pilot_baselines"].items():
        print(f"pilot {d}: {r['arm']} m={r['m']} acc T=128 {r['acc128']:.4f} T=512 {r['acc512']:.4f} class "
              f"{r['class']} final loss {r['final_loss']:.4f} ({r['epochs']} epochs)")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
