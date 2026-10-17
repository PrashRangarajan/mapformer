"""Per-stratum map-free floors on the torus eval stream (post hoc, CPU, no model).
Stream = eval_rank_strata / eval_noise_refine exactly: held-out map GridWorld(64, 16 obs, p_empty 0.5, seed 10000),
np.random.seed(1234 + s) for model seed s, 100 walks of T = 1024, scored on revisit targets. Strata as eval_rank_strata
(kinds_vec): wrap = the unwrapped position is new; plain_lag<128 / plain_lag>=128 by the gap since the last visit to
the wrapped cell. Per seed and stratum:
  const    best constant prediction within the stratum (= the JSON 'floor'; asserted equal to every *_STRATA.json entry
           of that seed that is given with --check)
  blank    always-blank
  retrace  the one-level retrace rule of docs/audits/2026-10-16/sign_floors.py (while a run of actions reverses the
           previous run, step j copies the observation 2j tokens back; else blank)
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/torus_strata_floors.py \
      --seeds 0 1 2 3 4 5 6 7 --out torus_strata_floors_s0-7.json --check SIGN_MATCHED_STRATA.json"""
import argparse, collections, json, os
import numpy as np

from mapformer.environment import GridWorld
from mapformer.eval_rank_strata import kinds_vec

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
BLANK = 20; ST = ("all", "plain_lag<128", "plain_lag>=128", "wrap")


def per_walk(tok, rev, size):
    a = tok[0::2].numpy(); o = tok[1::2].numpy(); r = rev[1::2].numpy()
    new_u, lag = kinds_vec(tok, size)
    c = {k: collections.Counter() for k in ST}; tg = {k: [] for k in ST}
    prev_run = cur = 0
    for t in range(len(a)):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
            cur = 1
        if not r[t]:
            continue
        pr = o[t - 2 * cur] if (prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0) else BLANK
        k = "wrap" if new_u[t] else ("plain_lag<128" if lag[t] < 128 else "plain_lag>=128")
        for kk in ("all", k):
            c[kk]["n"] += 1; c[kk]["blank"] += o[t] == BLANK; c[kk]["retrace"] += pr == o[t]; tg[kk].append(o[t])
    return c, tg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--check", nargs="*", default=[])
    a = ap.parse_args()
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000); res = {}
    checks = [json.load(open(f"{REPO}/{f}")) for f in a.check]
    for s in a.seeds:
        np.random.seed(1234 + s); C = {k: collections.Counter() for k in ST}; TG = {k: [] for k in ST}
        for _ in range(100):
            tok, _o, rev = env.generate_trajectory(1024)
            c, tg = per_walk(tok, rev, 64)
            for k in ST:
                C[k].update(c[k]); TG[k] += tg[k]
        res[s] = {}
        for k in ST:
            n = C[k]["n"]; const = np.bincount(np.array(TG[k])).max() / n
            res[s][k] = {"n": n, "const": float(const), "blank": C[k]["blank"] / n, "retrace": C[k]["retrace"] / n}
        for d in checks:
            for key, v in d.items():
                if key.split("|")[1] == str(s) and key.split("|")[2] == "1024":
                    for k in ST:
                        assert v[k]["n"] == res[s][k]["n"] and abs(v[k]["floor"] - res[s][k]["const"]) < 1e-12, (key, k)
        print(s, {k: (res[s][k]["n"], round(res[s][k]["const"], 4), round(res[s][k]["retrace"], 4)) for k in ST}, flush=True)
    json.dump({"seeds": a.seeds, "per_seed": res, "checked_against": a.check}, open(f"{HERE}/{a.out}", "w"), indent=1)


if __name__ == "__main__":
    main()
