"""POST HOC, CPU only (Opus scope expansion 2026-10-29). Which map-free mechanism could serve each torus revisit, and which
revisits do trained index models get right?

Per revisit target o_t (T = 1024 torus, held-out map env seed 10000, walks np.random.seed(1234 + s): eval_rank_strata's
stream), labels:
  stratum   short (gap < 128) / long / wrap-only (eval_rank_strata.kinds_vec)
  retrace   sign_floors.py's rule applies AND is correct
  ind_m     in-context suffix match ("induction" predictor): the longest token window tok[i-m:i] (m = 2..MAXM tokens, ending
            with the action a_t) that occurred earlier in the same walk; predict the token that followed its most recent
            earlier occurrence. ind_ok = that prediction is correct; ind_same = the matched occurrence was at the same cell
            (a genuine transition recall rather than a coincidence); m recorded.
  comb      a realisable map-free predictor: retrace if its rule applies, else induction if a match of >= MMIN tokens
            exists, else blank.
Models (eval mode, CPU): FACTORIAL (s500-503) and FACTORIAL_N8 (s2800-2803) checkpoints: RoPE L1, RoPE L4, RoPE loop x4,
path MapWM r4 L1. Output: accuracy of each arm in cells (stratum x retrace-applies x induction-correct), and the floors.
  CUDA_VISIBLE_DEVICES= python3 docs/research/scripts_opus/q1_revisit_taxonomy.py [n_walks]
"""
import collections, json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
from mapformer.environment import GridWorld  # noqa: E402
from mapformer.eval_rank_strata import kinds_vec  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402

REPO = "/home/prashr/mapformer"
OUT = f"{REPO}/docs/research/scripts_opus/out"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)
N_WALKS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
T = 1024; BLANK = 20; MAXM = 24; MMIN = 4
DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
ARMS = {"RoPE_L1": ("L1", "RoPE"), "RoPE_L4": ("L4", "RoPE"), "RoPE_Loop": ("LOOP", "RoPELooped"),
        "Path_L1": ("L1", "Vanilla_r4")}
BATCHES = {"factorial": (500, 501, 502, 503), "factorial_n8": (2800, 2801, 2802, 2803)}


def labels(tok):
    tok = tok.numpy(); a = tok[0::2]; o = tok[1::2]; n = len(a)
    new_u, lag = kinds_vec(torch.from_numpy(tok), 64)
    U = np.cumsum(DELTA[a], 0) % 64
    cellid = U[:, 0] * 64 + U[:, 1]
    rows = []; prev_run = cur = 0
    dicts = [dict() for _ in range(MAXM + 1)]                  # window of length m -> (last continuation, its step index)
    seen = set()
    for t in range(n):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
            cur = 1
        i = 2 * t + 1
        # induction: longest earlier match of tok[i-m:i]
        im, ipred, isame = 0, BLANK, False
        for m in range(min(MAXM, i), 1, -1):
            hit = dicts[m].get(tok[i - m:i].tobytes())
            if hit is not None:
                im, ipred = m, hit[0]; isame = cellid[hit[1]] == cellid[t]
                break
        if cellid[t] in seen:                                  # revisit (start cell excluded as in the env)
            rows.append(dict(t=t, tgt=int(o[t]), wrap=bool(new_u[t]), lag=int(lag[t]), rt_app=bool(prev_run > 0 and cur <= prev_run),
                             rt_pred_o=int(o[t - 2 * cur]) if (prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0) else BLANK,
                             im=im, ipred=int(ipred), isame=bool(isame)))
        seen.add(cellid[t])
        for m in range(2, min(MAXM, i) + 1):
            dicts[m][tok[i - m:i].tobytes()] = (int(tok[i]), t)
    return rows


def load(batch, sub, v, s):
    b = torch.load(f"{REPO}/runs/{batch}/{sub}/p0/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
    c = b["config"]
    m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                       grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); m.eval()
    return m


def main():
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    recs = []
    for batch, seeds in BATCHES.items():
        for s in seeds:
            np.random.seed(1234 + s)
            walks = [env.generate_trajectory(T) for _ in range(N_WALKS)]
            models = {k: load(batch, sub, v, s) for k, (sub, v) in ARMS.items()}
            for w, (tok, _om, rev) in enumerate(walks):
                L = labels(tok)
                with torch.no_grad():
                    preds = {k: m(tok[None, :-1]).argmax(-1)[0].numpy() for k, m in models.items()}
                revt = set(np.nonzero(rev.numpy()[1::2])[0].tolist())
                assert revt == {r["t"] for r in L}, (len(revt), len(L))
                for r in L:
                    i = 2 * r["t"] + 1
                    r.update({k: int(p[i - 1] == r["tgt"]) for k, p in preds.items()})
                    r["seed"] = s; r["walk"] = w
                    recs.append(r)
            print(f"{batch} s{s}: {len(recs)} revisits so far", flush=True)
    json.dump(recs, open(f"{OUT}/q1_revisit_rows.json", "w"))
    summarize(recs)


def summarize(recs):
    lines = []
    P = lambda *x: (print(*x), lines.append(" ".join(str(y) for y in x)))
    R = recs
    for r in R:
        r["stratum"] = "wrap" if r["wrap"] else ("short" if r["lag"] < 128 else "long")
        r["rt_ok"] = int(r["rt_app"] and r["rt_pred_o"] == r["tgt"])
        r["ind_ok"] = int(r["im"] >= 2 and r["ipred"] == r["tgt"])
        for mm in (4, 6, 8, 12):
            pr = r["rt_pred_o"] if r["rt_app"] else (r["ipred"] if r["im"] >= mm else BLANK)
            r[f"comb{mm}"] = int(pr == r["tgt"])
        r["blank"] = int(r["tgt"] == BLANK)
    arms = list(ARMS)
    P(f"n revisits {len(R)} ({N_WALKS} walks x 8 seeds, T = {T})")
    P("== floors (model-free) by stratum: share, blank, retrace(rule), induction(longest match, any m>=2), comb(m>=4/6/8/12) ==")
    for st in ("all", "short", "long", "wrap"):
        S = [r for r in R if st == "all" or r["stratum"] == st]
        rt = np.mean([r["rt_pred_o"] == r["tgt"] if r["rt_app"] else r["tgt"] == BLANK for r in S])
        P(f"{st:6s} n {len(S):6d} share {len(S)/len(R):.3f} blank {np.mean([r['blank'] for r in S]):.3f} retrace {rt:.3f} "
          f"induction {np.mean([r['ipred'] == r['tgt'] if r['im'] >= 2 else r['tgt'] == BLANK for r in S]):.3f} "
          + " ".join(f"comb{mm} {np.mean([r[f'comb{mm}'] for r in S]):.3f}" for mm in (4, 6, 8, 12))
          + f" | ind same-cell match share {np.mean([r['isame'] for r in S]):.3f}; median match len {np.median([r['im'] for r in S]):.0f}")
    P("== model accuracy by stratum ==")
    for st in ("all", "short", "long", "wrap"):
        S = [r for r in R if st == "all" or r["stratum"] == st]
        P(f"{st:6s} " + " ".join(f"{k} {np.mean([r[k] for r in S]):.3f}" for k in arms))
    P("== model accuracy in cells: stratum x (retrace applies) x (induction: same-cell match of >= m tokens) ==")
    for mm in (4, 8):
        P(f"-- induction cell defined as a same-cell match of >= {mm} tokens --")
        for st in ("short", "long", "wrap"):
            for rta in (True, False):
                for ind in (True, False):
                    S = [r for r in R if r["stratum"] == st and r["rt_app"] == rta and ((r["isame"] and r["im"] >= mm) == ind)]
                    if len(S) < 30:
                        continue
                    P(f"{st:5s} retrace {'Y' if rta else 'N'} ind {'Y' if ind else 'N'}: n {len(S):5d} blank {np.mean([r['blank'] for r in S]):.3f} "
                      f"ind_pred {np.mean([r['ipred'] == r['tgt'] for r in S]):.3f} | "
                      + " ".join(f"{k} {np.mean([r[k] for r in S]):.3f}" for k in arms))
    # per-seed for the key cell: non-retrace short-gap revisits, induction-yes vs no, RoPE_L4 and Loop
    P("== per seed, non-retrace revisits (all strata), same-cell match >= 8 tokens: Y vs N (acc) ==")
    for k in arms:
        ys, ns = [], []
        for s in sorted({r["seed"] for r in R}):
            S = [r for r in R if r["seed"] == s and not r["rt_app"]]
            Y = [r[k] for r in S if r["isame"] and r["im"] >= 8]; N = [r[k] for r in S if not (r["isame"] and r["im"] >= 8)]
            ys.append(np.mean(Y)); ns.append(np.mean(N))
        d = np.array(ys) - np.array(ns)
        P(f"{k}: Y {np.mean(ys):.3f} N {np.mean(ns):.3f} d {d.mean():+.3f} (seeds + {int((d > 0).sum())}/{len(d)})")
    open(f"{OUT}/q1_revisit_taxonomy_out.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "summ":
        summarize(json.load(open(f"{OUT}/q1_revisit_rows.json")))
    else:
        main()
