"""POST HOC (CPU, eval-only; plan item 7): revisit accuracy on the 1D ring by stratum, for the signed vs non-negative (|.|)
rotary step. Strata of a revisited observation at move t (cell c = position mod 32):
  gap       moves since the LAST earlier visit to c (any kind); bins GAP_BINS
  wrap-only every earlier visit to c is at unwrapped displacement +-32k, k != 0 (the cell was only ever seen "a lap away";
            a clock that works locally cannot retrieve it); otherwise "plain" (some earlier visit at the same unwrapped
            position).
Unwrapped position = cumulative sum of the move tokens (0 -> +1, 1 -> -1, environment_cancel); the start cell is not
observed, so it does not count as a visit (as in environment_cancel's revisit mask).
Stream: GridWorldCancel(size 32, seed 10000, p_plus 0.5), np.random.seed(10**6), maps redrawn per trajectory, T = 128 moves,
N_TRAJ trajectories -- the registered eval stream of train_ring_dial / train_wrap_exist / train_wrap_mech (they use 400
trajectories; the first 200 are the same 200 a 200-trajectory eval sees). Eval mode (dropout off), no 1/(1-p) re-score.
Check: the overall accuracy must equal each run's eval.json "128" accuracy (same stream, same n) -- printed per arm.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/ring_strata.py [--test]
Writes ring_strata.json and ring_strata_out.txt next to this script."""
import glob, json, os, sys

import numpy as np
import torch

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = f"{REPO}/runs"
SIZE = 32; T = 128; N_TRAJ = 400; EVAL_SEED = 10 ** 6
GAP_BINS = [(2, 8), (8, 32), (32, 129)]        # [lo, hi) moves since the last visit


def strata(moves, size=SIZE):
    """moves: (T,) array of move tokens relative to action_offset (0 = +1, 1 = -1). Returns per move t (the observation
    after move t): revisit (bool), gap (int, moves since the last earlier visit to the cell; -1 if not a revisit),
    wrap_only (bool)."""
    u = np.cumsum(np.where(np.asarray(moves) == 0, 1, -1))
    first = {}                        # cell -> list of unwrapped positions visited
    last = {}                         # cell -> last time visited
    Tn = len(u); rev = np.zeros(Tn, bool); gap = np.full(Tn, -1); wo = np.zeros(Tn, bool)
    for t in range(Tn):
        c = int(u[t] % size)
        if c in first:
            rev[t] = True; gap[t] = t - last[c]
            wo[t] = all(p != u[t] for p in first[c])
            first[c].append(int(u[t]))
        else:
            first[c] = [int(u[t])]
        last[c] = t
    return rev, gap, wo


def test():
    """Constructed cases with known labels."""
    # 32 x (+1) from 0: visits u = 1..32 (cells 1..31, 0), no revisits. Then +1 -> u 33, cell 1, only earlier visit at u 1
    # (displacement 32): wrap-only, gap 32. Then -1 -> u 32, cell 0, earlier visit at u 32: plain, gap 2.
    # Then -1 -> u 31 (cell 31, seen at u 31 at move 30: plain, gap 4). Then +1, +1 -> u 32 (plain, gap 2), u 33 (cell 1: seen at u 1
    # and u 33 -> plain, gap 4).
    mv = [0] * 32 + [0, 1, 1, 0, 0]
    rev, gap, wo = strata(mv)
    assert not rev[:32].any()
    assert (rev[32], gap[32], wo[32]) == (True, 32, True), (rev[32], gap[32], wo[32])
    assert (rev[33], gap[33], wo[33]) == (True, 2, False)
    assert (rev[34], gap[34], wo[34]) == (True, 4, False)
    assert (rev[35], gap[35], wo[35]) == (True, 2, False)
    assert (rev[36], gap[36], wo[36]) == (True, 4, False)
    # back-and-forth: +1, -1 -> u 1, 0: cell 0 never seen before (start unobserved): not a revisit; +1 -> u 1: plain, gap 2
    rev, gap, wo = strata([0, 1, 0])
    assert list(rev) == [False, False, True] and gap[2] == 2 and not wo[2]
    # 32 x (-1) then 1 x (-1): u -1..-32, then u -33 (cell 31, earlier at u -1): wrap-only, gap 32
    rev, gap, wo = strata([1] * 33)
    assert rev[32] and wo[32] and gap[32] == 32 and not rev[:32].any()
    # 64 x (+1): every move from 32 on is a wrap-only revisit; at t >= 64 cells would have two earlier laps (none here)
    rev, gap, wo = strata([0] * 70)
    assert rev[32:].all() and wo[32:].all() and (gap[32:] == 32).all()
    # agreement of the revisit flag with environment_cancel's own mask on real trajectories
    from mapformer.environment_cancel import GridWorldCancel
    env = GridWorldCancel(size=SIZE, seed=10000, p_plus=0.5); np.random.seed(123)
    for _ in range(50):
        tok, _, r = env.generate_trajectory(T)
        rv, _, _ = strata((tok[0::2] - env.action_offset).numpy())
        assert (rv == r[1::2].numpy()).all()
        pos = np.array([p[0] for p in env.visited_locations])
        u = np.cumsum(np.where((tok[0::2] - env.action_offset).numpy() == 0, 1, -1))
        assert ((pos - u) % SIZE == (pos[0] - u[0]) % SIZE).all()      # unwrapped walk matches the env's cells
    print("strata test: OK (constructed cases + revisit mask agrees with environment_cancel on 50 trajectories)")


def stream():
    from mapformer.environment_cancel import GridWorldCancel
    te = GridWorldCancel(size=SIZE, seed=10000, p_plus=0.5); np.random.seed(EVAL_SEED)
    toks, S = [], []
    for _ in range(N_TRAJ):
        tok, _, r = te.generate_trajectory(T)
        rv, gap, wo = strata((tok[0::2] - te.action_offset).numpy())
        assert (rv == r[1::2].numpy()).all()
        toks.append(tok); S.append((rv, gap, wo))
    return torch.stack(toks), S, te


def load_models():
    """(group, arm, seed) -> (model, solved, eval.json acc at 128)."""
    from mapformer.train_variant import VARIANT_MAP
    from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4
    from mapformer.stats_core import classify_run
    out = {}
    for arm in ("Signed_r4", "Abs_r4"):
        for d in sorted(glob.glob(f"{RUNS}/ring_dial/p0/{arm}_p0.5_s*/")):
            b = torch.load(f"{d}{arm}.pt", map_location="cpu", weights_only=False)
            m = VARIANT_MAP[arm](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=SIZE)
            m.load_state_dict(b["model_state_dict"])
            out["RING_DIAL", arm, b["seed"]] = (m, classify_run(b["losses"])["registered"] == "SOLVED",
                                                json.load(open(f"{d}eval.json"))["eval"]["128"]["acc"])
    for grp, root, arms in (("WRAP_EXISTENCE", "wrap_existence", ("SO", "AO")),
                            ("MECHANISM", "wrap_mech", ("SO", "AO", "AO_high"))):
        for arm in arms:
            for d in sorted(glob.glob(f"{RUNS}/{root}/p0/{arm}_s[0-9]*/")):
                b = torch.load(f"{d}{arm}.pt", map_location="cpu", weights_only=False)
                cls = MapFormerWM_SignedAO_r4 if arm.startswith("SO") else MapFormerWM_AbsAO_r4
                m = cls(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=SIZE,
                        action_ids=(0, 1))
                m.load_state_dict(b["model_state_dict"])
                out[grp, arm, b["seed"]] = (m, classify_run(b["losses"])["registered"] == "SOLVED",
                                            json.load(open(f"{d}eval.json"))["eval"]["128"]["acc"])
    return out


@torch.no_grad()
def correct(m, toks):
    m.eval(); ok = []
    for i in range(0, len(toks), 100):
        x = toks[i:i + 100]
        pred = m(x[:, :-1]).argmax(-1)                    # position p predicts token p + 1
        ok.append((pred[:, 0::2] == x[:, 1::2]).numpy())  # observation after move t is predicted at position 2t
    return np.concatenate(ok)                             # (N, T)


def main():
    test()
    toks, S, te = stream()
    rev = np.stack([s[0] for s in S]); gap = np.stack([s[1] for s in S]); wo = np.stack([s[2] for s in S])
    tgt = toks[:, 1::2].numpy()
    masks = {f"plain gap {lo}-{hi - 1}": rev & ~wo & (gap >= lo) & (gap < hi) for lo, hi in GAP_BINS}
    masks["plain, all"] = rev & ~wo
    masks["wrap-only"] = rev & wo
    masks["all revisits"] = rev
    lines = [f"stream: {N_TRAJ} trajectories x {T} moves; revisits {rev.sum()}; wrap-only {int((rev & wo).sum())} "
             f"({(rev & wo).sum() / rev.sum():.3f}); wrap-only gap range {gap[rev & wo].min()}-{gap[rev & wo].max()}"]
    floor = {}
    for k, mk in masks.items():
        v = tgt[mk]; vals, cnt = np.unique(v, return_counts=True)
        floor[k] = float(cnt.max() / cnt.sum()); lines.append(f"  {k:16s} n {mk.sum():6d}  constant-predictor floor {floor[k]:.3f}")
    M = load_models(); res = {}
    for key, (m, solved, acc_ev) in M.items():
        ok = correct(m, toks)
        r = {k: float(ok[mk].mean()) for k, mk in masks.items()}
        # plain revisits re-weighted to the wrap-only gap distribution (bins of 8 moves over 32-127): is the wrap-only
        # deficit more than the long gap?
        pw = mw = 0.0
        for g0 in range(32, 128, 8):
            wb = rev & wo & (gap >= g0) & (gap < g0 + 8); pb = rev & ~wo & (gap >= g0) & (gap < g0 + 8)
            if wb.sum() and pb.sum():
                pw += wb.sum() * ok[pb].mean(); mw += wb.sum() * ok[wb].mean()
        nw = sum(int((rev & wo & (gap >= g0) & (gap < g0 + 8)).sum()) for g0 in range(32, 128, 8)
                 if (rev & ~wo & (gap >= g0) & (gap < g0 + 8)).sum())
        r["plain, wrap-gap-matched"] = float(pw / nw); r["wrap-only, matched bins"] = float(mw / nw)
        r["solved"] = bool(solved); r["eval_json_acc"] = acc_ev
        assert abs(r["all revisits"] - acc_ev) < 2e-3, (key, r["all revisits"], acc_ev)
        res["|".join(map(str, key))] = r
    lines.append("check: overall revisit accuracy reproduces eval.json acc(128) for every run (max |diff| "
                 f"{max(abs(v['all revisits'] - v['eval_json_acc']) for v in res.values()):.1e})")
    groups = sorted({tuple(k.split("|")[:2]) for k in res}, key=lambda g: list(res).index(next(k for k in res if k.startswith('|'.join(g) + '|'))))
    cols = list(masks)
    lines.append("\n| batch | arm | SOLVED | " + " | ".join(cols) + " |")
    lines.append("|---" * (3 + len(cols)) + "|")
    lines.append("| (stream) | n targets | -- | " + " | ".join(str(int(masks[c].sum())) for c in cols) + " |")
    lines.append("| (stream) | constant floor | -- | " + " | ".join(f"{floor[c]:.3f}" for c in cols) + " |")
    for g in groups:
        rs = [v for k, v in res.items() if tuple(k.split("|")[:2]) == g]
        cells = []
        for c in cols:
            a = np.array([r[c] for r in rs]); cells.append(f"{a.mean():.3f} ({a.min():.2f}-{a.max():.2f})")
        lines.append(f"| {g[0]} | {g[1]} | {sum(r['solved'] for r in rs)}/{len(rs)} | " + " | ".join(cells) + " |")
    # per run: the wrap-only deficit relative to long plain revisits (gap >= 32), unsolved |.| runs
    from mapformer.stats_core import signflip_p
    lines.append("\nPOST HOC, per run: wrap-only minus plain revisits re-weighted to the wrap-only gap distribution (8-move bins, "
                 "32-127; bins with both kinds); mean over runs, range, paired sign-flip p over runs (descriptive):")
    for g in groups:
        rs = [v for k, v in res.items() if tuple(k.split("|")[:2]) == g]
        d = np.array([r["wrap-only, matched bins"] - r["plain, wrap-gap-matched"] for r in rs])
        p = signflip_p(d)["p"] if np.abs(d).max() > 0 else float("nan")
        lines.append(f"  {g[0]:15s} {g[1]:10s} wrap-only {np.mean([r['wrap-only, matched bins'] for r in rs]):.3f} vs plain matched "
                     f"{np.mean([r['plain, wrap-gap-matched'] for r in rs]):.3f}: {d.mean():+.3f} ({d.min():+.3f}..{d.max():+.3f}), "
                     f"{int((d < 0).sum())}/{len(d)} runs negative, sign-flip p {p:.4f}")
    txt = "\n".join(lines); print(txt)
    open(f"{HERE}/ring_strata_out.txt", "w").write(txt + "\n")
    json.dump({"floor": floor, "n": {k: int(v.sum()) for k, v in masks.items()}, "runs": res},
              open(f"{HERE}/ring_strata.json", "w"), indent=1)


if __name__ == "__main__":
    if "--test" in sys.argv:
        test()
    else:
        main()
