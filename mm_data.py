"""Memory Maze batch data layer (MEMORY_MAZE_PREREG.md). numpy only; imports the gates' helpers unchanged.

Arrays (written once by `mm_encoder.py prep`, a few large .npy files per split; GPFS dislikes many small files):
  <dir>/<split>_emb.npy    float16 (N, 1001, 256)  frozen-encoder embedding of image[t]
  <dir>/<split>_act.npy    int8    (N, 1001)       action index of action[t] (-1 at t = 0, the dummy row)
  <dir>/<split>_pos.npy    float32 (N, 1001, 2)    agent_pos
  <dir>/<split>_dir.npy    float32 (N, 1001, 2)    agent_dir
  <dir>/<split>_layout.npy uint8   (N, 15, 15)     maze_layout[0] (constant within an episode; checked at prep)
  <dir>/<split>_tvec.npy   float32 (N, 1001, 6, 2) targets_vec (agent-centric object positions; offline probe)
  <dir>/<split>_names.txt  one npz member name per line, in sorted-archive order; row i = archive index lo + i
Splits: "train0" = 15x15 train-part0 indices 0-2899 (training uses 0-1599; 1600-2899 is the held-out secondary
scoring set), "eval" = the official 15x15 eval archive (primary scoring set).

Sequence layout used by every arm: [f_0, s_1, f_1, s_2, f_2, ..., s_1000, f_1000] (T = 2001 tokens). s_t is the step
token for action a_t (it produced frame t); the model's output at s_t predicts f_t. Frame tokens on blind steps carry a
learned blank vector instead of the embedding.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_gate as g1  # noqa: E402
import mm_gate_v2 as g2  # noqa: E402

N_ACT = 6
N_PAIR = 7 * N_ACT          # previous action in {none, 0..5} x action in 0..5
E_DIM = N_PAIR * 4          # pair one-hot (42) x [dir_{t-1}, dir_t] (4) = 168
B_BLIND, S_SIGHT = 16, 16   # the registered cell (MEMORY_MAZE_GATE_V3.md)
LONG_GAP = g1.LONG_GAP      # 128
TRAIN_RANGE = (0, 1600)     # train-part0 indices used for training
HELD_RANGE = (1600, 2900)   # train-part0 indices held out (secondary scoring set)
ENC_RANGE = (0, 300)        # train-part0 indices the frame encoder is trained on


def load_split(d, split, mmap=True):
    m = "r" if mmap else None
    out = {k: np.load(os.path.join(d, f"{split}_{k}.npy"), mmap_mode=m) for k in ("emb", "act", "pos", "dir", "layout", "tvec")}
    with open(os.path.join(d, f"{split}_names.txt")) as f:
        out["names"] = [x.strip() for x in f if x.strip()]
    return out


# ---------------------------------------------------------------- step features
def e_feature(act, dr):
    """E: one-hot (a_{t-1}, a_t) times [dir_{t-1}, dir_t]; row t is the feature of step token s_t (t >= 1); row 0 zero.
    act (T,) int with -1 at t = 0; dr (T, 2). Returns (T, 168) float32. Linear in heading per action pair, so a linear
    step map on it is exactly the gate's class E (mm_gate.fit's ME, per (previous action, action) pair)."""
    T = len(act)
    out = np.zeros((T, N_PAIR, 4), np.float32)
    ap = np.where(act[:-1] >= 0, act[:-1], 6)            # previous action, 6 = none
    a = act[1:]
    ok = a >= 0
    pair = ap * N_ACT + np.maximum(a, 0)
    h = np.hstack([dr[:-1], dr[1:]]).astype(np.float32)   # (T-1, 4)
    rows = np.arange(1, T)[ok]
    out[rows, pair[ok]] = h[ok]
    return out.reshape(T, E_DIM)


def odo_feature(pos):
    """Odometry: d_t = pos_t - pos_{t-1} (row 0 zero). (T, 2) float32."""
    d = np.zeros_like(pos, dtype=np.float32)
    d[1:] = np.diff(pos, axis=0)
    return d


def raw_feature(act):
    """Raw egocentric action one-hot (row 0 zero). (T, 6) float32."""
    out = np.zeros((len(act), N_ACT), np.float32)
    ok = act >= 0
    out[np.where(ok)[0], act[ok]] = 1.0
    return out


# ---------------------------------------------------------------- pose bins, blinding, scored steps
def pose_bins(pos, dr, N=15):
    cell = np.clip(np.floor(pos).astype(int), 0, N - 1)
    cid = cell[:, 0] * N + cell[:, 1]
    return cid * g1.N_HB + g1.heading_bin(dr), cid


def sighted_mask(T, ep_idx, B=B_BLIND, S=S_SIGHT):
    """The gate's schedule exactly (mm_gate_v2.sighted_mask): phase offset seeded by [ep_idx, B, S]."""
    return g2.sighted_mask(T, B, S, ep_idx)


def random_sighted_mask(T, rng, B=B_BLIND, S=S_SIGHT):
    """Training-time schedule: same period and lengths, random phase per sample (frame 0 sighted)."""
    off = int(rng.integers(0, S + B))
    sm = ((np.arange(T) + off) % (S + B)) < S
    sm[0] = True
    return sm


def scored_steps(pose, cid, sm, B=B_BLIND):
    """Blind re-entries exactly as mm_gate_v2.episode defines them: t = first sighted step after a full-length blind
    run [b0, t-1], pose(t) != pose(b0-1), pose(t) sighted at some s < b0. Returns a list of dicts
    {t, b0, s, gap, back, nonlocal}. (test_mm checks equality with the gate's rows.)"""
    T = len(pose)
    rows = []
    for t in range(1, T):
        if not (sm[t] and not sm[t - 1]):
            continue
        b0 = t
        while b0 > 0 and not sm[b0 - 1]:
            b0 -= 1
        if t - b0 != B or b0 - 2 < 0:
            continue
        p = pose[t]
        if pose[b0 - 1] == p:
            continue
        prev = np.where((pose[:b0] == p) & sm[:b0])[0]
        if len(prev) == 0:
            continue
        s = int(prev[-1])
        r = g1.rle(list(cid[s:t + 1]))
        back = r == r[::-1]
        gap = t - s
        rows.append(dict(t=t, b0=b0, s=s, gap=gap, back=bool(back), nonlocal_=bool(gap >= LONG_GAP and not back)))
    return rows


def fit_e_map(acts, dirs, poss):
    """Least-squares E map (the gate's ME): per (previous action, action) pair a 4x2 matrix from [dir_{t-1}, dir_t] to
    the displacement. Returns M (7, 6, 4, 2). Used only to INSTALL a solution (rule 14), never as an arm's step."""
    FF = np.zeros((7, N_ACT, 4, 4)); Fd = np.zeros((7, N_ACT, 4, 2))
    for act, dr, pos in zip(acts, dirs, poss):
        act = np.asarray(act).astype(int); d = np.diff(pos, axis=0)
        Fh = np.hstack([dr[:-1], dr[1:]])
        ap = np.where(act[:-1] >= 0, act[:-1], 6); a = act[1:]; ok = a >= 0
        for i in np.where(ok)[0]:
            FF[ap[i], a[i]] += np.outer(Fh[i], Fh[i]); Fd[ap[i], a[i]] += np.outer(Fh[i], d[i])
    M = np.zeros((7, N_ACT, 4, 2))
    for i in range(7):
        for j in range(N_ACT):
            M[i, j] = np.linalg.solve(FF[i, j] + 1e-6 * np.eye(4), Fd[i, j])
    return M


def e_map_as_matrix(M):
    """The E map as a (168, 2) matrix acting on e_feature rows: e_feature(t) @ W = predicted displacement."""
    return M.reshape(N_PAIR * 4, 2)
