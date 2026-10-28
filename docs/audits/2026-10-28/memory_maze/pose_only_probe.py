"""Post hoc (unregistered, CPU, no model): how far does the agent's pose alone take the Memory Maze Objects probe?

The registered offline probe (train_mm.run_probe) reads [model state, agent_pos, agent_dir]. targets_vec is the
agent-centric vector to each of the 6 objects; it is an exact rotation of (object world position - agent_pos) by
agent_dir (checked below: the reconstructed world positions are constant within an episode). A predictor with no memory
that is given the pose can therefore output R(dir) (mean world position of object k over the training episodes - pos).
Its squared-distance error equals the spread of the objects' world positions, whatever the pose.

Scored exactly like the probe: eval episodes, steps 501-1000, squared distance averaged over objects; training means
from train-part0 0-1599 (the models' training episodes). Also reproduces the constant baseline (training-set mean of
targets_vec), which the batch recorded as 65.09.

  python3 -I pose_only_probe.py <arrays dir>     (the prep arrays: {train0,eval}_{pos,dir,tvec}.npy)
"""
import sys
import numpy as np

D = sys.argv[1].rstrip("/") + "/"
tp, td, tv = (np.load(D + f"train0_{k}.npy", mmap_mode="r") for k in ("pos", "dir", "tvec"))
ep, ed, ev = (np.load(D + f"eval_{k}.npy", mmap_mode="r") for k in ("pos", "dir", "tvec"))


def world(P, Dd, V):
    """targets_vec -> world positions: o = p + M v, M = [[dy, dx], [-dx, dy]] (orthonormal)."""
    p, d = P[:, :, None, :], Dd[:, :, None, :]
    return np.stack([p[..., 0] + d[..., 1] * V[..., 0] + d[..., 0] * V[..., 1],
                     p[..., 1] - d[..., 0] * V[..., 0] + d[..., 1] * V[..., 1]], -1)


def agent_frame(P, Dd, O):
    """world positions O (n, 6, 2) -> targets_vec at every step: v = M^T (o - p)."""
    p, d = P[:, :, None, :], Dd[:, :, None, :]
    r = O[:, None] - p
    return np.stack([d[..., 1] * r[..., 0] - d[..., 0] * r[..., 1], d[..., 0] * r[..., 0] + d[..., 1] * r[..., 1]], -1)


chk = world(np.asarray(tp[:50]), np.asarray(td[:50]), np.asarray(tv[:50]))
print(f"convention check: within-episode sd of reconstructed world positions (50 train episodes) "
      f"{float(chk.std(1).mean()):.2e} (0 = exact)")
O_tr = np.concatenate([world(np.asarray(tp[i:i + 100]), np.asarray(td[i:i + 100]), np.asarray(tv[i:i + 100]))[:, 0]
                       for i in range(0, 1600, 100)])
obar = O_tr.mean(0)
P, Dd, Y = np.asarray(ep[:, 501:1001]), np.asarray(ed[:, 501:1001]), np.asarray(ev[:, 501:1001])
pose_only = ((agent_frame(P, Dd, np.broadcast_to(obar, (len(P), 6, 2))) - Y) ** 2).sum(-1).mean()
tv_mean = np.concatenate([np.asarray(tv[i:i + 100]).reshape(-1, 6, 2) for i in range(0, 1600, 100)]).mean(0)
const = ((Y - tv_mean) ** 2).sum(-1).mean()
print(f"Objects (squared distance averaged over objects), eval steps 501-1000, {len(P)} episodes:")
print(f"  constant baseline (training mean of targets_vec)            {const:.2f}")
print(f"  pose only, no memory (training-mean world position, rotated) {pose_only:.2f}")
print("  registered probe, every arm (MEMORY_MAZE_ANALYSIS.txt)       36.86-37.02 blinded, 36.78-37.19 unblinded")
