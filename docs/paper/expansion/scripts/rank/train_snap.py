"""CPU training with geometry snapshots (ANALYSIS for docs/paper/expansion/RANK_THEORY.md; not a registered batch).

Recipe = RANK_SWEEP's (run_rank_sweep.sh): paper torus, grid 64, 16 objects, p_empty 0.5, T = 128 moves, batch 128,
98 batches/epoch, AdamW lr 1e-3 wd 0.05, 5% warmup + cosine to 10%, grad clip 1.0, 1 layer, 2 heads, d 128;
serial data path (np seeded by --seed, as train_variant). Loss on revisited observations only (train.py).
Run here (CPU, shared machine): --batch-size 32 --epochs 40 --sdpa, i.e. a quarter of the batch and
~1/30 of the steps x batch of RANK_SWEEP: a dynamics testbed, not a reproduction. At T = 128 the rank-2 defect is visible in the END STATE (03_formal 1.4:
rank-2 heads drifting or collapsed on 32/32 while fitting the task; rank 4 drift-free on 31/32), so this is a cheap
testbed for the DYNAMICS of the drift, not for SOLVED counts.

Arms run: Vanilla (shared r2), Vanilla_r4mi (shared r4, r2's base) and (any VARIANT_MAP name also works)
R2Clock (r2 + per-channel learned clock slot, = Vanilla at init; rank_geom.MapFormerWM_r2Clock).
Writes <out>/<arm>_s<seed>.json: per-snapshot geometry (rank_geom.head_geometry), epoch losses, and final held-out
accuracy at T = 128 / 512 (eval_noise_refine-style: fresh env seed, unseen map; eval mode).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_geom import MapFormerWM_r2Clock, head_geometry, basin  # noqa: E402
from mapformer.environment import GridWorld  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402

ARMS = dict(VARIANT_MAP)
ARMS["R2Clock"] = MapFormerWM_r2Clock


@torch.no_grad()
def held_out_acc(m, T, n_traj=200, seed=10_000):
    """Revisit accuracy on fresh maps (unseen): argmax over the observation vocabulary at revisited observation slots."""
    m.eval()
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=seed)
    st = np.random.get_state(); np.random.seed(seed)
    corr = tot = 0
    for _ in range(n_traj // 20):
        tok, om, rev, _ = env.generate_batch(20, T)
        logits = m(tok[:, :-1])
        tgt = tok[:, 1:]; mask = rev[:, 1:]
        pred = logits.argmax(-1)
        corr += (pred[mask] == tgt[mask]).sum().item(); tot += mask.sum().item()
    np.random.set_state(st)
    m.train()
    return corr / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=128); ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True); ap.add_argument("--snap-every", type=int, default=49)
    ap.add_argument("--sdpa", action="store_true", help="model.USE_SDPA (CPU speed; agrees to ~1e-6 in logits)")
    a = ap.parse_args()
    if a.sdpa:
        import mapformer.model as _M
        _M.USE_SDPA = True
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=a.seed)
    m = ARMS[a.arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * a.n_batches; warm = max(1, int(0.05 * total))

    def _lr(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    crit = nn.CrossEntropyLoss()
    snaps, losses = [], []

    def snap(step, ep):
        m.eval(); g = head_geometry(m); m.train()
        snaps.append(dict(step=step, epoch=ep, heads=g, basin=basin(g)))

    snap(0, 0)
    step = 0; t0 = time.time()
    for ep in range(a.epochs):
        el = 0.0; nb = 0
        for _ in range(a.n_batches):
            tok, om, rev, _ = env.generate_batch(a.batch_size, a.n_steps)
            logits = m(tok[:, :-1]); tgt = tok[:, 1:]; mask = rev[:, 1:]
            if mask.sum() == 0:
                continue
            loss = crit(logits[mask], tgt[mask])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); step += 1
            el += loss.item(); nb += 1
            if step % a.snap_every == 0:
                snap(step, ep + 1)
        losses.append(el / max(nb, 1))
        print(f"{a.arm} s{a.seed} ep {ep + 1} loss {losses[-1]:.4f} {time.time() - t0:.0f}s "
              f"kappa {[round(h['kappa'], 3) for h in snaps[-1]['heads']]}", flush=True)
    snap(step, a.epochs)
    acc = {T: held_out_acc(m, T) for T in (128, 512)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(arm=a.arm, seed=a.seed, epochs=a.epochs, losses=losses, snaps=snaps, acc=acc,
                   final_basin=snaps[-1]["basin"]), open(f"{a.out}/{a.arm}_s{a.seed}.json", "w"))
    torch.save(m.state_dict(), f"{a.out}/{a.arm}_s{a.seed}.pt")


if __name__ == "__main__":
    main()
