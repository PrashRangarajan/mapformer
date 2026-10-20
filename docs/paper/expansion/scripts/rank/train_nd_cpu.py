"""CPU pilot (ANALYSIS, post hoc, NOT registered) for docs/paper/expansion/RANK_THEORY.md, check C7: MapFormer's own
reported failure -- MapWM with inner rank = D in 5D (v4 Table 6: grid 5, sequence length 16, IID 0.75, "on par with
RoPE") -- under our recipe, with per-head rank D, D + 1, and rank D plus a per-channel clock slot.

Task: environment_nd.GridWorldND(dims, size), directed walk (runs of 1-10), map redrawn per seed; T moves. Loss on
revisited observations only. Recipe (ours, not the paper's, which is unstated for 5D): AdamW lr 1e-3 wd 0.05, 5% warmup
+ cosine, batch 128, 98 batches / epoch, 1 layer, 2 heads, d 128 (head size 64, as the paper), attention dropout 0.1.
Arms: PH<r> = per-head rank r (model_rank_perhead.MapFormerWM_PerHead with PER_HEAD_R = r, built from Vanilla's base);
PH<r>C = PH<r> + clock slot (rank_geom.with_clock; equal to PH<r> at init); RoPE = index RoPE (MapFormerWM_RoPE).
Readouts: SOLVED (final-5% training loss < 0.05, the registered rule), held-out accuracy on unseen maps at T and 2T
(eval mode), per-head geometry (rank_geom.head_geometry, D = dims) at snapshots and at the end.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_geom import head_geometry, basin, with_clock  # noqa: E402
from mapformer.environment_nd import GridWorldND  # noqa: E402
from mapformer.model_rank_perhead import MapFormerWM_PerHead  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402
from mapformer.stats_core import classify_run  # noqa: E402


def make(arm):
    if arm == "RoPE":
        return VARIANT_MAP["RoPE"]
    clock = arm.endswith("C")
    r = int(arm[2:-1] if clock else arm[2:])
    cls = type(f"PH{r}", (MapFormerWM_PerHead,), {"PER_HEAD_R": r})
    return with_clock(cls) if clock else cls


@torch.no_grad()
def held_out_acc(m, dims, size, T, n=1000, seed=10_000):
    m.eval()
    env = GridWorldND(dims=dims, size=size, n_obs_types=16, p_empty=0.5, seed=seed)
    st = np.random.get_state(); np.random.seed(seed)
    corr = tot = 0
    for _ in range(n // 100):
        tok, om, rev, _ = env.generate_batch(100, T)
        pred = m(tok[:, :-1]).argmax(-1); tgt = tok[:, 1:]; mask = rev[:, 1:]
        corr += (pred[mask] == tgt[mask]).sum().item(); tot += mask.sum().item()
    np.random.set_state(st); m.train()
    return corr / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dims", type=int, default=5); ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--T", type=int, default=16); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-batches", type=int, default=98); ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--snap-every", type=int, default=490); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    env = GridWorldND(dims=a.dims, size=a.size, n_obs_types=16, p_empty=0.5, seed=a.seed)
    m = make(a.arm)(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=a.size)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * a.n_batches; warm = max(1, int(0.05 * total))

    def _lr(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    crit = nn.CrossEntropyLoss()
    geo = a.arm != "RoPE"
    snaps, losses = [], []

    def snap(step, ep):
        if geo:
            m.eval(); g = head_geometry(m, D=a.dims); m.train()
            snaps.append(dict(step=step, epoch=ep, heads=g, basin=basin(g)))

    snap(0, 0); step = 0; t0 = time.time()
    for ep in range(a.epochs):
        el = 0.0; nb = 0
        for _ in range(a.n_batches):
            tok, om, rev, _ = env.generate_batch(a.batch_size, a.T)
            logits = m(tok[:, :-1]); tgt = tok[:, 1:]; mask = rev[:, 1:]
            if mask.sum() == 0:
                continue
            loss = crit(logits[mask], tgt[mask])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); step += 1; el += loss.item(); nb += 1
            if step % a.snap_every == 0:
                snap(step, ep + 1)
        losses.append(el / max(nb, 1))
        if (ep + 1) % 10 == 0:
            print(f"{a.arm} s{a.seed} ep {ep + 1} loss {losses[-1]:.4f} {time.time() - t0:.0f}s", flush=True)
    snap(step, a.epochs)
    acc = {a.T: held_out_acc(m, a.dims, a.size, a.T), 2 * a.T: held_out_acc(m, a.dims, a.size, 2 * a.T)}
    cl = classify_run(losses)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(arm=a.arm, seed=a.seed, dims=a.dims, size=a.size, T=a.T, epochs=a.epochs, losses=losses,
                   snaps=snaps, acc=acc, cls=cl["registered"], tail=cl.get("tail"),
                   final_basin=snaps[-1]["basin"] if snaps else None),
              open(f"{a.out}/{a.arm}_s{a.seed}.json", "w"))


if __name__ == "__main__":
    main()
