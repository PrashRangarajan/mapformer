"""Re-run a continuation-recipe run with snapshots (review probe; not a repo script).
Replicates mapformer.train.train (cosine, 900-epoch schedule, AdamW wd .05, clip 1.0,
data-workers 3, base_seed = seed + offset) and saves state every --every epochs up to --stop."""
import argparse, math, os, time, sys
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.data_parallel import ParallelBatchGenerator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--offset", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=900); ap.add_argument("--stop", type=int, default=400)
    ap.add_argument("--every", type=int, default=25); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=a.seed)
    m = VARIANT_MAP[a.variant](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    b = torch.load(a.init, map_location="cpu", weights_only=False)
    m.load_state_dict(b["model_state_dict"]); dev = torch.device(a.device); m = m.to(dev)
    opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    nb = 98; total = a.epochs * nb; warm = max(1, int(0.05 * total))
    def _lr(step):
        if step < warm: return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    sch = optim.lr_scheduler.LambdaLR(opt, _lr); crit = nn.CrossEntropyLoss()
    gen = ParallelBatchGenerator(env, 16, 1024, n_workers=3, base_seed=(torch.initial_seed() + a.offset) % (2**31))
    os.makedirs(a.out, exist_ok=True); losses = []
    torch.save({"model_state_dict": m.state_dict(), "epoch": 0, "losses": []}, f"{a.out}/ep0000.pt")
    for ep in range(a.stop):
        m.train(); el = 0.0; t0 = time.time()
        for _ in range(nb):
            tok, obs, rev, _l = gen.next_batch(); tok = tok.to(dev); rev = rev.to(dev)
            logits = m(tok[:, :-1]); tm = rev[:, 1:]
            if tm.sum() == 0: continue
            loss = crit(logits[tm], tok[:, 1:][tm])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(); el += loss.item()
        losses.append(el / nb)
        if (ep + 1) % 5 == 0:
            print(f"ep {ep+1} loss {losses[-1]:.4f} lr {sch.get_last_lr()[0]:.2e} {time.time()-t0:.1f}s", flush=True)
        if (ep + 1) % a.every == 0:
            torch.save({"model_state_dict": m.state_dict(), "epoch": ep + 1, "losses": list(losses)}, f"{a.out}/ep{ep+1:04d}.pt")
    gen.close()

if __name__ == "__main__":
    main()
