"""Gate for the new-object transfer task (NEWOBJ_PREREG.md), calling environment_newobj and model_codes."""
import numpy as np, torch
from mapformer.environment_newobj import NewObjectWorld, N_SPECIAL, BLANK
from mapformer.model_codes import use_object_codes, set_pool
from mapformer.train_newobj import ARMS

T = 1024
for pool in ("train", "test"):
    env = NewObjectWorld(size=32, seed=10000, pool=pool); np.random.seed(10**6)
    ids, tot, c_const, c_ret, rv = set(), 0, 0, 0, []
    for _ in range(60):
        tok, om, rev = env.generate_trajectory(T); a = tok[0::2].numpy(); o = tok[1::2].numpy(); r = rev[1::2].numpy()
        ids |= set(o[o >= N_SPECIAL].tolist()); rv.append(r.mean())
        # retrace: while a run reverses the previous run, predict the observation j steps back in the previous run
        prev_len = cur = 0; k = 0
        for t in range(T):
            if t > 0 and a[t] == a[t - 1]: cur += 1
            else:
                rev_run = t > 0 and (a[t] ^ 1) == a[t - 1]
                prev_len = cur if rev_run else 0; cur = 1; k = 0
            if prev_len > 0:
                k += 1
            if r[t]:
                tot += 1; c_const += o[t] == BLANK
                guess = o[t - 2 * k] if (prev_len > 0 and k <= prev_len and t - 2 * k >= 0) else BLANK
                c_ret += guess == o[t]
    lo, hi = (N_SPECIAL, N_SPECIAL + 1000) if pool == "train" else (N_SPECIAL + 1000, N_SPECIAL + 2000)
    print(f"[{pool}] object ids in [{min(ids)}, {max(ids)}] within [{lo}, {hi}): {lo <= min(ids) and max(ids) < hi}; "
          f"revisit rate {np.mean(rv):.3f}; floors: constant {c_const / tot:.3f}, retrace {c_ret / tot:.3f} ({tot} targets)")
V = NewObjectWorld().unified_vocab_size
t = torch.randint(0, V, (2, 300)); t2 = t.clone(); t2[:, 250:] = torch.randint(0, V, (2, 50))
codes = None
for name, cls in ARMS.items():
    torch.manual_seed(0); m = cls(vocab_size=V, d_model=128, n_heads=2, n_layers=1, grid_size=32)
    use_object_codes(m, N_SPECIAL, 1000); m.eval()
    with torch.no_grad():
        y1, y2 = m(t)[:, :250], m(t2)[:, :250]; fin = torch.isfinite(y1)
        leak = (y1[fin] - y2[fin]).abs().max().item() if bool(fin.all() == torch.isfinite(y2).all()) else float('nan')
        set_pool(m, "test"); ok_mask = torch.isinf(m(t)[..., N_SPECIAL:N_SPECIAL + 1000]).all().item()
    same = codes is None or torch.equal(codes, m.token_emb.codes); codes = m.token_emb.codes
    print(f"{name:8s} causal leak {leak:.1e}; test mode masks train-pool logits: {ok_mask}; codebook shared: {same}")
