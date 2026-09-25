"""Is bf16 autocast safe for the MAESTRO batch? Registered BEFORE running.

Project rule: verify a fast path is loss-equivalent before a batch relies on it.
bf16 cannot be row-exact, so the question is narrower and sharper:

  Does bf16 change the DIFFERENCES BETWEEN ARMS?

That is the only thing that matters for a 2x2, and it is where bf16 could bite:
path integration cumulatively sums per-token increments over 2048 positions, and
the resulting angle is multiplied by frequencies up to ~1 rad/token before cos/sin.
If bf16 error accumulates there, it would hurt the path-integrated arms more than
the index arms -- a confound between arms, not just noise.

Design: each of the 4 arms at the MAESTRO config (d=384, 6 layers, 8 heads,
seq 2048, batch 4), trained 400 steps in fp32 and in bf16-autocast from the SAME
init and the SAME batches (real byte data, not random tokens).

LICENSE bf16 only if, for every arm:
  (a) mean |loss_fp32 - loss_bf16| over the last 100 steps < 0.02 nats, AND
  (b) the between-arm gap (arm - RoPE, last-100 mean) moves by < 0.005 nats,
      i.e. well under the 0.015 published PoPE-RoPE effect we would be measuring, AND
  (c) at init, bf16 logits differ from fp32 by a similar amount on every arm
      (no arm-specific precision penalty).
"""
import os, time, json
import numpy as np, torch, torch.nn.functional as F
from mapformer.train_hourglass_enwik8 import build

REPO = os.path.dirname(os.path.abspath(__file__))
ARMS = ["RoPE", "PoPE-Flat", "Vanilla", "MapPoPE-Flat"]
STEPS, B, T = 400, 4, 2048


def batches(data, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(data) - T - 1, B)
        x = np.stack([data[i:i + T] for i in idx]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + T] for i in idx]).astype(np.int64)
        out.append((torch.from_numpy(x), torch.from_numpy(y)))
    return out


def run(arm, bf16, bs, init_sd):
    torch.manual_seed(0)
    m = build(arm, dim=384, heads=8, n_layers=6, grid_size=T, bottleneck_r=4).cuda()
    m.load_state_dict(init_sd)
    opt = torch.optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.01, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))
    losses = []
    torch.cuda.synchronize(); t0 = time.time()
    for x, y in bs:
        x, y = x.cuda(), y.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = m(x)
        loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(float(loss))
    torch.cuda.synchronize()
    return np.array(losses), STEPS / (time.time() - t0)


@torch.no_grad()
def init_logit_gap(arm, init_sd, x):
    m = build(arm, dim=384, heads=8, n_layers=6, grid_size=T, bottleneck_r=4).cuda().eval()
    m.load_state_dict(init_sd)
    a = m(x).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        b = m(x).float()
    return float((a - b).abs().max()), float((a - b).abs().mean())


def main():
    data = np.fromfile(os.path.join(REPO, "data", "code_train.bin"), dtype=np.uint8)
    bs = batches(data, STEPS, seed=1234)
    res = {}
    for arm in ARMS:
        torch.manual_seed(0)
        sd = {k: v.clone() for k, v in build(arm, dim=384, heads=8, n_layers=6,
              grid_size=T, bottleneck_r=4).state_dict().items()}
        gmax, gmean = init_logit_gap(arm, sd, bs[0][0].cuda())
        l32, s32 = run(arm, False, bs, sd)
        l16, s16 = run(arm, True, bs, sd)
        res[arm] = dict(l32=l32, l16=l16, s32=s32, s16=s16, gmax=gmax, gmean=gmean)
        tail = np.abs(l32[-100:] - l16[-100:]).mean()
        print(f"{arm:13s} init logit |diff| max {gmax:.4f} mean {gmean:.5f} | "
              f"last-100 |dloss| {tail:.4f} | fp32 {s32:.2f} it/s  bf16 {s16:.2f} it/s "
              f"({s16/s32:.2f}x)", flush=True)
        torch.cuda.empty_cache()

    print("\nBETWEEN-ARM GAPS (arm - RoPE, mean loss over last 100 steps):")
    ok = True
    r32, r16 = res["RoPE"]["l32"][-100:].mean(), res["RoPE"]["l16"][-100:].mean()
    for arm in ARMS[1:]:
        g32 = res[arm]["l32"][-100:].mean() - r32
        g16 = res[arm]["l16"][-100:].mean() - r16
        moved = abs(g16 - g32)
        ok &= moved < 0.005
        print(f"  {arm:13s} fp32 {g32:+.4f}  bf16 {g16:+.4f}  moved {moved:.4f} "
              + ("OK" if moved < 0.005 else "*** FAILS (b) ***"))
    for arm in ARMS:
        tail = np.abs(res[arm]["l32"][-100:] - res[arm]["l16"][-100:]).mean()
        ok &= tail < 0.02
    speed = np.mean([res[a]["s16"] / res[a]["s32"] for a in ARMS])
    gm = [res[a]["gmean"] for a in ARMS]
    print(f"\ninit logit mean |diff| across arms: " + ", ".join(f"{a} {res[a]['gmean']:.5f}" for a in ARMS)
          + f"  (spread {max(gm)/max(min(gm),1e-12):.1f}x)")
    print(f"mean speedup {speed:.2f}x")
    print("\nVERDICT: " + ("bf16 LICENSED for the MAESTRO batch" if ok else "bf16 NOT licensed -- keep fp32"))
    json.dump({a: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
               for a, d in res.items()}, open(os.path.join(REPO, "BF16_CHECK.json"), "w"))


if __name__ == "__main__":
    main()
