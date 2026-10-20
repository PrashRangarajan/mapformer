"""SEARCH_THEORY A3 (CPU, exploratory, not registered): the step-map landscape with the content held at a solution.

Content (embeddings, attention, FFN, output) is taken from a SOLVED signed observation-step-free ring model
(runs/wrap_mech/p0/SO_s<seed>, MECHANISM) and FROZEN. Only the step map (W_in, W_out, omega) trains. Both classes have an exact
solution with this content: the source's signed code (signed_balanced) and its wrapped |.| copy (balanced; WRAP_EXISTENCE
Amendment 1 construction, `model_wrap_exist.wrap_construct`). So any difference between the classes here is a difference of
the step map's search alone, with nothing else co-adapting.

Starts (balanced gauge, `train_wrap_mech.set_move_angles`; the seed's default-init angles are read from the source's eval.json):
  S_def      signed, the default (zero-mean, small) angles
  S_abs      signed, |default| (both moves rotate the same way: the |.| start)
  S_abs_fz   as S_abs with W_in frozen (the move latents cannot move: no collective route; W_out and omega free)
  A_abs      |.|, |default|
  A_high     |.|, a(+1), a(-1) iid U(1.7, 4.5) (MECHANISM's AO_high draw)
  A_seg      |.|, angles (1 - tau) * wrapped solution + tau * |default| (per channel, stays >= 0): basin along the segment
  S_seg      signed, (1 - tau) * signed solution + tau * |default|
Eval: revisit accuracy on a fresh stream (GridWorldCancel seed 10000, np seed 10**6, 100 walks, T = 128), eval mode.
  python3 step_basin.py --seed 800 --arm A_abs --steps 1500 --out runs_basin/A_abs_s800.json
"""
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, "/home/prashr")
import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_cancel import GridWorldCancel
from mapformer.model_wrap_exist import (MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, wrap_construct, action_angles,
                                        wrap_share)
from mapformer.train_wrap_mech import set_move_angles
from mapformer.train_cancel import evaluate

R = Path("/home/prashr/mapformer/runs/wrap_mech/p0")


def build_models(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    env = GridWorldCancel(size=32, seed=seed, p_plus=0.5)
    kw = dict(d_model=128, n_heads=2, n_layers=1, grid_size=32)
    act = (env.action_offset, env.action_offset + 1)
    src = MapFormerWM_SignedAO_r4(env.unified_vocab_size, action_ids=act, **kw)
    src.load_state_dict(torch.load(R / f"SO_s{seed}/SO.pt", map_location="cpu", weights_only=False)["model_state_dict"])
    a_sol, rep_a = wrap_construct(src, env.unified_vocab_size, act, mode="balanced", **kw)
    s_sol, rep_s = wrap_construct(src, env.unified_vocab_size, act, mode="signed_balanced", **kw)
    default = torch.tensor(json.load(open(R / f"SO_s{seed}/eval.json"))["default_angles"])
    return env, a_sol, s_sol, default, rep_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=["S_def", "S_abs", "S_abs_fz", "A_abs", "A_high", "A_seg", "S_seg",
                                                     "A_sol", "S_sol"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    env, a_sol, s_sol, default, rep = build_models(a.seed)
    signed = a.arm.startswith("S")
    m = s_sol if signed else a_sol
    with torch.no_grad():
        sol_ang = action_angles(m).clone()                          # (2, C): signed angles or wrapped [0, 2 pi)
    err = None
    if a.arm == "S_def":
        err = set_move_angles(m, default)
    elif a.arm in ("S_abs", "S_abs_fz", "A_abs"):
        err = set_move_angles(m, default.abs())
    elif a.arm == "A_high":
        g = np.random.RandomState(a.seed + 7919)
        err = set_move_angles(m, torch.from_numpy(g.uniform(1.7, 4.5, default.shape)).float())
    elif a.arm in ("A_seg", "S_seg"):
        err = set_move_angles(m, (1 - a.tau) * sol_ang + a.tau * default.abs())
    # A_sol / S_sol: start at the solution (held-ness control)
    for n, p in m.named_parameters():
        p.requires_grad_(n in ("action_to_lie.w_in.weight", "action_to_lie.w_out.weight", "path_integrator.omega"))
    if a.arm == "S_abs_fz":
        m.action_to_lie.w_in.weight.requires_grad_(False)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    warm = max(1, a.steps // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (s - warm) / (a.steps - warm)))))
    np.random.seed(a.seed + 12345)                                  # the same walk stream for every arm at a seed
    hist = []; t0 = time.time(); losses = []
    acc0, nll0 = evaluate(m, GridWorldCancel(size=32, seed=10000, p_plus=0.5), 128, 100, "cpu", seed=10 ** 6)
    np.random.seed(a.seed + 12345)
    hist.append({"step": 0, "acc": acc0, "nll": nll0, "wrap": wrap_share(m)[1]})
    m.eval()                                                        # no dropout: deterministic step-map search
    for step in range(1, a.steps + 1):
        tok, _om, rev, _loc = env.generate_batch(a.batch, 128)
        x, y = tok[:, :-1], tok[:, 1:]; msk = rev[:, 1:].reshape(-1)
        if msk.sum() == 0:
            continue
        lg = m(x)
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1])[msk], y.reshape(-1)[msk])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        losses.append(float(loss))
        if step % a.log_every == 0 or step == a.steps:
            hist.append({"step": step, "loss": float(np.mean(losses[-a.log_every:])), "wrap": wrap_share(m)[1],
                         "t": time.time() - t0})
            print(f"{a.arm} s{a.seed} step {step} loss {hist[-1]['loss']:.4f} wrap {hist[-1]['wrap']:.2f} ({time.time() - t0:.0f}s)",
                  flush=True)
    st = np.random.get_state()
    acc, nll = evaluate(m, GridWorldCancel(size=32, seed=10000, p_plus=0.5), 128, 100, "cpu", seed=10 ** 6)
    np.random.set_state(st)
    final = float(np.mean(losses[-max(1, len(losses) // 20):]))
    res = {"args": vars(a), "init_angle_err": err, "acc0": acc0, "acc": acc, "nll": nll, "final_loss": final,
           "solved": bool(final < 0.05), "wrap_end": wrap_share(m), "hist": hist, "construct": rep,
           "angles_end": action_angles(m).tolist(), "secs": time.time() - t0}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(f"DONE {a.arm} s{a.seed} acc0 {acc0:.3f} -> acc {acc:.3f} final loss {final:.4f} solved {res['solved']}", flush=True)


if __name__ == "__main__":
    main()
