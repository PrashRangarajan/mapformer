"""THEORY REVIEW (2026-10-14; CPU; exploratory, not registered). Two probes on toy_wrap's model (imported, not edited).

(1) Softmax temperature at init. toy_wrap starts with score scale beta * sum_c softplus(u_c) ~ 44 (log_beta 0, u 0): a
    saturated softmax (initial loss 20-28, accuracy 0). A fresh transformer starts near-uniform (q.k / sqrt(d) std ~0.3).
    --log-beta0 -4 gives a max score ~0.8. Prediction (theory review, linear-response potential): at small beta a SIGNED
    model started at the |.| angles ('absmag') crosses zero and solves, as SO_absmag did in the transformer (7/8), while
    the |.| model still fails.
(2) Why a held wrapped code drifts. AdamW's decoupled weight decay multiplies omega and W_out by (1 - lr wd) per step: a
    common scaling of a channel's two move angles. The signed cancelling set a(-) = -a(+) is invariant under it; the
    wrapped set a(+) + a(-) = 2 pi is not. Prediction: starting exactly on the wrapped code ('band'), wd = 0 keeps a larger
    wrap share than wd = 0.05; freezing omega removes part of the scaling.
Usage: python3 toy_probe.py --mode abs --init band --seed 20 --wd 0 [--freeze-omega] [--log-beta0 -4] --out x.json
"""
import argparse, json, math, sys, time
sys.path.insert(0, "/home/prashr")
import numpy as np
import torch
import torch.nn.functional as F
from mapformer.toy_wrap import Toy, walks, loss_acc, wrap_stats, set_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["signed", "abs"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", default="random")
    ap.add_argument("--freeze-omega", action="store_true")
    ap.add_argument("--log-beta0", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    m = Toy(a.mode)
    if a.init != "random":
        if a.init in ("absmag", "same"):
            with torch.no_grad():
                m.angles_init = (m.w_out(m.z) * m.omega).detach(); m.angles_init_abs = m.angles_init.abs()
        set_init(m, a.init, rng, clamp_omega=True)
    with torch.no_grad():
        m.log_beta.fill_(a.log_beta0)
    if a.freeze_omega:
        m.omega.requires_grad_(False)
    A0 = m.angles().detach().clone()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    warm = max(1, a.steps // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (s - warm) / (a.steps - warm)))))
    ev_rng = np.random.RandomState(10 ** 6); emv, epos = walks(256, a.T, rng=ev_rng)
    hist = []; t0 = time.time()
    for step in range(a.steps + 1):
        if step % 500 == 0 or step == a.steps:
            with torch.no_grad():
                l, acc = loss_acc(m, emv, epos)
            A = m.angles().detach()
            hist.append({"step": step, "loss": float(l), "acc": float(acc), "wrap_share": wrap_stats(m),
                         "opp_sign": float((torch.sign(A[0]) != torch.sign(A[1])).float().mean()),
                         "sum_err_med": float(torch.angle(torch.exp(1j * (A[0] + A[1]).double())).abs().median())})
            print(json.dumps(hist[-1]), f"{time.time() - t0:.0f}s", flush=True)
        if step == a.steps:
            break
        mv, pos = walks(64, a.T, rng=rng)
        l, _ = loss_acc(m, mv, pos)
        opt.zero_grad(); l.backward(); opt.step(); sch.step()
    A = m.angles().detach()
    json.dump({"args": vars(a), "hist": hist, "final_acc": hist[-1]["acc"], "angles0": A0.tolist(), "angles": A.tolist(),
               "omega": m.omega.detach().tolist(), "w": F.softplus(m.u).detach().tolist(),
               "beta": float(torch.exp(m.log_beta).detach())}, open(a.out, "w"))


if __name__ == "__main__":
    main()
