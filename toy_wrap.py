"""Minimal model of the ring sign result (WRAP_MECHANISM; CPU). No transformer: only rotary channels and retrieval.

C rotary channels with frequencies omega_c (PathIntegrator's geometric init, grid 32, learnable). The two moves have steps
s(a) = W_out z(a) (rank-4 latents z(a), as SignedAO_r4 / AbsAO_r4); the angle per move is omega_c * s_c(a) (signed) or
omega_c * |s_c(a)| (non-negative). theta_c(t) = sum of the angles of the moves up to t. A query at move t retrieves an
earlier move t' by softmax_t' ( beta * sum_c w_c cos(theta_c(t) - theta_c(t')) ), w_c = softplus(u_c); the loss is
-log sum over earlier t' at the SAME cell (queries whose cell was visited before). Walks: GridWorldCancel's walk (ring
32, p_plus, runs of 1-10 moves), T moves. Initial scale matched to the real model (z ~ N(0, 1) * 0.05 * sqrt(128), W_out
default Linear init: initial |angle| mean ~0.5 rad as measured on SignedAO_r4).
  python3 -m mapformer.toy_wrap --mode abs --seed 0 [--init wrap|band|offband] [--freeze-omega]
"""
import argparse, json, math, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def walks(n, T, size=32, p_plus=0.5, rng=None):
    """(n, T) move signs (+1 / -1) and positions after each move, GridWorldCancel's walk (runs of k ~ U{1..10})."""
    rng = rng or np.random
    mv = np.empty((n, T), np.int64); pos = np.empty((n, T), np.int64)
    for b in range(n):
        p = rng.randint(0, size); t = 0
        while t < T:
            a = 1 if rng.random() < p_plus else -1
            for _ in range(rng.randint(1, 11)):
                if t >= T:
                    break
                p = (p + a) % size; mv[b, t] = a; pos[b, t] = p; t += 1
    return torch.from_numpy(mv), torch.from_numpy(pos)


class Toy(nn.Module):
    def __init__(self, mode, C=64, r=4, grid=32):
        super().__init__()
        self.mode = mode
        i = torch.arange(C) % (C // 2)                      # two heads of C/2 channels, as MapWM (n_heads 2, nb 32)
        self.omega = nn.Parameter(2 * math.pi * (1.0 / grid) ** (i / (C // 2 - 1)))
        self.z = nn.Parameter(torch.randn(2, r) * 0.05 * math.sqrt(128))   # latents of +1 / -1 (|W_in e| scale)
        self.w_out = nn.Linear(r, C, bias=False)
        self.u = nn.Parameter(torch.zeros(C))
        self.log_beta = nn.Parameter(torch.zeros(()))

    def steps(self):
        s = self.w_out(self.z)                              # (2, C): +1, -1
        return s.abs() if self.mode == "abs" else s

    def angles(self):
        return self.steps() * self.omega                    # (2, C)

    def forward(self, mv):
        A = self.angles()
        per = torch.where((mv == 1)[..., None], A[0], A[1])  # (B, T, C)
        th = per.cumsum(1)
        w = F.softplus(self.u)
        q = torch.cat([torch.cos(th) * w, torch.sin(th) * w], -1)     # sum_c w_c cos(th_t - th_t') as a matmul
        k = torch.cat([torch.cos(th), torch.sin(th)], -1)
        return torch.exp(self.log_beta) * q @ k.transpose(1, 2)


def obs_of(pos, seed_rng, size=32, n_types=16, p_empty=0.5):
    """ACCOUNTS T3: an observation map per trajectory (as environment_cancel: occupied with prob 1 - p_empty, a type in
    0..15, else blank = 16) and the observation at every move's cell."""
    B = pos.shape[0]
    occ = seed_rng.random_sample((B, size)) >= p_empty
    om = np.where(occ, seed_rng.randint(0, n_types, (B, size)), n_types)
    return torch.from_numpy(np.take_along_axis(om, pos.numpy(), 1))


def loss_acc(model, mv, pos, obs=None):
    """target 'cell' (obs None): correct keys are earlier moves at the same cell. target 'obs' (ACCOUNTS T3): correct keys are
    earlier moves whose cell shows the same observation as the query's cell (what the transformer task rewards); queries
    are still the revisits."""
    B, T = mv.shape
    logits = model(mv)
    causal = torch.ones(T, T, dtype=torch.bool).tril(-1)    # keys strictly earlier
    q = ((pos[:, :, None] == pos[:, None, :]) & causal).any(-1)   # queries whose cell was visited before
    same = ((pos[:, :, None] == pos[:, None, :]) if obs is None else (obs[:, :, None] == obs[:, None, :])) & causal
    lg = logits.masked_fill(~causal, float("-inf"))
    lse = torch.logsumexp(lg, -1)
    good = torch.logsumexp(lg.masked_fill(~same, float("-inf")), -1)
    loss = (lse - good)[q].mean()
    acc = same.gather(-1, lg.argmax(-1, keepdim=True))[..., 0][q].float().mean()
    return loss, acc


def wrap_stats(model, tol=0.1, big=0.05):
    A = model.angles().detach().double()
    sel = torch.angle(torch.exp(1j * A[0])).abs() > big
    w = torch.angle(torch.exp(1j * (A[0] + A[1])))[sel].abs()
    return float((w < tol).double().mean()) if w.numel() else float("nan")


def set_init(model, kind, rng, clamp_omega=True):
    """kind: wrap / band = steps on sigma = 2 pi (abs) or -a (signed) with a ~ U(0.2, 2 pi - 0.2); offband = a+, a- iid
    U(0.2, 2 pi - 0.2). Balanced gauge: z = c e1, c e2; W_out columns = angles / omega / c."""
    with torch.no_grad():
        C = model.omega.numel(); c = float(model.z.norm(dim=1).mean())
        a = torch.from_numpy(rng.uniform(0.2, 2 * math.pi - 0.2, C)).float()
        if kind == "offband":
            b = torch.from_numpy(rng.uniform(0.2, 2 * math.pi - 0.2, C)).float()
        elif kind in ("absmag", "same"):                  # this seed's default-init angles: |.| of them, or as they are (signed)
            A0 = model.angles_init_abs if kind == "absmag" else model.angles_init
            a, b = A0[0].clone(), A0[1].clone()
        elif kind in ("low", "high", "mid"):              # iid a+, a-: low U(0.05, 1.5); mid U(0.9, 1.5) (sum < pi, large); high U(1.7, 4.5)
            lo, hi = {"low": (0.05, 1.5), "mid": (0.9, 1.5), "high": (1.7, 4.5)}[kind]
            a = torch.from_numpy(rng.uniform(lo, hi, C)).float(); b = torch.from_numpy(rng.uniform(lo, hi, C)).float()
        else:
            b = (2 * math.pi - a) if model.mode == "abs" else -a
        om = model.omega.abs().clamp_min(1.0) if clamp_omega else model.omega.abs(); model.omega.copy_(om)
        model.z.zero_(); model.z[0, 0] = c; model.z[1, 1] = c
        model.w_out.weight.zero_(); model.w_out.weight[:, 0] = a / om / c; model.w_out.weight[:, 1] = b / om / c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["signed", "abs"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", default="random", choices=["random", "band", "offband", "omega1", "low", "high", "absmag", "same", "mid"])
    ap.add_argument("--freeze-omega", action="store_true")
    ap.add_argument("--no-omega-clamp", action="store_true")
    ap.add_argument("--target", default="cell", choices=["cell", "obs"])
    ap.add_argument("--init-scale", type=float, default=1.0, help="multiply W_out at random init")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--p-plus", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    m = Toy(a.mode)
    if a.init_scale != 1.0:
        with torch.no_grad():
            m.w_out.weight.mul_(a.init_scale)
    if a.init == "omega1":                                  # random init, slow channels' omega clamped to >= 1 only
        with torch.no_grad():
            m.omega.copy_(m.omega.abs().clamp_min(1.0))
    elif a.init != "random":
        if a.init in ("absmag", "same"):                    # THIS seed's default-init angles, in the balanced gauge
            with torch.no_grad():
                m.angles_init = (m.w_out(m.z) * m.omega).detach(); m.angles_init_abs = m.angles_init.abs()
        set_init(m, a.init, rng, clamp_omega=not a.no_omega_clamp)
    if a.freeze_omega:
        m.omega.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    warm = max(1, a.steps // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (s - warm) / (a.steps - warm)))))
    ev_rng = np.random.RandomState(10 ** 6); emv, epos = walks(256, a.T, p_plus=a.p_plus, rng=ev_rng)
    eobs = obs_of(epos, ev_rng) if a.target == "obs" else None
    hist = []; t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.log_every == 0 or step == a.steps:
            with torch.no_grad():
                l, acc = loss_acc(m, emv, epos, eobs)
                acc_cell = loss_acc(m, emv, epos)[1] if eobs is not None else acc
            hist.append({"step": step, "loss": float(l), "acc": float(acc), "acc_cell": float(acc_cell),
                         "wrap_share": wrap_stats(m)})
            print(f"{step:6d} loss {float(l):.4f} acc {float(acc):.4f} wrap {hist[-1]['wrap_share']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if step == a.steps:
            break
        mv, pos = walks(a.batch, a.T, p_plus=a.p_plus, rng=rng)
        l, _ = loss_acc(m, mv, pos, obs_of(pos, rng) if a.target == "obs" else None)
        opt.zero_grad(); l.backward(); opt.step(); sch.step()
    A = m.angles().detach()
    res = {"args": vars(a), "hist": hist, "final_acc": hist[-1]["acc"], "final_acc_cell": hist[-1]["acc_cell"],
           "final_wrap_share": hist[-1]["wrap_share"],
           "angles": A.tolist(), "omega": m.omega.detach().tolist(), "w": F.softplus(m.u).detach().tolist(),
           "beta": float(torch.exp(m.log_beta))}
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
