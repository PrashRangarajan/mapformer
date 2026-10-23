"""Rule 14 existence for G-3 (G3_SSM_PREREG.md Sec. 4): build an exact solution in each non-negative step class from a
TRAINED signed SSM, check it frozen against the source, and save it for the trainable (held) continuation.

Constructions (MAIN_APPENDIX2.md E2 = Proposition 3; APPENDIX_theory_v3.md P1), per channel c and stepping token j, with
Theta = omega * Delta the source's signed angle table and rem = Theta mod 2 pi in [0, 2 pi):

  latents   W_in' sends move j to c0 u_j (u_j the j-th unit vector of R^r, c0 = the source's mean move-latent norm) and,
            when every token steps (steps = "all"), every observation embedding to 0 (min-norm solve against the full
            embedding matrix; exact because V <= d_model and the embeddings are linearly independent, checked).
            Rule 14's scale clause: the balanced construction of D3 keeps W_in near its trained scale.
  abs       (a)  omega'_c = max(|omega_c|, 1),  W_out'[c, j] = rem / omega'_c / c0                       steps >= 0
  softplus  (f)  omega'_c = 2 pi / ln 2,        W_out'[c, j] = softplus^-1((rem + 2 pi) / omega'_c) / c0   steps > 0
  exp       (g)  omega'_c = 2 pi,               W_out'[c, j] = ln((rem + 2 pi) / 2 pi) / c0                steps > 0
  signed    control: omega' = max(|omega|, 1), W_out' = Theta / omega' / c0 (the UNWRAPPED angle), so it differs from the
            abs construction only in the wrap (D3's signed_balanced). One control serves all three classes.

With steps = "all" an annihilated token has latent 0, so it steps |0| = 0 (abs, signed), softplus(0) = ln 2 (an angle of
exactly omega' ln 2 = 2 pi) or exp(0) = 1 (an angle of exactly 2 pi): the full-turn part of (f) and (g). The source's
observation angles (small, not exactly 0, in a trained signed "all" model) are replaced by 0 mod 2 pi, so in that mode the
construction is exact against the SIGNED CONTROL (same function mod 2 pi on every token) and approximate against the
source; both are reported. With steps = "moves" observation steps are 0 by construction and every construction equals the
source mod 2 pi.

  cd /home/prashr && python3 -m mapformer.construct_ssm_sign --source runs/.../signed.pt [--save DIR]
"""
import argparse
import copy
import json
import math
import os

import numpy as np
import torch

from mapformer.environment_steps import GridWorldSteps
from mapformer.model_ssm_sign import build
from mapformer.train_cancel import evaluate

CLASSES = ("abs", "softplus", "exp", "signed")
TWO_PI = 2 * math.pi


def softplus_inv(y):
    return y + torch.log(-torch.expm1(-y))


def load_source(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    c = b["config"]
    m = build(b["variant"], c["steps"], c["vocab_size"], c["grid_size"], action_ids=tuple(c["action_ids"]),
              mixer=c.get("mixer", "ssm"), decay=c.get("decay", "none"))
    m.load_state_dict(b["model_state_dict"])
    m.eval()
    return m, b


@torch.no_grad()
def construct(src, target, steps=None, balance=True):
    """A model of class `target` (CLASSES) whose per-token angles equal the source's mod 2 pi (see the module doc).

    steps: the built model's step mode (default the source's). steps = "all" from a "moves" source is D4's full-arm
    construction: the observation embeddings are annihilated, so a moves-only map code is reproduced EXACTLY by a model in
    which every token steps (|.|, signed: observations step 0; softplus, exp: a full turn).
    balance: rescale W_in -> lam W_in, W_out -> W_out / lam (W_out W_in, hence every step, unchanged) so that
    max|W_in| = max|W_out|. Rule 14's scale clause: a trained signed solution has nearly antiparallel move embeddings, so
    orthogonal latents need a large W_in (pilot s2183: 7.6 against the source's 0.50) and a small W_out."""
    assert src.mode == "signed", "the source must be a trained SIGNED model"
    assert target in CLASSES, target
    steps = steps or src.steps
    assert not (src.steps == "all" and steps == "moves"), "a moves-only model cannot reproduce observation steps"
    S = src.state_dict()
    dev = src.token_emb.weight.device
    dt = torch.float64
    E = src.token_emb.weight.detach().to(dt)                       # (V, d)
    V, d = E.shape
    r = src.action_to_lie.w_in.weight.shape[0]
    act = src.act.tolist()
    A = len(act)
    assert A <= r, (A, r)
    Theta = src.angle_table().detach().to(dt).reshape(V, -1)        # (V, C), C = H * N
    C = Theta.shape[1]
    lat = src.action_to_lie.w_in.weight.detach().to(dt) @ E[act].T  # (r, A)
    c0 = float(lat.norm(dim=0).mean())
    # target latents: moves -> c0 u_j; every other token -> 0 (only enforced where it is read: steps == "all")
    if steps == "all":
        Z = torch.zeros(r, V, dtype=dt)
        for j, a in enumerate(act):
            Z[j, a] = c0
        Et = E.T                                                    # (d, V)
        W_in = Z @ torch.linalg.pinv(Et)
        resid = float((W_in @ Et - Z).abs().max())
    else:
        Z = torch.zeros(r, A, dtype=dt)
        for j in range(A):
            Z[j, j] = c0
        Et = E[act].T                                               # (d, A)
        W_in = Z @ torch.linalg.pinv(Et)
        resid = float((W_in @ Et - Z).abs().max())
    om = src.path_integrator.omega.detach().to(dt).reshape(-1)      # (C,)
    rem = torch.remainder(Theta[act], TWO_PI)                       # (A, C) in [0, 2 pi)
    W_out = torch.zeros(C, r, dtype=dt)
    if target == "abs":
        omp = om.abs().clamp_min(1.0)
        W_out[:, :A] = (rem / omp).T / c0
    elif target == "signed":
        omp = om.abs().clamp_min(1.0)
        W_out[:, :A] = (Theta[act] / omp).T / c0
    elif target == "softplus":
        omp = torch.full((C,), TWO_PI / math.log(2.0), dtype=dt)
        W_out[:, :A] = softplus_inv((rem + TWO_PI) / omp).T / c0
    else:                                                           # exp
        omp = torch.full((C,), TWO_PI, dtype=dt)
        W_out[:, :A] = torch.log((rem + TWO_PI) / TWO_PI).T / c0
    lam = 1.0
    if balance:
        lam = math.sqrt(float(W_out.abs().max()) / float(W_in.abs().max()))
        W_in, W_out = W_in * lam, W_out / lam
    m = build(target, steps, V, src.grid_size, action_ids=tuple(act),
              mixer=src.mixer_kind, decay=src.decay).to(dev)
    S2 = copy.deepcopy(S)
    S2["action_to_lie.w_in.weight"] = W_in.to(S["action_to_lie.w_in.weight"].dtype)
    S2["action_to_lie.w_out.weight"] = W_out.to(S["action_to_lie.w_out.weight"].dtype)
    S2["path_integrator.omega"] = omp.reshape(src.path_integrator.omega.shape).to(S["path_integrator.omega"].dtype)
    m.load_state_dict(S2)
    m.eval()
    info = {"class": target, "steps": steps, "c0": c0, "balance_lambda": lam, "w_in_solve_resid": resid,
            "max_abs_w_in": float(W_in.abs().max()), "max_abs_w_out": float(W_out.abs().max()),
            "src_max_abs_w_in": float(S["action_to_lie.w_in.weight"].abs().max()),
            "src_max_abs_w_out": float(S["action_to_lie.w_out.weight"].abs().max()),
            "max_abs_omega": float(omp.abs().max()), "src_max_abs_omega": float(om.abs().max())}
    return m, info


def sensitivity(m):
    """Amendment 1 (MAJOR 2): move-angle sensitivity to an Adam-sized step. L1 norm of d angle / d parameter over the move
    embeddings' rows of token_emb, W_in, W_out and omega, for each of the A x C move angles; (mean, max) over angles. Adam
    moves each coordinate by at most about lr, so lr x this bounds one step's angle change (the audit's c3_built.py
    definition; pilot s2183: control 30, abs 116, softplus 191, exp 238; attention AO_bal 87-96, SO_bal 15-18)."""
    ps = [m.token_emb.weight, m.action_to_lie.w_in.weight, m.action_to_lie.w_out.weight, m.path_integrator.omega]
    with torch.enable_grad():
        A = m.angle_table()[m.act].reshape(-1)
        tot = torch.zeros(A.numel(), dtype=torch.float64)
        for i in range(A.numel()):
            g = torch.autograd.grad(A[i], ps, retain_graph=True, allow_unused=True)
            tot[i] = sum(float(x.abs().sum()) for x in g if x is not None)
    return float(tot.mean()), float(tot.max())


def wrap_err(a, b):
    """max |(a - b) wrapped to (-pi, pi]|"""
    return float(torch.angle(torch.exp(1j * (a.double() - b.double()))).abs().max())


@torch.no_grad()
def compare(src, m, ref=None, env=None, T=128, n_trials=400, device="cpu", tokens=None):
    """Angle error (every token, mod 2 pi) against `ref` (default the source), minimum step, logit difference on `tokens`,
    and frozen accuracy of both on the registered evaluation stream."""
    ref = ref if ref is not None else src
    out = {"angle_err_all_tokens": wrap_err(m.angle_table(), ref.angle_table()),
           "angle_err_moves": wrap_err(m.angle_table()[m.act], ref.angle_table()[ref.act]),
           "min_move_step": float(m.step_table()[m.act].min()),
           "min_step_all_tokens": float(m.step_table().min())}
    if m.steps == "all":
        obs = [i for i in range(m.vocab_size) if i not in m.act.tolist()]
        out["obs_angle_vs_0_mod_2pi"] = wrap_err(m.angle_table()[obs], torch.zeros_like(m.angle_table()[obs]))
        out["src_obs_angle_vs_0_mod_2pi"] = wrap_err(src.angle_table()[obs], torch.zeros_like(src.angle_table()[obs]))
    if tokens is not None:
        a, b = m(tokens), ref(tokens)
        out["max_logit_diff"] = float((a - b).abs().max())
        out["logit_scale"] = float(b.abs().max())
    if env is not None:
        out["frozen_acc"] = evaluate(m, env, T, n_trials, device, seed=10 ** 6)[0]
        out["ref_acc"] = evaluate(ref, env, T, n_trials, device, seed=10 ** 6)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--save", default=None, help="directory: writes <class>_built.pt for each class")
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--json", default=None)
    ap.add_argument("--steps", default=None, choices=("moves", "all"), help="built step mode (default the source's)")
    a = ap.parse_args()
    src, b = load_source(a.source)
    cfg = b["config"]
    env = GridWorldSteps(size=cfg["size"], seed=10000, p_first=0.5, deltas=tuple(cfg["deltas"]))
    np.random.seed(5)
    tok = torch.stack([env.generate_trajectory(cfg["n_steps"])[0][:-1] for _ in range(4)])
    built = {k: construct(src, k, steps=a.steps) for k in CLASSES}
    s_mean, s_max = sensitivity(src)
    rep = {"source": a.source, "source_steps": src.steps, "steps": a.steps or src.steps, "size": cfg["size"],
           "source_sensitivity_mean": s_mean, "source_sensitivity_max": s_max}
    for k, (m, info) in built.items():
        info["sensitivity_mean"], info["sensitivity_max"] = sensitivity(m)
        r_src = compare(src, m, env=env, T=cfg["n_steps"], n_trials=a.n_trials, tokens=tok)
        r_ctl = compare(src, m, ref=built["signed"][0], tokens=tok)
        rep[k] = {**info, "vs_source": r_src, "vs_signed_control": r_ctl}
        print(f"{k:9s} angle err vs source {r_src['angle_err_all_tokens']:.2e} (moves {r_src['angle_err_moves']:.2e})"
              f" | vs control {r_ctl['angle_err_all_tokens']:.2e} logits {r_ctl['max_logit_diff']:.2e}"
              f" | min move step {r_src['min_move_step']:+.4f} | frozen {r_src['frozen_acc']:.4f} vs source "
              f"{r_src['ref_acc']:.4f} | logit diff vs source {r_src['max_logit_diff']:.2e} (scale "
              f"{r_src['logit_scale']:.1f}) | max|W_in| {info['max_abs_w_in']:.3f} (src {info['src_max_abs_w_in']:.3f})"
              f" max|W_out| {info['max_abs_w_out']:.3f} (src {info['src_max_abs_w_out']:.3f}) max|omega| "
              f"{info['max_abs_omega']:.2f} (src {info['src_max_abs_omega']:.2f}) | sensitivity mean "
              f"{info['sensitivity_mean']:.1f} max {info['sensitivity_max']:.1f} (src {s_mean:.1f})", flush=True)
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            torch.save({"model_state_dict": m.state_dict(), "variant": k, "seed": b["seed"],
                        "config": dict(cfg, steps=info["steps"], built_from=os.path.abspath(a.source),
                                       construction={**info, **r_src}, vs_signed_control=r_ctl)},
                       os.path.join(a.save, f"{k}_built.pt"))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
