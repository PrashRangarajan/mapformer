"""SIGNSEP: rule 14 existence for every arm, and the numerical checks of which account each arm removes.

  cd /home/prashr && python3 -m mapformer.construct_signsep [--save DIR] [--json PATH]

EXISTENCE (rule 14: construct a solution in the class; frozen = source; the trainable test is a pilot warm start).
Source: a TRAINED SignedAO_r4 ring model on this recipe (LRSWEEP `runs/lrsweep/lr1e-03/SO_s1500/SO.pt`, acc 1.000; used
only as a source of a solution, never as a comparison arm, rule 12). Constructions:
  signed          the source itself.
  abs             `wrap_construct(mode='balanced')` (P1(a); WRAP_EXISTENCE: held 8/8).
  abs_wrap (W)    ON THE COLLECTIVE WRAP SHEET: latents and W_in kept, per channel omega' = sign(A_c)|omega_c| and
                  w'_c = sign(A_c omega_c) w_c (A_c = the source's +1 angle), tau = (0, 2). Then a'(x) = A_c and
                  a'(y) = B_c + 2 pi sign(A_c) = B_c mod 2 pi, every pre-|.| step >= 0, and EVERY channel wraps through the
                  one shared coordinate tau_y = 2 (not channel by channel).
  signed_frozen   `wrap_construct(mode='signed_balanced')` (orthogonal latents c e1, c e2, the form F's latents are
                  frozen in), then `freeze_move_latents()`. With --frozen-c-seed S (Amendment 1 finding 5) the latents
                  are rescaled to the norm c that F's frozen latents have at seed S's common start (0.42-1.60 against
                  the source's 4.42) and W_out by the inverse, so the solution sits at the weight scale F trains at.
  abs_wrap_past   same class as abs_wrap.

ACCOUNT CHECKS at the common start (`model_signsep.build`), pilot seeds 2380-2387 (init only, no training):
  (1) Barrier on the collective route. S's straight route z_y(t) = (1-t) z_y0 - t z_x0 gives k_c(t) = (1-t) k_c(0):
      monotone to 0, never crossing pi. W's route (the same latents plus tau_y = 2t) gives k_c(t) = (1-t) k_c(0) + 2 pi t:
      monotone to 2 pi, crossing pi in every channel that starts below pi, and staying in the region where the pre-|.|
      steps are >= 0 iff a_c(x)(0) <= 2 pi. Both computed THROUGH THE MODEL CODE at t = 0, 1/2, 1 and compared with the
      formula.
  (2) Collective reach. Share of the clock change to the class's collective target that the route coordinates can
      produce (least squares in the tangent space at the start; balanced gauge, so on the >= 0 region the clock is
      linear in the route coordinates): S (latents, target 0): 1 by S2(a); W (latents + tau, target 2 pi): 1; A
      (latents, its nearest non-blind target 2 pi): < 1 (S3; SEARCH_THEORY's latent_reach 0.34 at the default start).
      S's and W's 1.000 are IDENTITIES (k0 = J u0, and W's J contains the ones vector), and F's 0 is by definition
      (Amendment 1 finding 7); reach_A and the formula-vs-code error are the measurements.
  (3) Collective parameters. Number of trainable step-path scalars whose perturbation moves >= 2 channels' clocks
      (autograd Jacobian of the clock vector), over action_to_lie, path_integrator, tau AND the token embedding
      (Amendment 1 finding 6: the move-embedding rows feed the latent in S, A, W; counting them makes the F check able
      to catch a step path that read the live embedding).
  (4) The barrier operationally. The same-cell phase surrogate (mean over the task's revisit pairs and channels of
      1 - cos phase; 0 = every revisit's phase matches) along each route: slope at the start (negative = the route
      descends from the first step) and the largest rise above the start. W from the common start must climb before it
      descends; S descends; W_past (its own start, tau_sum 1.5 -> 2) descends.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from mapformer.environment_steps import GridWorldSteps
from mapformer.model_signsep import ARM_CLASS, ARMS, TAU_PAST, angles, build
from mapformer.model_wrap_exist import wrap_construct
from mapformer.train_cancel import evaluate

REPO = "/home/prashr/mapformer"
SOURCE = f"{REPO}/runs/lrsweep/lr1e-03/SO_s1500/SO.pt"
SIZE = 32
CHECK_SEEDS = list(range(2380, 2388))
EVAL_SEED = 10 ** 6


def _mk(arm, vocab, act):
    return ARM_CLASS[arm](vocab_size=vocab, d_model=128, n_heads=2, n_layers=1, grid_size=SIZE, action_ids=act)


def load_source(path=SOURCE):
    b = torch.load(path, map_location="cpu", weights_only=False)
    c = b["config"]
    act = (c["action_offset"], c["action_offset"] + 1)
    m = _mk("signed", c["vocab_size"], act)
    m.load_state_dict(b["model_state_dict"])
    return m.eval(), c["vocab_size"], act


def start_latent_norm(seed, vocab, act):
    """Mean move-latent norm c of F's frozen latents at `seed`'s common start."""
    m, _ = build("signed_frozen", vocab, act, seed=seed, grid_size=SIZE)
    return float(m.latents().double().norm(dim=1).mean())


@torch.no_grad()
def construct(arm, src, vocab, act, frozen_c=None):
    """A model of class `arm` with the source's content and its move angles mod 2 pi. `frozen_c`: for signed_frozen,
    rescale the (orthogonal) latents to this norm and W_out by the inverse (same angles)."""
    if arm == "signed":
        return src
    if arm == "abs":
        built, _ = wrap_construct(src, vocab, act, mode="balanced", d_model=128, n_heads=2, n_layers=1,
                                  grid_size=SIZE)
        m = _mk("abs", vocab, act); m.load_state_dict(built.state_dict()); return m.eval()
    if arm == "signed_frozen":
        built, _ = wrap_construct(src, vocab, act, mode="signed_balanced", d_model=128, n_heads=2, n_layers=1,
                                  grid_size=SIZE)
        m = _mk("signed", vocab, act); m.load_state_dict(built.state_dict())
        if frozen_c is not None:
            c_now = float(m.latents().double().norm(dim=1).mean())
            m.action_to_lie.w_in.weight.mul_(frozen_c / c_now)
            m.action_to_lie.w_out.weight.mul_(c_now / frozen_c)
        m.freeze_move_latents()
        return m.eval()
    assert arm in ("abs_wrap", "abs_wrap_past"), arm
    m = _mk(arm, vocab, act)
    sd = {k: v.detach().clone() for k, v in src.state_dict().items()}
    om = src.path_integrator.omega.detach().reshape(-1)
    A = angles(src)[0]                                                    # (C,) the +1 move's angles
    sA = torch.where(A >= 0, torch.ones_like(A), -torch.ones_like(A))
    sd["path_integrator.omega"] = (sA * om.abs()).reshape(src.path_integrator.omega.shape)
    sd["action_to_lie.w_out.weight"] = src.action_to_lie.w_out.weight.detach() * (sA * torch.sign(om))[:, None]
    sd["wrap_tau"] = torch.tensor([0.0, 2.0])
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def _logit_diff(a, b, vocab, T=128, n=8):
    te = GridWorldSteps(size=SIZE, seed=10000, p_first=0.5, deltas=(1, -1))
    np.random.seed(7)
    tok = torch.stack([te.generate_trajectory(T)[0] for _ in range(n)])
    return float((a(tok) - b(tok)).abs().max())


def existence(save_dir=None, n_trials=400, frozen_c_seed=None):
    src, vocab, act = load_source()
    te = GridWorldSteps(size=SIZE, seed=10000, p_first=0.5, deltas=(1, -1))
    src_acc = evaluate(src, te, 128, n_trials, "cpu", seed=EVAL_SEED)[0]
    a_src = angles(src).double()
    scale = lambda m: {"w_in_absmax": float(m.action_to_lie.w_in.weight.abs().max()),
                       "w_out_absmax": float(m.action_to_lie.w_out.weight.abs().max()),
                       "omega_absmax": float(m.path_integrator.omega.abs().max())}
    rep = {"source": SOURCE, "source_acc": src_acc, "source_scale": scale(src), "arms": {}}
    print(f"source {SOURCE}: acc T=128 {src_acc:.4f} scale {rep['source_scale']}")
    todo = [(a, None) for a in ARMS if a != "abs_wrap_past"]          # abs_wrap_past: same class as abs_wrap
    if frozen_c_seed is not None:
        fc = start_latent_norm(frozen_c_seed, vocab, act)
        rep["frozen_c_seed"], rep["frozen_c"] = frozen_c_seed, fc
        print(f"F at seed {frozen_c_seed}'s start scale: latent norm c = {fc:.4f}")
        todo.append(("signed_frozen", fc))
    for arm, fcv in todo:
        m = construct(arm, src, vocab, act, frozen_c=fcv)
        key = arm if fcv is None else f"{arm}_startc_s{frozen_c_seed}"
        a = angles(m).double()
        err = float(torch.atan2(torch.sin(a - a_src), torch.cos(a - a_src)).abs().max())
        acc = evaluate(m, te, 128, n_trials, "cpu", seed=EVAL_SEED)[0]
        r = {"frozen_acc": acc, "angle_err_max": err, "logit_diff_max": _logit_diff(m, src, vocab),
             "scale": scale(m), "sign_coherent_share": float(((a[0] * a[1]) >= 0).double().mean())}
        if hasattr(m, "pre_steps"):
            pre = m.pre_steps().double()
            k = a[0] + a[1]
            r.update({"pre_min": float(pre.min()), "tau": m.wrap_tau.tolist(),
                      "raw_clock_abs_min": float(k.abs().min()), "raw_clock_abs_max": float(k.abs().max()),
                      "share_wrapped": float((k.abs() > math.pi).double().mean())})
        if arm == "signed_frozen":
            r["w_in_requires_grad"] = bool(m.action_to_lie.w_in.weight.requires_grad)
            r["latent_norm"] = float(m.latents().double().norm(dim=1).mean())
        rep["arms"][key] = r
        print(f"  {key:24s} frozen acc {acc:.4f} (source {src_acc:.4f}) angle err {err:.2e} rad "
              f"logit diff {r['logit_diff_max']:.2e} " + json.dumps({k: v for k, v in r.items()
                                                                     if k not in ('frozen_acc',)}))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if arm in ("abs_wrap", "signed_frozen"):        # warm starts are run for the NEW arms only
                p = os.path.join(save_dir, f"{key}_construct.pt")
                torch.save({"model_state_dict": m.state_dict(), "variant": arm, "source": SOURCE,
                            "frozen_c": fcv}, p)
                rep["arms"][key]["saved"] = p
    return rep


# ------------------------------------------------------------------------------------------ account checks
def _route(m, t, kind, tau0=(0.0, 0.0), tau1=(0.0, 2.0)):
    """Angles (2, C) and pre-|.| steps along a straight route from the start, by the FORMULA. Latents: z_y(t) =
    (1-t) z_y0 - t z_x0 (S2(b)'s route). kind 'S': signed steps; 'W': |.| steps with tau moving tau0 -> tau1."""
    a2l = m.action_to_lie
    with torch.no_grad():
        ea = m.token_emb.weight[m.act].double()
        z = ea @ a2l.w_in.weight.double().T                          # (2, r)
        W = a2l.w_out.weight.double(); om = m.path_integrator.omega.double().reshape(-1)
        zx, zy = z[0], (1 - t) * z[1] - t * z[0]
        tau = torch.zeros(2, dtype=torch.float64)
        if kind == "W":
            tau = (1 - t) * torch.tensor(tau0, dtype=torch.float64) + t * torch.tensor(tau1, dtype=torch.float64)
        pre = torch.stack([W @ zx, W @ zy]) + tau[:, None] * (math.pi / om.abs())[None, :]
        return (pre.abs() if kind == "W" else pre) * om, pre


def revisit_counts(n_walks=200, T=128, seed=12345):
    """(P, 2) counts (n_+, n_-) of the moves between each revisit query and the most recent earlier visit of the same
    cell, over walks of the batch's task (GridWorldSteps, ring 32, +1/-1). Phase of the pair in channel c:
    n_+ a_c(+1) + n_- a_c(-1)."""
    env = GridWorldSteps(size=SIZE, seed=0, p_first=0.5, deltas=(1, -1))
    np.random.seed(seed)
    rows = []
    for _ in range(n_walks):
        tok = env.generate_trajectory(T)[0].numpy()
        acts = tok[0::2] - env.action_offset                          # 0 = +1, 1 = -1
        pos = np.cumsum(np.where(acts == 0, 1, -1)) % SIZE
        last = {}
        for i, p in enumerate(pos):
            if p in last:
                seg = acts[last[p] + 1:i + 1]
                rows.append(((seg == 0).sum(), (seg == 1).sum()))
            last[p] = i
    return np.asarray(rows, float)


def pair_surrogate(A, N):
    """Same-cell phase surrogate: mean over revisit pairs of mean_c (1 - cos phase_c); 0 = a perfect map for every
    revisit, ~1 = unrelated phases. A (2, C) angles, N (P, 2) counts."""
    phi = N @ A                                                       # (P, C)
    return float((1 - np.cos(phi)).mean())


def _route_through_code(m, t, kind):
    """The same point built in the model (W_in re-solved so the latents are the route's), angles via the code."""
    import copy
    mm = copy.deepcopy(m)
    with torch.no_grad():
        e = mm.token_emb.weight[mm.act].double(); a2l = mm.action_to_lie
        z = e @ a2l.w_in.weight.double().T
        Z = torch.stack([z[0], (1 - t) * z[1] - t * z[0]])
        a2l.w_in.weight.copy_((torch.linalg.pinv(e) @ Z).T.float())
        if kind == "W":
            mm.wrap_tau.copy_(torch.tensor([0.0, 2 * t]))
    return angles(mm).double()


def account_checks(seeds=CHECK_SEEDS):
    global REVISIT_N
    REVISIT_N = revisit_counts()
    print(f"revisit pairs for the surrogate: {len(REVISIT_N)}; median moves between visits {np.median(REVISIT_N.sum(1)):.0f}")
    env = GridWorldSteps(size=SIZE, seed=0, p_first=0.5, deltas=(1, -1))
    V, act = env.unified_vocab_size, (env.action_offset, env.action_offset + 1)
    ts = np.linspace(0, 1, 101)
    rows = []
    for s in seeds:
        mS, _ = build("signed", V, act, seed=s, grid_size=SIZE)
        mW, _ = build("abs_wrap", V, act, seed=s, grid_size=SIZE)
        mA, _ = build("abs", V, act, seed=s, grid_size=SIZE)
        mF, _ = build("signed_frozen", V, act, seed=s, grid_size=SIZE)
        mP, _ = build("abs_wrap_past", V, act, seed=s, grid_size=SIZE)
        r = {"seed": s}
        # (1) routes: formula vs code, clock profiles
        code_err = 0.0
        for kind, m in (("S", mS), ("W", mW)):
            for t in (0.0, 0.5, 1.0):
                f, _ = _route(m, t, kind)
                c = _route_through_code(m, t, kind)
                code_err = max(code_err, float(torch.atan2(torch.sin(f - c), torch.cos(f - c)).abs().max()))
        r["route_formula_vs_code_err"] = code_err
        K = {kind: np.stack([(_route(m, t, kind)[0].sum(0)).numpy() for t in ts]) for kind, m in (("S", mS), ("W", mW))}
        k0 = K["S"][0]
        r["S_clock_monotone_to_0"] = bool(np.all(np.abs(K["S"][1:]) <= np.abs(K["S"][:-1]) + 1e-9)
                                          and np.abs(K["S"][-1]).max() < 1e-6)
        r["S_route_max_abs_clock_over_start"] = float((np.abs(K["S"]).max(0) / np.maximum(np.abs(k0), 1e-12)).max())
        dW = np.abs(K["W"] - 2 * math.pi)                              # distance to the wrap target, per channel
        r["W_clock_monotone_to_2pi"] = bool(np.all(dW[1:] <= dW[:-1] + 1e-9) and dW[-1].max() < 1e-5)
        below = k0 < math.pi
        crossed = (K["W"].max(0) > math.pi)
        r["W_share_start_below_pi"] = float(below.mean())
        r["W_share_of_those_crossing_pi"] = float(crossed[below].mean())
        pre_min = min(float(_route(mW, t, "W")[1].min()) for t in ts)
        r["W_route_pre_min"] = pre_min
        r["W_route_share_channels_stay_nonneg"] = float(np.mean(
            np.min(np.stack([_route(mW, t, "W")[1].min(0).values.numpy() for t in ts]), 0) >= -1e-12))
        # (2) collective reach (least squares in the route-coordinate tangent space; >= 0 region, balanced gauge)
        with torch.no_grad():
            W = mS.action_to_lie.w_out.weight.double(); om = mS.path_integrator.omega.double().reshape(-1)
            J_lat = om[:, None] * W                                   # dk / du  (C, r)
            J_W = torch.cat([J_lat, math.pi * torch.ones(len(om), 1, dtype=torch.float64)], 1)   # + dk / dT
            k0t = torch.from_numpy(k0)

            def reach(J, dk):
                proj = J @ (torch.linalg.pinv(J) @ dk)                # orthogonal projection onto col(J)
                return float(proj.norm() / dk.norm())
            r["reach_S_to_0"] = reach(J_lat, -k0t)
            r["reach_W_to_2pi"] = reach(J_W, 2 * math.pi - k0t)
            r["reach_A_to_2pi"] = reach(J_lat, 2 * math.pi - k0t)
            r["reach_F"] = 0.0                                        # no trainable latent: dk/du does not exist
        # (3) collective parameters: trainable step-path scalars moving >= 2 channels' clocks
        for nm, m in (("S", mS), ("A", mA), ("W", mW), ("F", mF)):
            ps = [(n, p) for n, p in m.named_parameters() if p.requires_grad and
                  (n.startswith("action_to_lie") or n.startswith("path_integrator") or n == "wrap_tau"
                   or n == "token_emb.weight")]
            ncoll = 0
            for n, p in ps:
                Jt = torch.autograd.functional.jacobian(
                    lambda q: _clock_with(m, n, q), p.detach().clone())   # (C, *p.shape)
                nz = (Jt.reshape(Jt.shape[0], -1).abs() > 1e-12).sum(0)
                ncoll += int((nz >= 2).sum())
            r[f"collective_params_{nm}"] = ncoll
        # (4) the barrier operationally: the same-cell phase surrogate (revisit pairs of the task) along each route
        N = REVISIT_N
        prof = {kind: np.array([pair_surrogate(_route(m, t, kind)[0].numpy(), N) for t in ts])
                for kind, m in (("S", mS), ("W", mW))}
        tp0, tp1 = (TAU_PAST / 2, TAU_PAST / 2), (TAU_PAST / 2, 2 - TAU_PAST / 2)
        prof["P"] = np.array([pair_surrogate(_route(mP, t, "W", tp0, tp1)[0].numpy(), N) for t in ts])
        with torch.no_grad():
            ap = angles(mP).double().numpy()
        r["Wpast_route_start_matches_model"] = float(np.abs(np.angle(np.exp(1j * (
            _route(mP, 0.0, "W", tp0, tp1)[0].numpy() - ap)))).max())
        for kind, nm in (("S", "S"), ("W", "W"), ("P", "Wpast")):
            v = prof[kind]
            r[f"{nm}_pairsurr_start"] = float(v[0]); r[f"{nm}_pairsurr_end"] = float(v[-1])
            r[f"{nm}_pairsurr_slope0"] = float((v[1] - v[0]) / (ts[1] - ts[0]))
            r[f"{nm}_pairsurr_max_rise"] = float(v.max() - v[0])
            r[f"{nm}_pairsurr_argmax_t"] = float(ts[int(v.argmax())])
        r["Wpast_share_start_past_pi"] = float((ap.sum(0) > math.pi).mean())
        rows.append(r)
        print(json.dumps(r))
    return rows


def _clock_with(m, name, value):
    """Clock vector (C,) of model m with parameter `name` replaced by `value` (functional, for the Jacobian)."""
    from torch.func import functional_call
    params = {n: p for n, p in m.named_parameters()}
    params[name] = value
    bufs = {n: b for n, b in m.named_buffers()}

    class _C(torch.nn.Module):
        def __init__(s, mm):
            super().__init__(); s.mm = mm

        def forward(s):
            st = s.mm.step_table()[s.mm.act].reshape(len(s.mm.act), -1)
            return (st * s.mm.path_integrator.omega.reshape(-1)).sum(0)
    return functional_call(_C(m), {f"mm.{n}": v for n, v in {**params, **bufs}.items()}, ())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None, help="directory for the warm-start checkpoints of the new arms")
    ap.add_argument("--json", default=None)
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--skip-existence", action="store_true")
    ap.add_argument("--frozen-c-seed", type=int, default=None,
                    help="also build F at this seed's start latent scale (Amendment 1 finding 5)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = {}
    if not a.skip_existence:
        print("== existence (rule 14) ==")
        out["existence"] = existence(a.save, a.n_trials, a.frozen_c_seed)
    print("== account checks at the common start (pilot seeds, init only) ==")
    out["checks"] = account_checks()
    keys = [k for k in out["checks"][0] if k != "seed"]
    print("== summary over seeds (min / max) ==")
    for k in keys:
        v = [r[k] for r in out["checks"]]
        if isinstance(v[0], bool):
            print(f"  {k:40s} True at {sum(v)}/{len(v)} seeds")
        else:
            print(f"  {k:40s} {min(v):+.4g} / {max(v):+.4g}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
