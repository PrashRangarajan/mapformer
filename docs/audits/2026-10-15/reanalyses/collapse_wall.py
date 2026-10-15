"""Fix-14 R3 (post hoc, CPU, eval-only; PREDICTIONS.md): zero-collapsed move steps in every |.| / softplus arm, and, for the
ring observation-step-free arms, whether the collapsed entries started near zero, whether attention reads them, and which
way the training-loss gradient points at the end of training (the 'wall certificate' for P4).

Per (move, channel) entry: angle a = omega_c * Delta_{c,move} (Delta after the sign constraint). Collapsed: |a| <= 0.05 rad.
Also printed for continuity with fig_mechanism_out.txt: the share of channels with |wrap(a(+1))| <= 0.05.
Gradient: dL/dDelta of the training loss (cross-entropy at revisited observation positions, train.py's loss), model in
eval mode (no dropout), float64, on 256 trajectories of the ring eval stream (GridWorldCancel(32, seed 10000, p_plus 0.5),
np seed 10**6, T = 128). dL/dDelta > 0: the loss wants the step smaller, i.e. into the wall at 0.
Usage: model_wrap_exist.usage_weights (mean |q_c| at move positions x mean |k_c| at observation positions), same stream.
Check first (--test): autograd dL/dDelta equals a central finite difference on single entries.
  cd /home/prashr && OMP_NUM_THREADS=4 PYTHONPATH=/home/prashr python3 \
      mapformer/docs/audits/2026-10-15/reanalyses/collapse_wall.py [--test]"""
import json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.model_wrap_exist import (MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, action_angles, usage_weights)
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"; RUNS = f"{REPO}/runs"; HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.05
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def ring_ao(path, arm):
    b = torch.load(path, map_location="cpu", weights_only=False)
    cls = MapFormerWM_SignedAO_r4 if arm.startswith("SO") else MapFormerWM_AbsAO_r4
    off = b["config"].get("action_offset", 0)
    m = cls(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(off, off + 1))
    if "frozen_act_emb" in b["model_state_dict"]:
        m.freeze_move_latents()
    m.load_state_dict(b["model_state_dict"]); m.eval(); return m


def variant(path, V, grid):
    b = torch.load(path, map_location="cpu", weights_only=False); c = b["config"]
    m = VARIANT_MAP[V](vocab_size=c["vocab_size"], d_model=c.get("d_model", 128), n_heads=c.get("n_heads", 2),
                       n_layers=c.get("n_layers", 1), grid_size=c.get("grid_size", grid))
    m.load_state_dict(b["model_state_dict"]); m.eval(); return m


@torch.no_grad()
def move_angles(m, n_moves):
    """(n_moves, C) angles of the move tokens 0..n_moves-1 (observation-step-free models: their step table; full models:
    action_to_lie of the move embeddings), times omega."""
    if hasattr(m, "act"):
        return action_angles(m).double().numpy()
    e = m.token_emb.weight[:n_moves]
    D = m.action_to_lie(e[None])[0].reshape(n_moves, -1)
    return (D * m.path_integrator.omega.reshape(-1)).double().numpy()


def stream(n=256, T=128):
    from mapformer.environment_cancel import GridWorldCancel
    env = GridWorldCancel(size=32, seed=10000, p_plus=0.5); np.random.seed(10 ** 6)
    toks, revs = [], []
    for _ in range(n):
        t, _o, r = env.generate_trajectory(T); toks.append(t); revs.append(r)
    return torch.stack(toks), torch.stack(revs)


def loss_with_offset(m, tok, rev, off=None):
    """train.py's loss; `off` (V, H, nb) is added to the step table (for the finite-difference check)."""
    orig = type(m).step_table
    cache = {}

    def st(self=m):
        t = orig(self)
        if off is not None:
            t = t + off
        if t.requires_grad:
            t.retain_grad()
        cache["t"] = t
        return t
    m.step_table = st
    try:
        logits = m(tok[:, :-1]); tgt = tok[:, 1:]; mask = rev[:, 1:]
        L = F.cross_entropy(logits[mask], tgt[mask])
    finally:
        del m.step_table
    return L, cache


def grad_delta(m, tok, rev):
    m.zero_grad(set_to_none=True)
    L, cache = loss_with_offset(m, tok, rev)
    L.backward()
    return cache["t"].grad[m.act].reshape(len(m.act), -1).double().numpy(), float(L.detach())


def test():
    """Finite-difference check of dL/dDelta on a trained |.| checkpoint and a signed one (float64)."""
    tok, rev = stream(32)
    ok = True
    for arm in ("AO_absmag", "SO"):
        m = ring_ao(f"{RUNS}/wrap_mech/p0/{arm}_s800/{arm}.pt", arm).double()
        G, _ = grad_delta(m, tok, rev)
        A = move_angles(m, 2)
        idx = [(0, int(np.argmin(np.abs(A[0])))), (1, int(np.argmax(np.abs(A[1])))), (0, 5), (1, 17), (0, 40)]
        V, H, nb = m.step_table().shape
        for mv, c in idx:
            h, j = divmod(c, nb); h_eps = 1e-5
            with torch.no_grad():
                off = torch.zeros(V, H, nb, dtype=torch.float64); off[m.act[mv], h, j] = h_eps
                Lp, _ = loss_with_offset(m, tok, rev, off); Lm, _ = loss_with_offset(m, tok, rev, -off)
            fd = float(Lp - Lm) / (2 * h_eps); g = G[mv, c]
            rel = abs(fd - g) / max(abs(g), 1e-9)
            good = rel < 1e-3 or abs(fd - g) < 1e-9
            ok &= good
            print(f"  {arm} move {mv} channel {c}: autograd {g:+.6e} finite diff {fd:+.6e} rel err {rel:.1e} {'ok' if good else 'BAD'}")
    print("TEST", "PASS" if ok else "FAIL"); return ok


def part_a():
    """Collapse fractions in every |.| / softplus arm and its signed control (weights only)."""
    rows = []
    specs = []
    for a, cls in (("Signed_r4", "Signed_r4"), ("Abs_r4", "Abs_r4"), ("Pos_r4", "Pos_r4")):
        specs.append(("SIGN_MATCHED torus", a, 4, lambda s, a=a, cls=cls: variant(f"{RUNS}/sign_matched/p0/{a}_s{s}/{a}.pt", cls, 64),
                      range(8)))
    for p in (0.5, 0.75, 0.9):
        for a in ("Signed_r4", "Abs_r4"):
            specs.append((f"RING_DIAL p{p}", a, 2, lambda s, a=a, p=p: variant(f"{RUNS}/ring_dial/p0/{a}_p{p}_s{s}/{a}.pt", a, 32),
                          range(30, 38)))
    specs.append(("RING_BUDGET", "Abs_r4_e900", 2,
                  lambda s: variant(f"{RUNS}/ring_budget/p0/Abs_r4_p0.5_e900_s{s}/Abs_r4.pt", "Abs_r4", 32), range(40, 48)))
    for a in ("SO", "AO"):
        specs.append(("WRAP_EXISTENCE ring", a, 2, lambda s, a=a: ring_ao(f"{RUNS}/wrap_existence/p0/{a}_s{s}/{a}.pt", a),
                      range(600, 608)))
    for a in ("SO", "SO_absmag", "SO_high", "AO", "AO_absmag", "AO_mid", "AO_high"):
        specs.append(("MECHANISM ring", a, 2, lambda s, a=a: ring_ao(f"{RUNS}/wrap_mech/p0/{a}_s{s}/{a}.pt", a), range(800, 808)))
    for a, V in (("SO4", "SignedAO4_r4"), ("AO4", "AbsAO4_r4")):
        specs.append(("GENERALITY torus", a, 4,
                      lambda s, a=a, V=V: variant(f"{RUNS}/generality/torus/{a}/p0/{V}_s{s}/{V}.pt", V, 64), range(700, 708)))
    for a, V in (("SO4", "SignedAO4_r4"), ("AO4", "AbsAO4_r4"), ("SO4_high", "SignedAO4_r4"), ("AO4_high", "AbsAO4_r4"),
                 ("AO4_mid", "AbsAO4_r4")):
        specs.append(("TORUS_START torus", a, 4, lambda s, a=a, V=V: variant(f"{RUNS}/torus_start/p0/{a}_s{s}/{a}.pt", V, 64),
                      range(1200, 1208)))
    out = {}
    for batch, arm, nm, f, seeds in specs:
        col, colp, per = [], [], []
        for s in seeds:
            A = move_angles(f(s), nm)                                       # (moves, C)
            col.append(np.abs(A) <= EPS); colp.append(np.abs(wrap(A[0])) <= EPS); per.append((np.abs(A) <= EPS).mean())
        col = np.stack(col); colp = np.stack(colp)
        out[batch, arm] = float(col.mean())
        rows.append(f"| {batch} | {arm} | {col.mean():.3f} ({min(per):.2f}-{max(per):.2f}) | {colp.mean():.3f} |")
    return ["| batch | arm | collapsed (move, channel) entries, |a| <= 0.05: mean (run range) | channels with |wrap(a(+1))| <= 0.05 |",
            "|---|---|---|---|"] + rows, out


def part_b():
    """Ring observation-step-free arms (MECHANISM, WRAP_EXISTENCE): start, usage, gradient of the collapsed entries."""
    tok, rev = stream(256)
    lines = ["| batch | arm | collapsed entries (of all) | started near 0 (init |a| <= 0.05) / median init |a|, collapsed vs not | "
             "median usage, collapsed / not | dL/dDelta > 0 among collapsed | median |dL/dDelta|, collapsed / not | reading |",
             "|---|---|---|---|---|---|---|---|"]
    for batch, root, arms, seeds in (("MECHANISM ring", f"{RUNS}/wrap_mech/p0",
                                      ("AO", "AO_absmag", "AO_mid", "AO_high", "SO", "SO_absmag"), range(800, 808)),
                                     ("WRAP_EXISTENCE ring", f"{RUNS}/wrap_existence/p0", ("AO", "SO"), range(600, 608))):
        for arm in arms:
            C, I, U, G, has_init = [], [], [], [], True
            for s in seeds:
                d = f"{root}/{arm}_s{s}"; m = ring_ao(f"{d}/{arm}.pt", arm).double()
                A = move_angles(m, 2); ev = json.load(open(f"{d}/eval.json"))
                if "init_angles" in ev:
                    I.append(np.abs(np.array(ev["init_angles"])))
                else:
                    has_init = False
                u = usage_weights(m, tok, action_offset=int(m.act[0]), n_actions=2).double().numpy()
                g, _ = grad_delta(m, tok, rev)
                C.append(np.abs(A) <= EPS); U.append(np.broadcast_to(u, A.shape)); G.append(g)
            C, U, G = map(np.stack, (C, U, G))
            c, nc = C, ~C
            if c.sum() == 0:
                lines.append(f"| {batch} | {arm} | 0 | -- | -- | -- | -- | -- |"); continue
            uc, un = np.median(U[c]), np.median(U[nc]); gc, gn = np.median(np.abs(G[c])), np.median(np.abs(G[nc]))
            pos = G[c] > 0; strong = pos & (np.abs(G[c]) >= 0.1 * gn)
            if strong.mean() >= 2 / 3 and uc >= 0.5 * un:
                read = "WALL"
            elif uc < 0.2 * un:
                read = "UNUSED"
            else:
                read = "MIXED"
            if has_init:
                I = np.stack(I); z0 = I <= EPS
                init = (f"{(I[c] <= EPS).mean():.2f} / {np.median(I[c]):.2f} vs {np.median(I[nc]):.2f} rad (base rate: {z0.mean():.2f} "
                        f"of all entries start near 0; of those, {c[z0].mean():.2f} end collapsed)")
            else:
                init = "not stored"
            lines.append(f"| {batch} | {arm} | {int(c.sum())} of {c.size} ({c.mean():.3f}) | {init} | {uc:.3g} / {un:.3g} "
                         f"(ratio {uc / un:.2f}) | {pos.mean():.2f} (of which |g| >= 0.1 x non-collapsed median: {strong.mean():.2f}) | "
                         f"{gc:.2e} / {gn:.2e} | {read} |")
    return lines


def main():
    torch.set_num_threads(4)
    if not test():
        sys.exit(1)
    if "--test" in sys.argv:
        return
    a_lines, _ = part_a()
    b_lines = part_b()
    txt = "\n".join(["# R3: collapsed steps and the wall (post hoc; PREDICTIONS.md)", "", "## (a) Collapse fractions", ""] + a_lines
                    + ["", "## (b) Ring observation-step-free arms: start, usage, gradient at the end", ""] + b_lines)
    print(txt); open(f"{HERE}/collapse_wall_out.md", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
