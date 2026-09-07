"""Regenerate every numerical magnitude quoted in mapformer_math.tex.

WHY THIS EXISTS. An audit found that seven verification figures in the note --
"checked to 9e-7", "verified numerically to 1e-11", "wrong by four orders of
magnitude", the closure perturbation table, "interval-invariant to 7e-16" -- had
no script behind them anywhere in the repository. They were true (they reproduce
below) but unreproducible, which by this project's own standards makes them
unciteable. `verify_inclusion.py` already covers the inclusion construction; this
covers the rest.

    cd /home/prashr && python3 -m mapformer.verify_theory
"""
import math

import numpy as np
import torch

from mapformer.model import MapFormerWM


def hdr(t):
    print(f"\n=== {t} " + "=" * max(0, 58 - len(t)))


def factorisation():
    """theta = omega (*) cumsum(W_out W_in x) == diag(omega) W_out cumsum(W_in x).

    cumsum is linear, so it commutes with W_out: all H*n_b phase channels are
    readouts of ONE rank-r accumulator. Quoted in the note as eq:factor.
    """
    hdr("factorisation identity (eq:factor)")
    torch.manual_seed(0)
    m = MapFormerWM(vocab_size=21).eval()
    tok = torch.randint(0, 21, (4, 128))
    with torch.no_grad():
        x = m.token_emb(tok)
        cos_a, sin_a = m.path_integrator(m.action_to_lie(x))
        # right-hand side: accumulate in the r-dimensional latent FIRST
        lat = torch.cumsum(m.action_to_lie.w_in(x), dim=1)          # (B,T,r)
        B, T, _ = x.shape
        d2 = m.action_to_lie.w_out(lat).view(B, T, m.n_heads, m.n_blocks)
        ang = (d2 * m.path_integrator.omega).transpose(1, 2)
        dc = (cos_a - torch.cos(ang)).abs().max().item()
        ds = (sin_a - torch.sin(ang)).abs().max().item()
    print(f"  max|d cos| = {dc:.3e}   max|d sin| = {ds:.3e}   (float32)")
    print(f"  rank of the accumulator: {m.action_to_lie.w_in.out_features} "
          f"of {m.n_heads * m.n_blocks} phase channels")
    return max(dc, ds)


def log_polar():
    """eq:unified: the conjugate on S_t is load-bearing.

    a_ts = Re[conj(q~_t) k~_s] must depend on the interval (t,s] only. With
    -conj(S_t) the phases SUBTRACT and the magnitudes ADD; with -S_t the phases
    add instead and the prefix leaks.
    """
    hdr("log-polar accumulator (eq:unified)")
    rng = np.random.default_rng(0)
    T, C = 12, 5
    cq, ck = rng.normal(size=(T, C)) + 1j * rng.normal(size=(T, C)), \
             rng.normal(size=(T, C)) + 1j * rng.normal(size=(T, C))
    G = rng.normal(size=(T, C)) * .3 + 1j * rng.normal(size=(T, C))
    S = np.cumsum(G, axis=0)

    def score(t, s, conj):
        q = np.exp(cq[t] - (np.conj(S[t]) if conj else S[t]))
        k = np.exp(ck[s] + S[s])
        return np.real(np.conj(q) * k).sum()

    def closed(t, s):
        g = np.real(cq[t]) + np.real(ck[s]) + np.real(S[s] - S[t])
        ph = np.imag(ck[s]) - np.imag(cq[t]) + np.imag(S[s] - S[t])
        return (np.exp(g) * np.cos(ph)).sum()

    e_ok = max(abs(score(t, s, True) - closed(t, s))
               for t in range(T) for s in range(t, T))
    e_bad = max(abs(score(t, s, False) - closed(t, s))
                for t in range(T) for s in range(t, T))
    print(f"  with -conj(S_t):  max error vs the closed form = {e_ok:.3e}")
    print(f"  with -S_t     :  max error vs the closed form = {e_bad:.3e}"
          f"   ({e_bad / max(e_ok, 1e-300):.1e}x worse)")
    return e_ok, e_bad


def sign_asymmetry():
    """Which of Re G, Im G must be ALIGNED on q and k for interval-relativity.

    Work on q~ and k~ AS WRITTEN, not on a hand-rolled formula -- a first version
    of this function parameterised the two branches inconsistently and reported
    Re G aligned as the good case, the opposite of the truth.

    q~_t = exp(c^q - conj(S_t)) puts -Re S_t and +Im S_t on the query;
    k~_s = exp(c^k + S_s)       puts +Re S_s and +Im S_s on the key.
    So AS WRITTEN the imaginary parts are ALIGNED (both +) and the real parts are
    OPPOSED. The Hermitian form conjugates the query, which turns the aligned
    phases into a difference and the opposed magnitudes into a sum-of-differences.
    Flipping either choice makes the score depend on the prefix u <= t.
    """
    hdr("sign asymmetry (the four-row table)")
    rng = np.random.default_rng(1)
    T, C = 10, 4
    cq = rng.normal(size=(T, C)) + 1j * rng.normal(size=(T, C))
    ck = rng.normal(size=(T, C)) + 1j * rng.normal(size=(T, C))
    G = rng.normal(size=(T, C)) * .2 + 1j * rng.normal(size=(T, C))

    def scores(G_, re_opposed, im_aligned):
        S = np.cumsum(G_, axis=0)
        # what the query carries, per the two independent sign choices
        qs = (-np.real(S) if re_opposed else np.real(S)) \
             + 1j * (np.imag(S) if im_aligned else -np.imag(S))
        out = []
        for t in range(T):
            q = np.exp(cq[t] + qs[t])
            for s in range(t, T):
                k = np.exp(ck[s] + S[s])
                out.append(np.real(np.conj(q) * k).sum())
        return np.array(out)

    G2 = G.copy(); G2[0] += 0.7 + 0.7j          # perturb the PREFIX only
    for re_opp in (True, False):
        for im_al in (True, False):
            a, b = scores(G, re_opp, im_al), scores(G2, re_opp, im_al)
            rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
            print(f"  Re G {'opposed' if re_opp else 'aligned':7s} | "
                  f"Im G {'aligned' if im_al else 'opposed':7s} -> "
                  f"relative prefix leak {rel:.3e}")
    print("  (only Re OPPOSED + Im ALIGNED is interval-relative)")


def shear():
    """The unipotent slot: q^T M_t^-1 M_s k is interval-relative and commutes."""
    hdr("shear / unipotent slot")
    rng = np.random.default_rng(2)
    b = rng.normal(size=16)

    def M(t):
        A = np.eye(2); A[0, 1] = b[:t].sum(); return A

    q, k = rng.normal(size=2), rng.normal(size=2)
    t, s = 3, 9
    lhs = q @ np.linalg.inv(M(t)) @ M(s) @ k
    rhs = q[0] * k[0] + q[1] * k[1] + b[t:s].sum() * q[0] * k[1]
    comm = np.abs(M(3) @ M(5) - M(5) @ M(3)).max()
    print(f"  |score - closed form|   = {abs(lhs - rhs):.3e}")
    print(f"  |[M_3, M_5]|            = {comm:.3e}   (commutes exactly)")
    return abs(lhs - rhs)


def mapem_lift():
    """MapEM's Hadamard product IS a single bilinear form -- on the tensor product.

    (A_X . A_P)_ts = (q^x_t (x) q^p_t) . (k^x_s (x) k^p_s), dimension d_x*d_p.
    So MapEM does not leave the frame of eq:unified; it evaluates it on a lifted
    space. An earlier draft listed it under "breaks the single bilinear form".
    """
    hdr("MapEM = the frame lifted to the tensor product")
    torch.manual_seed(0)
    T, dx, dp = 6, 8, 5
    qx, kx = torch.randn(T, dx), torch.randn(T, dx)
    qp, kp = torch.randn(T, dp), torch.randn(T, dp)
    had = (qx @ kx.T) * (qp @ kp.T)
    Q = torch.einsum("ti,tj->tij", qx, qp).reshape(T, dx * dp)
    K = torch.einsum("si,sj->sij", kx, kp).reshape(T, dx * dp)
    e = float((had - Q @ K.T).abs().max())
    print(f"  max|Hadamard - tensor-product bilinear| = {e:.3e}   (float32)")
    print(f"  lifted dimension d_x*d_p = {dx * dp}, against {dx} and {dp} separately")
    return e


def tem_conjugation():
    """TEM-t's logit is interval-relative iff the transition group is ABELIAN and
    there is no nonlinearity.

    logit = e_t^T W_e^T W_e e_s. With e_s = W_{t->s} e_t and e_t = W_{0->t} e_0 and
    W_e orthogonal, this is e_0^T W_{0->t}^T W_{t->s} W_{0->t} e_0 -- the interval
    operator CONJUGATED BY THE PREFIX. Conjugation is trivial exactly when the group
    commutes. MapFormer is the surviving configuration: SO(2)^n_b, no nonlinearity.
    """
    hdr("TEM-t: interval-relative iff abelian and linear")
    torch.manual_seed(0)
    d = 8

    def blocks(angles):
        M = torch.zeros(d, d)
        for i, a in enumerate(angles):
            c, s_ = math.cos(a), math.sin(a)
            M[2*i, 2*i] = c; M[2*i, 2*i+1] = -s_
            M[2*i+1, 2*i] = s_; M[2*i+1, 2*i+1] = c
        return M

    Wc = torch.stack([blocks([.3*k, .7*k, 1.1*k, .2*k]) for k in (1, -1, 2, -2)])
    S = torch.randn(4, d, d); S = S - S.transpose(1, 2)
    Wg = torch.matrix_exp(0.6 * S)
    e0 = torch.randn(d)

    def logit(prefix, suffix, W, relu):
        e = e0.clone()
        for a in prefix:
            e = e @ W[a]
            if relu:
                e = torch.relu(e)
        et = e.clone()
        for a in suffix:
            e = e @ W[a]
            if relu:
                e = torch.relu(e)
        return float(et @ e)

    print(f"  {'transition group':24s} {'prefix A':>11s} {'prefix B':>11s}  interval-relative")
    for W, lab, relu in ((Wc, "abelian, linear", False),
                         (Wg, "general SO(d), linear", False),
                         (Wc, "abelian, ReLU", True),
                         (Wg, "general SO(d), ReLU", True)):
        a, b = logit([0, 1], [1, 3], W, relu), logit([2, 3, 2], [1, 3], W, relu)
        print(f"  {lab:24s} {a:11.6f} {b:11.6f}  {abs(a - b) < 1e-4}")
    print(f"  ||[W_0,W_1]||: abelian {float((Wc[0]@Wc[1]-Wc[1]@Wc[0]).abs().max()):.1e}, "
          f"general {float((Wg[0]@Wg[1]-Wg[1]@Wg[0]).abs().max()):.3f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    factorisation(); log_polar(); sign_asymmetry(); shear()
    mapem_lift(); tem_conjugation()
    print("\nQuote these magnitudes, not remembered ones.")
