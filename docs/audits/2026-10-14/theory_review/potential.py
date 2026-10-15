"""THEORY REVIEW (2026-10-14; CPU; not registered). The one-channel retrieval potential on the ring.

Model: one rotary channel, move angles a+ (for +1) and a- (for -1); phase difference between query move t and key move t'
phi = a+ * dn+ + a- * dn- (dn+- = number of +-1 moves in (t', t]). Same cell iff dn+ - dn- = 0 mod N. Retrieval loss at
inverse temperature beta: L = E_q[ lse_all(beta cos phi) - lse_same(beta cos phi) ] over revisit queries (toy_wrap's loss).
Small beta (near-uniform attention, as a freshly initialised transformer): L = L0 - beta * V + O(beta^2),
V(a+, a-) = E_q[ mean_same cos phi - mean_all cos phi ]. V is a 2D trigonometric polynomial in (a+, a-) with the pair
histogram as coefficients, so its value and gradient on a grid are matrix products.
Walk: GridWorldCancel / toy_wrap (ring 32, runs of U{1..10} moves, p_plus), T = 128.
Outputs: pair statistics (gap parity), V along the sum axis, gradient-flow endpoints from the MECHANISM starts, signed vs |.|.
"""
import math, sys
import numpy as np

N, T = 32, 128


def walks(n, T, size=N, p_plus=0.5, rng=None):
    mv = np.empty((n, T), np.int64); pos = np.empty((n, T), np.int64)
    for b in range(n):
        p = rng.randint(0, size); t = 0
        while t < T:
            a = 1 if rng.random() < p_plus else -1
            for _ in range(rng.randint(1, 11)):
                if t >= T:
                    break
                p = (p + a) % size; mv[b, t] = a; pos[b, t] = p; t += 1
    return mv, pos


def pair_hist(n=2000, p_plus=0.5, seed=0):
    """W[m, k] = E_q[ (1/#same) 1(same) - (1/#all) ] summed over pairs with dn+ = m, dn- = k (revisit queries only)."""
    rng = np.random.RandomState(seed)
    mv, pos = walks(n, T, p_plus=p_plus, rng=rng)
    Ws = np.zeros((T + 1, T + 1)); Wa = np.zeros((T + 1, T + 1)); nq = 0
    gaps_same = []
    for b in range(n):
        cp = np.concatenate([[0], np.cumsum(mv[b] == 1)]); cm = np.concatenate([[0], np.cumsum(mv[b] == -1)])
        for t in range(1, T):
            keys = np.arange(t)
            same = pos[b, keys] == pos[b, t]
            if not same.any():
                continue
            nq += 1
            dp = cp[t + 1] - cp[keys + 1]; dm = cm[t + 1] - cm[keys + 1]
            np.add.at(Ws, (dp[same], dm[same]), 1.0 / same.sum())
            np.add.at(Wa, (dp, dm), 1.0 / t)
            gaps_same.extend((t - keys[same]).tolist())
    return Ws / nq, Wa / nq, np.array(gaps_same)


def V_and_grad(W, ap, am):
    """V = sum W[m,k] cos(ap m + am k) on a grid; returns V, dV/dap, dV/dam (arrays over the ap x am grid)."""
    m = np.arange(W.shape[0])
    Ep = np.exp(1j * np.outer(ap, m)); Em = np.exp(1j * np.outer(am, m))
    Z = Ep @ W @ Em.T
    Zp = (Ep * (1j * m)) @ W @ Em.T; Zm = Ep @ W @ (Em * (1j * m)).T
    return Z.real, Zp.real, Zm.real


def Vpt(W, a):
    m = np.arange(W.shape[0])
    ep = np.exp(1j * a[0] * m); em = np.exp(1j * a[1] * m)
    return (ep @ W @ em).real, ((1j * m * ep) @ W @ em).real, (ep @ W @ (1j * m * em)).real


def flow(W, a0, mode, steps=4000, lr=0.005, adam=False):
    """Gradient ASCENT on V (= descent on the small-beta loss). mode 'signed': parameters are the angles; 'abs': parameters
    s with a = |s| (s initialised = a0 >= 0). adam: sign-like normalised steps (Adam with beta2 -> 1 early: lr * sign)."""
    s = np.array(a0, float)
    for _ in range(steps):
        a = np.abs(s) if mode == "abs" else s
        _, gp, gm = Vpt(W, a)
        g = np.array([gp, gm])
        if mode == "abs":
            g = g * np.where(s >= 0, 1.0, -1.0)
        s = s + (0.002 * np.sign(g) if adam else lr * g)
    return np.abs(s) if mode == "abs" else s


def wrapd(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


if __name__ == "__main__":
    out = []
    P = lambda *x: (print(*x), out.append(" ".join(str(y) for y in x)))
    Ws, Wa, gs = pair_hist()
    W = Ws - Wa
    P(f"revisit-pair gaps: all even? {bool((gs % 2 == 0).all())}; share of same-cell weight at gap 2: "
      f"{Ws[1,1]:.3f}, gap 4: {Ws[2,2]:.3f}; displacement-0 share of same weight {sum(Ws[k,k] for k in range(T+1)):.3f}")
    # V along the sum axis at fixed difference: sigma = (a+ + a-)/2, delta = (a+ - a-)/2
    P("V on the sum axis (delta fixed), the same-cell part Vs, the all-keys part Va, and V = Vs - Va:")
    for dl in (0.0, 0.1, 2 * math.pi / N):
        rows = []
        for S in np.linspace(0, 2 * math.pi, 25):
            a = np.array([S / 2 + dl, S / 2 - dl])
            vs = Vpt(Ws, a)[0]; va = Vpt(Wa, a)[0]
            rows.append(f"{S:4.2f}:{vs:+.3f}/{va:+.3f}/{vs - va:+.3f}")
        P(f" delta {dl:.3f}: " + "  ".join(rows))
    # sign of dV/dsum along the diagonal a+ = a- (delta = 0) and slightly off it: where is the divide?
    P("dV/d(sum) along a+ = a- + 2*delta (sum S; positive = flows up toward the wrap at S = 2 pi):")
    for dl in (0.02, 0.1, 0.3):
        sg = []
        for S in np.linspace(0.2, 2 * math.pi - 0.2, 36):
            a = np.array([S / 2 + dl, S / 2 - dl]); _, gp, gm = Vpt(W, a)
            sg.append("+" if gp + gm > 0 else "-")
        P(f" delta {dl}: " + "".join(sg) + "   (S from 0.2 to 2pi-0.2 in 36 steps)")
    # flows from the MECHANISM starts (single channel, independent per channel at small beta)
    rng = np.random.RandomState(1)
    starts = {
        "absmag (small, same dir: |N(0,0.6)|)": np.abs(rng.normal(0, 0.6, (40, 2))),
        "default signed (small, N(0,0.6))": rng.normal(0, 0.6, (40, 2)),
        "mid U(0.9,1.5)": rng.uniform(0.9, 1.5, (40, 2)),
        "high U(1.7,4.5)": rng.uniform(1.7, 4.5, (40, 2)),
    }
    for adam in (False, True):
        P(f"--- gradient flow on V ({'sign-normalised (Adam-like)' if adam else 'plain gradient'}); per start, 40 channels: "
          "share ending cancelled (|wrap(a+ + a-)| < 0.1 with |wrap(a+)| > 0.05), of those by wrap (|raw sum| > pi), and share ending at the wall/origin")
        for mode in ("signed", "abs"):
            for name, A0 in starts.items():
                if mode == "abs" and "default" in name:
                    A0 = np.abs(A0)
                if mode == "abs" and (A0 < 0).any():
                    continue
                ends = np.array([flow(W, a, mode, adam=adam) for a in A0])
                canc = (np.abs(wrapd(ends.sum(1))) < 0.1) & (np.abs(wrapd(ends[:, 0])) > 0.05)
                byw = (np.abs(ends.sum(1))[canc] > math.pi).mean() if canc.any() else float("nan")
                wall = (np.min(np.abs(ends), 1) < 0.05).mean()
                cross = (np.sign(ends[:, 0]) != np.sign(ends[:, 1])).mean()
                P(f" {mode:6s} {name:38s} cancelled {canc.mean():.2f}  by wrap {byw:.2f}  opposite-sign {cross:.2f}  "
                  f"at wall/origin {wall:.2f}  init sum>pi {np.mean(A0.sum(1) > math.pi):.2f}")
    open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w").write("\n".join(out) + "\n")
