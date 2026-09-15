"""Row-by-row check that `batch_fast` builds exactly what `AdditionWorld.encode` builds for the same operands."""
import numpy as np
import torch

from mapformer.environment_addition import AdditionWorld, batch_fast, PAD


def main():
    bad = 0; n_rows = 0
    for fmt in ("shared", "role"):
        for kw in (dict(dmax=30, random_start=True), dict(dmax=30), dict(n_digits=100), dict(dmax=5, random_start=True, pad_to=95)):
            env = AdditionWorld(fmt, max_pos=202 if "n_digits" not in kw or kw["n_digits"] <= 200 else 512)
            T, M, P, _, A, Bd, la, lb, start = batch_fast(env, 400, np.random.RandomState(7), return_operands=True, **kw)
            for i in range(T.shape[0]):
                a = [int(A[i, j]) for j in range(la[i] - 1, -1, -1)]; b = [int(Bd[i, j]) for j in range(lb[i] - 1, -1, -1)]
                t, m, p, s = env.encode(a, b, start=int(start[i]))
                L = len(t); n_rows += 1
                ok = (T[i, :L].tolist() == t and M[i, :L].numpy().tolist() == m.tolist() and P[i, :L].tolist() == p
                      and bool((T[i, L:] == PAD).all()) and not bool(M[i, L:].any()))
                bad += not ok
    # sampling distribution: digit-length marginals and MSB rule
    env = AdditionWorld("shared", max_pos=202)
    _, _, _, _, A, Bd, la, lb, _ = batch_fast(env, 20000, np.random.RandomState(1), dmax=30, return_operands=True)
    msb_zero = int(((la > 1) & (A[np.arange(len(la)), la - 1] == 0)).sum())
    print(f"rows checked {n_rows}, mismatches {bad}; length marginal range {np.bincount(la)[1:].min()}-{np.bincount(la)[1:].max()} (uniform 1..30 expected ~667); multi-digit operands with a zero MSB: {msb_zero}")
    print("PASS" if bad == 0 and msb_zero == 0 else "FAIL")


if __name__ == "__main__":
    main()
