"""Dyck-2 valid continuations (MapFormer v4, Sec 4.1 / 5.3 / App. B.4).

Paper, verbatim where it specifies:
- two bracket types, () and [];
- "we generate sequences by first fixing a target length L and maximum depth D that must be
  reached ... we modify the sampling distribution at each step to ensure that depth D is
  reached, and that the stack comes back to depth 0 at the end of the sequence";
- the metric is the F1 valid-continuation score of Goodale et al. (2025):
      P_Val(s) = sum_{x in Val(s)} P(x|s)
      BT(s)    = #{x in Val(s) : P(x|s) > sum_{c not in Val(s)} P(c|s)} / |Val(s)|
      F1(s)    = 2 P_Val(s) BT(s) / (P_Val(s) + BT(s))
- Val(s) = {'(', '['} plus the closer of the last opened bracket when depth > 0.

NOT specified by the paper, chosen here and recorded in DYCK_PREREG.md:
- the exact "modification": a step is uniform over {open, close} when both keep the sequence
  feasible (depth <= D, D reached, back to 0 at L), forced otherwise; the opened type is uniform;
- a BOS token starts every sequence (Fig. 3f: "starting in symbol x");
- F1 is averaged over the L prefixes s_<1 .. s_<L of each sequence (BOS alone included).
"""
import numpy as np
import torch

OPEN_P, CLOSE_P, OPEN_B, CLOSE_B, BOS = 0, 1, 2, 3, 4
VOCAB = 5
CLOSER = np.array([CLOSE_P, CLOSE_P, CLOSE_B, CLOSE_B, -1])   # closer of an OPEN token


def _feasible(d_after, rem, reached_after, D):
    # can still reach D (if not yet reached) and return to 0 within rem tokens
    need = np.where(reached_after, d_after, 2 * D - d_after)
    return (d_after >= 0) & (d_after <= D) & (rem >= need)


class DyckWorld:
    vocab_size = VOCAB

    def sample(self, n, L, D, rng):
        """Returns tokens (n, L) int64, valid (n, L, VOCAB) bool for each prefix s_<t,
        and ent (n, L) the sampler's next-token entropy in nats (the CE floor)."""
        assert L % 2 == 0 and L >= 2 * D, (L, D)
        tok = np.zeros((n, L), np.int64)
        valid = np.zeros((n, L, VOCAB), bool)
        ent = np.zeros((n, L))
        stack = np.zeros((n, D + 1), np.int64)     # opened types, stack[:, depth-1] is the top
        d = np.zeros(n, np.int64)
        reached = np.zeros(n, bool)
        ar = np.arange(n)
        for t in range(L):
            rem = L - t - 1                        # tokens left after this one
            top = stack[ar, np.maximum(d - 1, 0)]
            valid[:, t, OPEN_P] = True; valid[:, t, OPEN_B] = True
            valid[ar[d > 0], t, CLOSER[top[d > 0]]] = True
            can_open = _feasible(d + 1, rem, reached | (d + 1 == D), D)
            can_close = (d > 0) & _feasible(d - 1, rem, reached, D)
            assert np.all(can_open | can_close), "sampler reached an infeasible state"
            both = can_open & can_close
            do_open = np.where(both, rng.random(n) < 0.5, can_open)
            typ = np.where(rng.random(n) < 0.5, OPEN_P, OPEN_B)
            ent[:, t] = np.where(both, 1.5 * np.log(2), np.where(can_open, np.log(2), 0.0))
            closer = CLOSER[top]
            tok[:, t] = np.where(do_open, typ, closer)
            stack[ar[do_open], d[do_open]] = typ[do_open]
            d = d + np.where(do_open, 1, -1)
            reached |= d == D
        assert np.all(d == 0) and np.all(reached)
        return tok, valid, ent

    def batch(self, n, L, D, rng):
        """Model input [BOS, s_1..s_{L-1}], targets s_1..s_L, valid set per input position."""
        tok, valid, ent = self.sample(n, L, D, rng)
        inp = np.concatenate([np.full((n, 1), BOS), tok[:, :-1]], 1)
        return (torch.from_numpy(inp), torch.from_numpy(tok),
                torch.from_numpy(valid), torch.from_numpy(ent))


def f1_valid(probs, valid):
    """probs (..., V) next-token distribution, valid (..., V) bool. Returns per-prefix
    (F1, P_Val, BT) tensors of shape (...)."""
    probs = probs.double()
    pv = (probs * valid).sum(-1)
    inv = (probs * ~valid).sum(-1)
    bt = ((probs > inv.unsqueeze(-1)) & valid).sum(-1).double() / valid.sum(-1).double()
    den = pv + bt
    f1 = torch.where(den > 0, 2 * pv * bt / den.clamp_min(1e-300), torch.zeros_like(den))
    return f1, pv, bt


def check_dyck(tok, L, D):
    """Independent (loop) validity check: well nested, returns to 0, max depth exactly D."""
    for row in tok:
        st, mx = [], 0
        for x in row:
            if x in (OPEN_P, OPEN_B):
                st.append(x); mx = max(mx, len(st))
            else:
                if not st or CLOSER[st[-1]] != x:
                    return False
                st.pop()
        if st or mx != D or len(row) != L:
            return False
    return True
