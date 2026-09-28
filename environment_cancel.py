"""1D ring with a biased directed walk: the "cancellation knob" for H3 (CANCEL_PREREG.md).

A ring of `size` cells, the observation map drawn as in GridWorldND (p_empty blanking, K types).
The walk is GridWorldND's directed walk at dims=1 -- draw an action, repeat it k ~ U{1..10}
times -- except that the action is +1 with probability `p_plus` and -1 otherwise.

  p_plus = 0.5  increments cancel in expectation; position is a genuine random walk.
  p_plus = 1.0  every step is +1; position is (start + t) mod size, a clock. The action token
                is constant, so the path integrator's accumulated phase is a learned multiple of
                the token index.
In between, cancellation becomes rarer.

The observation map is REDRAWN for every trajectory (from the global numpy RNG, which the data
workers seed): a 32-cell ring with a fixed map is memorised within 5 epochs (loss 0.01), after
which the model has no reason to retrieve in context. With a fresh map an observation is knowable
only from an earlier visit in the same sequence, as in Match-Query. A SEPARATE module (rule 22: environment_nd.py and
train_variant.py are imported by running batches and are not edited).
"""
import numpy as np
import torch

from mapformer.environment_nd import GridWorldND


class GridWorldCancel(GridWorldND):
    def __init__(self, size: int = 32, n_obs_types: int = 16, p_empty: float = 0.5,
                 seed=None, p_plus: float = 0.5):
        super().__init__(dims=1, size=size, n_obs_types=n_obs_types, p_empty=p_empty, seed=seed)
        assert 0.0 <= p_plus <= 1.0, p_plus
        self.p_plus = p_plus

    def generate_trajectory(self, n_steps: int = 128, start=None):
        pos = int(start[0]) if start is not None else int(np.random.randint(0, self.size))
        occ = np.random.random(self.size) >= self.p_empty
        obs = np.full(self.size, self.blank_token, dtype=np.int64)
        obs[occ] = np.random.randint(0, self.n_obs_types, int(occ.sum()))
        tokens, is_revisit, seen = [], [], set()
        self.visited_locations = []
        t = 0
        while t < n_steps:
            a = 0 if np.random.random() < self.p_plus else 1        # 0 -> +1, 1 -> -1
            k = np.random.randint(1, 11)
            for _ in range(k):
                if t >= n_steps:
                    break
                pos = (pos + (1 if a == 0 else -1)) % self.size
                tokens.append(a + self.action_offset)
                tokens.append(int(obs[pos]) + self.obs_offset)
                is_revisit.append(pos in seen)
                seen.add(pos)
                self.visited_locations.append((pos,))
                t += 1
        tok = torch.tensor(tokens, dtype=torch.long)
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool); obs_mask[1::2] = True
        rev = torch.zeros(2 * n_steps, dtype=torch.bool)
        rev[1::2] = torch.tensor(is_revisit, dtype=torch.bool)
        return tok, obs_mask, rev
