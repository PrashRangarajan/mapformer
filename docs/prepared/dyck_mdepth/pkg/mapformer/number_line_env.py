"""NumberLineWorld — arithmetic as navigation on a 1D additive torus.

A cognitive-map framing of modular arithmetic. State is a single number
v in {0, ..., N-1}; actions are additive operations (+1, -1, +2, ...);
the position wraps modulo N. MapFormer's path integrator —
theta = omega * cumsum(f_Delta(action)) — IS cumulative summation, so
running this model on a NumberLineWorld literally computes
(a + b + c + ...) mod N by path integration.

Interface matches GridWorld exactly so the existing train.py / model code
works unchanged:
  - interleaved token stream [a1, o1, a2, o2, ...]
  - unified vocab: [actions] [obs types] [blank] [landmarks]
  - generate_trajectory / generate_batch
  - revisit_mask True at obs positions for REVISITED values (the cognitive-
    map prediction target: predict the token at a number you've reached
    before, possibly via a different path)

The mental-number-line analogy: humans represent number spatially; grid
codes appear for abstract 1D dimensions (Constantinescu 2016). This env
is the simplest test of 'cognitive map for arithmetic'.
"""

import torch
import numpy as np
from typing import Optional


class NumberLineWorld:
    """1D additive torus. Actions add fixed deltas; position is mod N."""

    # 6 additive operations: +1, -1, +2, -2, +3, -3
    N_ACTIONS = 6
    ACTION_DELTAS = {0: 1, 1: -1, 2: 2, 3: -2, 4: 3, 5: -3}

    def __init__(
        self,
        size: int = 64,             # N — the modulus / number-line length
        n_obs_types: int = 16,
        p_empty: float = 0.5,
        n_landmarks: int = 0,
        seed: Optional[int] = None,
    ):
        self.size = size
        self.n_obs_types = n_obs_types
        self.p_empty = p_empty
        self.n_landmarks = n_landmarks

        # Unified vocab layout — same scheme as GridWorld.
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.unified_blank = self.N_ACTIONS + n_obs_types
        self.first_landmark_rel = n_obs_types + 1
        self.first_landmark_unified = self.N_ACTIONS + self.first_landmark_rel
        self.obs_vocab_size = n_obs_types + 1 + n_landmarks
        self.unified_vocab_size = self.N_ACTIONS + self.obs_vocab_size
        self.blank_token = n_obs_types

        rng = np.random.RandomState(seed)

        # obs_map: a token per value on the number line.
        obs_map = np.full((size,), self.blank_token, dtype=np.int64)
        is_occupied = rng.random(size) >= p_empty
        obs_map[is_occupied] = rng.randint(0, n_obs_types, is_occupied.sum())

        # Landmarks = "memorable numbers" with unique tokens.
        if n_landmarks > 0:
            assert n_landmarks <= size, \
                f"n_landmarks ({n_landmarks}) exceeds N ({size})"
            value_indices = rng.permutation(size)[:n_landmarks]
            self.landmark_cells = []
            for idx, vi in enumerate(value_indices):
                lm_rel = self.first_landmark_rel + idx
                obs_map[vi] = lm_rel
                self.landmark_cells.append((int(vi), idx))  # (value, landmark_idx)
        else:
            self.landmark_cells = []

        self.obs_map = torch.from_numpy(obs_map).long()
        self.visited_locations: list[int] = []

    def generate_trajectory(
        self, n_steps: int = 128, start: Optional[int] = None,
        p_transition_noise: float = 0.0,
    ):
        """Directed additive walk. Returns (tokens, obs_mask, revisit_mask).

        Token stream is interleaved [a1, o1, a2, o2, ...]. revisit_mask is
        True at obs positions whose VALUE has been reached before in this
        trajectory (predict-the-token-at-a-revisited-number — the cognitive-
        map target). p_transition_noise: executed op differs from the
        commanded (recorded) op with this probability.
        """
        if start is not None:
            v = start
        else:
            v = int(np.random.randint(0, self.size))

        tokens = []
        self.visited_locations = []
        is_revisit = []
        seen = set()

        t = 0
        while t < n_steps:
            a = np.random.randint(0, self.N_ACTIONS)
            k = np.random.randint(1, 11)        # directed walk: repeat op k times
            for _ in range(k):
                if t >= n_steps:
                    break
                a_exec = a
                if p_transition_noise > 0.0 and np.random.random() < p_transition_noise:
                    a_exec = np.random.randint(0, self.N_ACTIONS)
                v = (v + self.ACTION_DELTAS[a_exec]) % self.size
                tokens.append(a + self.action_offset)          # commanded op recorded
                tokens.append(int(self.obs_map[v].item()) + self.obs_offset)
                self.visited_locations.append(v)
                is_revisit.append(v in seen)
                seen.add(v)
                t += 1

        tokens = torch.tensor(tokens, dtype=torch.long)
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        obs_mask[1::2] = True
        revisit_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        for step_idx, rev in enumerate(is_revisit):
            if rev:
                revisit_mask[2 * step_idx + 1] = True
        return tokens, obs_mask, revisit_mask

    def generate_batch(
        self, batch_size: int, n_steps: int = 128,
        p_transition_noise: float = 0.0,
    ):
        all_tokens, all_masks, all_revisit, all_locations = [], [], [], []
        for _ in range(batch_size):
            tok, mask, rev = self.generate_trajectory(
                n_steps, p_transition_noise=p_transition_noise,
            )
            all_tokens.append(tok)
            all_masks.append(mask)
            all_revisit.append(rev)
            all_locations.append(list(self.visited_locations))
        return (torch.stack(all_tokens), torch.stack(all_masks),
                torch.stack(all_revisit), all_locations)
