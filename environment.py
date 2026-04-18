"""
2D Grid environment for MapFormer training and evaluation.

Matches the setup in Rambaud et al. (2025):
- TORUS grid (wrapping boundaries, not clamped)
- Directed walks: sample direction + k steps (1 <= k <= 10)
- p_empty fraction of cells are empty (blank token B)
- Returns INTERLEAVED token sequence s = (a1, o1, a2, o2, ..., aT, oT)
  with a unified vocabulary: [actions 0..3] [obs 4..4+K-1] [blank 4+K]
  (plus L landmark tokens if n_landmarks > 0)

Landmark extension:
  n_landmarks > 0 reserves that many unique token IDs, one per chosen cell.
  Each landmark cell emits its unique token (unambiguous position signal).
  Selected landmark cells OVERRIDE whatever regular obs / blank was there.
  This is the regime where Kalman/PC corrections have sharp measurements.
"""

import torch
import numpy as np
from typing import Optional


class GridWorld:
    """2D torus grid world matching the paper's forced-navigation task.

    Actions: 0=North, 1=South, 2=West, 3=East (in unified vocab: indices 0..3)
    Observations: K object types + 1 blank (in unified vocab: indices 4..4+K)

    The grid is a TORUS: movements wrap around edges.
    """

    N_ACTIONS = 4
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(
        self,
        size: int = 64,
        n_obs_types: int = 16,
        p_empty: float = 0.5,
        n_landmarks: int = 0,
        seed: Optional[int] = None,
    ):
        self.size = size
        self.n_obs_types = n_obs_types
        self.p_empty = p_empty
        self.n_landmarks = n_landmarks

        # Unified vocabulary layout:
        # [0..3]               = actions (N, S, W, E)
        # [4..4+K-1]           = K regular obs types
        # [4+K]                = blank token B
        # [4+K+1..4+K+L]       = L unique landmark tokens (one per landmark cell)
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS  # = 4
        self.unified_blank = self.N_ACTIONS + n_obs_types  # = 4 + K
        self.first_landmark_rel = n_obs_types + 1  # relative to obs vocab
        self.first_landmark_unified = self.N_ACTIONS + self.first_landmark_rel  # = 4+K+1

        self.obs_vocab_size = n_obs_types + 1 + n_landmarks
        self.unified_vocab_size = self.N_ACTIONS + self.obs_vocab_size

        self.blank_token = n_obs_types

        rng = np.random.RandomState(seed)

        # Assign regular observations: each cell is empty with prob p_empty
        obs_map = np.full((size, size), self.blank_token, dtype=np.int64)
        is_occupied = rng.random((size, size)) >= p_empty
        obs_map[is_occupied] = rng.randint(0, n_obs_types, is_occupied.sum())

        # Override with landmarks: pick n_landmarks random cells and assign
        # each a unique landmark token. Landmarks win over regular obs / blank.
        if n_landmarks > 0:
            n_cells = size * size
            assert n_landmarks <= n_cells, \
                f"n_landmarks ({n_landmarks}) exceeds n_cells ({n_cells})"
            cell_indices = rng.permutation(n_cells)[:n_landmarks]
            self.landmark_cells = []
            for idx, ci in enumerate(cell_indices):
                i, j = int(ci // size), int(ci % size)
                # Landmark relative to obs vocab:
                #   blank = n_obs_types, landmarks are n_obs_types+1 ... n_obs_types+L
                lm_rel = self.first_landmark_rel + idx
                obs_map[i, j] = lm_rel
                self.landmark_cells.append((i, j, idx))  # (x, y, landmark_idx)
        else:
            self.landmark_cells = []

        self.obs_map = torch.from_numpy(obs_map).long()

        # Convenience: boolean mask per cell indicating landmark-ness
        lm_mask = np.zeros((size, size), dtype=bool)
        for x, y, _ in self.landmark_cells:
            lm_mask[x, y] = True
        self.is_landmark_cell = torch.from_numpy(lm_mask)

        self.visited_locations: list[tuple[int, int]] = []
        self.last_x = size // 2
        self.last_y = size // 2

    def generate_trajectory(
        self, n_steps: int = 128, start: Optional[tuple[int, int]] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a directed-walk trajectory as an interleaved token sequence.

        Returns:
            tokens: (2*n_steps,) interleaved [a1, o1, a2, o2, ...] in unified vocab
            obs_mask: (2*n_steps,) bool, True at observation positions (odd indices)
            revisit_mask: (2*n_steps,) bool, True at obs positions for REVISITED cells.
                First visit: False. Revisit: True. Action positions: False.
                This is the paper's prediction target — "predict observation each
                time it comes back to a previously visited location."
        """
        if start is not None:
            x, y = start
        else:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)

        tokens = []
        self.visited_locations = []
        is_revisit = []  # per-step revisit flag
        seen = set()

        t = 0
        while t < n_steps:
            a = np.random.randint(0, self.N_ACTIONS)
            k = np.random.randint(1, 11)

            for _ in range(k):
                if t >= n_steps:
                    break
                dx, dy = self.ACTION_DELTAS[a]
                x = (x + dx) % self.size
                y = (y + dy) % self.size

                tokens.append(a + self.action_offset)
                obs_idx = self.obs_map[x, y].item()
                tokens.append(obs_idx + self.obs_offset)

                self.visited_locations.append((x, y))
                is_revisit.append((x, y) in seen)
                seen.add((x, y))
                t += 1

        self.last_x = x
        self.last_y = y

        tokens = torch.tensor(tokens, dtype=torch.long)
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        obs_mask[1::2] = True

        # revisit_mask aligned with obs positions
        revisit_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        for step_idx, rev in enumerate(is_revisit):
            if rev:
                revisit_mask[2 * step_idx + 1] = True  # obs position at step step_idx

        return tokens, obs_mask, revisit_mask

    def generate_batch(
        self, batch_size: int, n_steps: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
        """Generate a batch of interleaved trajectories.

        Returns:
            tokens: (batch_size, 2*n_steps)
            obs_mask: (batch_size, 2*n_steps)
            revisit_mask: (batch_size, 2*n_steps)
            all_locations: list of location lists for each trajectory
        """
        all_tokens = []
        all_masks = []
        all_revisit = []
        all_locations = []

        for _ in range(batch_size):
            tok, mask, rev = self.generate_trajectory(n_steps)
            all_tokens.append(tok)
            all_masks.append(mask)
            all_revisit.append(rev)
            all_locations.append(list(self.visited_locations))

        return (
            torch.stack(all_tokens),
            torch.stack(all_masks),
            torch.stack(all_revisit),
            all_locations,
        )
