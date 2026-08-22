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
        action_mode: str = "translate",
        obs_mode: str = "allo",
        boundary: str = "torus",
        score_moves_only: bool = False,
    ):
        """Three knobs isolate what differs between this torus and MiniGrid.

        `MINIGRID_2X2X2.md` and `FREQ_CONTROL.md` establish that path
        integration is worth +0.461 on this torus and **-0.060** on
        MiniGrid-DoorKey-16x16 -- the sign of a large effect flips between
        environments. But the two differ in FIVE ways at once (observation
        frame, action space, map size, aliasing, boundaries), so which one is
        responsible is unknown. Size and aliasing are already parameters; these
        add the other three, so they can be turned one at a time.

        action_mode  "translate" 4 actions, fixed displacements N/S/W/E (default,
                                 the paper's setting -- unchanged behaviour)
                     "rotate"    3 actions, turn-left / turn-right / forward, as
                                 in MiniGrid. Displacement depends on accumulated
                                 heading, which is exactly the assumption
                                 MapFormer's cumsum-of-fixed-deltas makes and
                                 which MiniGrid violates.
        obs_mode     "allo"      observation is the cell you occupy (default)
                     "ego"       observation is the cell one step AHEAD in the
                                 current heading, as in MiniGrid's egocentric
                                 view. In translate mode heading is taken from
                                 the last commanded action, so the two knobs stay
                                 independent.
        boundary     "torus"     wraps (default)
                     "wall"      a move into the boundary is a NO-OP: the action
                                 is still recorded but the position does not
                                 change, matching MiniGrid's bump semantics.

        Every default reproduces the previous behaviour exactly.
        """
        assert action_mode in ("translate", "rotate"), action_mode
        assert obs_mode in ("allo", "ego"), obs_mode
        assert boundary in ("torus", "wall"), boundary
        self.action_mode = action_mode
        self.obs_mode = obs_mode
        self.boundary = boundary
        # score_moves_only: skip steps where the OBSERVED CELL did not change.
        # Default False leaves every existing configuration untouched (in
        # translate+torus the agent moves on every step, so it is a no-op there
        # anyway). Needed for rotate mode, where turns do not translate: a run of
        # "turn left" emits the same observation repeatedly, and predicting "the
        # same as last time" then solves 93% of scored events (KNOB_SWEEP.md).
        # That is not a cognitive-map test, it is a copy test.
        self.score_moves_only = score_moves_only
        self.size = size
        self.n_obs_types = n_obs_types
        self.p_empty = p_empty
        self.n_landmarks = n_landmarks

        # Unified vocabulary layout:
        # [0..3]               = actions (N, S, W, E)
        # [4..4+K-1]           = K regular obs types
        # [4+K]                = blank token B
        # [4+K+1..4+K+L]       = L unique landmark tokens (one per landmark cell)
        # rotate mode uses 3 actions (turn-left, turn-right, forward); the
        # vocabulary keeps 4 action slots either way so a checkpoint trained in
        # one mode still loads in the other and the comparison is not confounded
        # by a vocabulary-size difference.
        self.n_actions = 3 if action_mode == "rotate" else self.N_ACTIONS
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
        self, n_steps: int = 128, start: Optional[tuple[int, int]] = None,
        p_transition_noise: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a directed-walk trajectory as an interleaved token sequence.

        Args:
            p_transition_noise: per-step probability that the *executed* action
                differs from the commanded action. The token sequence records
                the COMMANDED action; the position update uses a random
                replacement action with this probability. This models a
                stochastic-transition MDP — distinct from train.py's
                ``p_action_noise`` (which corrupts action *records* post-hoc).

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

        # Heading is only meaningful for the rotate/ego knobs. It MUST NOT be
        # drawn in the default configuration: consuming one extra value from the
        # global RNG shifts every subsequent draw, which silently changes the
        # default trajectory stream and would invalidate any comparison against
        # an existing checkpoint. Verified byte-identical to the pre-knob code.
        heading = (int(np.random.randint(0, self.N_ACTIONS))
                   if (self.action_mode == "rotate" or self.obs_mode == "ego")
                   else 0)

        def _step(x, y, heading, a_exec):
            """Apply one executed action. Returns (x, y, heading)."""
            if self.action_mode == "rotate":
                # 0 = turn left, 1 = turn right, 2 = forward. Displacement
                # depends on the ACCUMULATED heading, which is precisely what a
                # cumsum of per-token fixed deltas cannot represent.
                if a_exec == 0:
                    return x, y, (heading - 1) % self.N_ACTIONS
                if a_exec == 1:
                    return x, y, (heading + 1) % self.N_ACTIONS
                dx, dy = self.ACTION_DELTAS[heading]
            else:
                dx, dy = self.ACTION_DELTAS[a_exec]
                heading = a_exec                    # for ego view in translate mode
            nx, ny = x + dx, y + dy
            if self.boundary == "wall":
                # bumping a wall is a NO-OP, as in MiniGrid: the action is still
                # recorded, the position does not change.
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    return x, y, heading
                return nx, ny, heading
            return nx % self.size, ny % self.size, heading

        prev_obs_cell = None
        t = 0
        while t < n_steps:
            a = np.random.randint(0, self.n_actions)
            k = np.random.randint(1, 11)

            for _ in range(k):
                if t >= n_steps:
                    break

                # Stochastic-transition MDP: commanded action is `a`, executed
                # may differ. We RECORD the commanded action but APPLY the
                # (possibly different) executed one. This is mathematically
                # equivalent to action-record corruption for a uniform policy
                # but corresponds to a different real-world failure mode (env
                # stochasticity vs sensor/log corruption).
                a_exec = a
                if p_transition_noise > 0.0 and np.random.random() < p_transition_noise:
                    a_exec = np.random.randint(0, self.n_actions)

                x, y, heading = _step(x, y, heading, a_exec)

                tokens.append(a + self.action_offset)        # COMMANDED action recorded
                if self.obs_mode == "ego":
                    # the cell one step AHEAD in the current heading
                    hx, hy = self.ACTION_DELTAS[heading]
                    ox, oy = x + hx, y + hy
                    if self.boundary == "wall":
                        ox, oy = min(max(ox, 0), self.size - 1), min(max(oy, 0), self.size - 1)
                    else:
                        ox, oy = ox % self.size, oy % self.size
                else:
                    ox, oy = x, y
                obs_idx = self.obs_map[ox, oy].item()
                tokens.append(obs_idx + self.obs_offset)

                # Revisit is keyed on whatever DETERMINES the observation: the
                # agent's cell in allo mode, the OBSERVED cell in ego mode. An
                # earlier version keyed rotate mode on (x, y, heading), which was
                # wrong -- in allo mode the observation is obs_map[x, y]
                # regardless of heading, so heading in the key manufactured
                # spurious first-visits and spurious revisits, and combined with
                # spinning it produced the 0.932 order-1 shortcut that voided the
                # rotate condition.
                key = (ox, oy)
                moved = (ox, oy) != prev_obs_cell
                prev_obs_cell = (ox, oy)
                self.visited_locations.append((x, y))
                is_revisit.append((key in seen) and (moved or not self.score_moves_only))
                seen.add(key)
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
        self, batch_size: int, n_steps: int = 128,
        p_transition_noise: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
        """Generate a batch of interleaved trajectories.

        Args:
            p_transition_noise: forwarded to generate_trajectory; see there.

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
            tok, mask, rev = self.generate_trajectory(
                n_steps, p_transition_noise=p_transition_noise,
            )
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
