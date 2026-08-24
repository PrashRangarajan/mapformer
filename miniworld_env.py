"""MiniWorld adapter: continuous-3D navigation with a SYNTHETIC location-obs.

The 3D step between grid MiniGrid and photorealistic Habitat. MiniWorld gives
real continuous geometry and rotation-based actions (15 deg turns, ~0.15 forward
steps whose direction depends on accumulated heading), but its RGB observation
tokenizes degenerately (HABITAT_BUILD.md reconnaissance: 1 code, or 400 codes
over 185 positions). So we DO NOT tokenize the image. Instead each discretized
(x, z) floor cell gets a fixed random observation token, exactly as the torus
and MiniGrid tasks do -- the observation is LOCATION-DETERMINED, so a cognitive
map is required, while the continuous rotation-action geometry is real.

Action recoding (the axis under test):
  raw          {turn_left, turn_right, move_forward} = 3 MiniWorld commands. A
               'forward' displaces the agent along its accumulated heading, so a
               commutative cumsum cannot integrate it (the MiniGrid finding).
  allocentric  the realized per-step displacement DIRECTION, quantized to n_dir
               bins, plus a 'stay' class for turns. Forward magnitude is ~constant
               (0.15), so direction captures the world-fixed displacement. This is
               the input-matched comparison, and a discrete prototype of the
               continuous-displacement recode Habitat would need.

Headless: pyglet EGL (pyglet.options['headless']=True) -- no X display needed.
"""
import math
import numpy as np
import torch

import pyglet
pyglet.options["headless"] = True          # EGL; must precede miniworld import
import gymnasium as gym
import miniworld                            # noqa: F401  (registers envs)


class MiniWorldWorld:
    """MapFormer-compatible trajectory generator on a MiniWorld env."""

    def __init__(self, env_name="MiniWorld-OneRoom-v0", grid_size=12,
                 n_obs_types=16, p_empty=0.5, n_dir=24, seed=0,
                 allocentric=False, max_episode_steps=2000):
        self.env_name = env_name
        self.grid_size = grid_size
        self.n_obs_types = n_obs_types
        self.p_empty = p_empty
        self.n_dir = n_dir
        self.seed = seed
        self.allocentric = allocentric

        self.env = gym.make(env_name, render_mode=None,
                            max_episode_steps=max_episode_steps)
        self.env.reset(seed=seed)
        self.u = self.env.unwrapped
        # Disable the env's internal step-limit (OneRoom truncates at ~180) so a
        # trajectory is ONE continuous episode -- never reset mid-trajectory. An
        # episode reset would teleport the agent with no displacement token,
        # silently breaking path integration and handicapping the path-integrated
        # arm (a confound). `terminated` (touching the goal box) is treated as a
        # wall bump, not a reset.
        self.u.max_episode_steps = 10 ** 9
        # action ids: 0=turn_left, 1=turn_right, 2=move_forward (Discrete(3))
        assert self.env.action_space.n >= 3

        rng = np.random.RandomState(seed)
        self._measure_bounds(rng)
        # a "forward" macro moves ~one grid cell, so each recorded step changes
        # cell -> the observation actually changes (a single 0.15 forward step is
        # sub-cell and makes the task trivially copy-last-obs).
        self.cell_w = min((self.xmax - self.xmin), (self.zmax - self.zmin)) / grid_size
        # cap the forward macro by the WIDER-axis cell width (robust on non-square
        # rooms where the discretized cell can be wider than cell_w on one axis).
        self.cap_w = max((self.xmax - self.xmin), (self.zmax - self.zmin)) / grid_size

        # location-determined observation map (fixed per seed, like torus obs_map)
        self.blank_token = n_obs_types
        obs = np.full((grid_size, grid_size), self.blank_token, dtype=np.int64)
        occ = rng.random((grid_size, grid_size)) >= p_empty
        obs[occ] = rng.randint(0, n_obs_types, occ.sum())
        self.obs_map = obs

        # unified vocab: [actions][obs types][blank]
        self.N_ACTIONS = (1 + n_dir) if allocentric else 3
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.unified_blank = self.obs_offset + self.blank_token
        self.unified_vocab_size = self.N_ACTIONS + n_obs_types + 1
        self.size = grid_size                 # eval-script compatibility

    # -- geometry ----------------------------------------------------------
    def _measure_bounds(self, rng, n=4000):
        """Estimate the reachable (x, z) extent with a long random walk (one
        continuous episode; internal truncation already disabled)."""
        u = self.u
        u.reset(seed=self.seed); u.max_episode_steps = 10 ** 9
        xs, zs = [], []
        for _ in range(n):
            u.step(int(rng.randint(0, 3)))          # unwrapped; ignore term/trunc
            p = u.agent.pos
            xs.append(float(p[0])); zs.append(float(p[2]))
        m = 1e-3
        self.xmin, self.xmax = min(xs) - m, max(xs) + m
        self.zmin, self.zmax = min(zs) - m, max(zs) + m

    def _cell(self, pos):
        gx = int((float(pos[0]) - self.xmin) / (self.xmax - self.xmin) * self.grid_size)
        gz = int((float(pos[2]) - self.zmin) / (self.zmax - self.zmin) * self.grid_size)
        gx = min(max(gx, 0), self.grid_size - 1)
        gz = min(max(gz, 0), self.grid_size - 1)
        return gx, gz

    def _disp_dir(self, prev, cur):
        """Realized displacement direction -> class: 0 stay, 1..n_dir directions."""
        dx, dz = float(cur[0] - prev[0]), float(cur[2] - prev[2])
        if dx * dx + dz * dz < 1e-6:          # a turn, or a blocked forward
            return 0
        ang = math.atan2(dz, dx)              # [-pi, pi]
        b = int(((ang + math.pi) / (2 * math.pi)) * self.n_dir) % self.n_dir
        return 1 + b

    def _sample_action(self, rng):
        return int(rng.choice(3, p=[0.2, 0.2, 0.6]))   # forward-biased

    def _macro(self, a, rng):
        """Execute one MACRO action. a=0/1 turn 15 deg; a=2 forward until the
        grid CELL changes (or wall-blocked / step cap). Uses the UNWRAPPED env
        (u.step) and never resets -- `terminated` (goal box) is a wall bump."""
        u = self.u
        if a in (0, 1):
            u.step(a)
            return
        start_cell = self._cell(u.agent.pos)
        cap = int(math.ceil(2.0 * self.cap_w / 0.15)) + 2
        for _ in range(cap):
            u.step(2)
            if self._cell(u.agent.pos) != start_cell:
                break

    # -- trajectory --------------------------------------------------------
    def generate_trajectory(self, n_steps=128, rng=None):
        if rng is None:
            rng = np.random
        # FRESH location-obs map per episode -> forces IN-CONTEXT map building
        # (the model cannot memorise a single fixed map; the held-out-map eval
        # then cleanly tests path-integration retrieval). Matches the
        # compositional task's fresh_per_episode. Stored on self so external
        # oracle checks can recompute obs for the most recent trajectory.
        obs_map = np.full((self.grid_size, self.grid_size), self.blank_token, dtype=np.int64)
        occ = rng.random((self.grid_size, self.grid_size)) >= self.p_empty
        obs_map[occ] = rng.randint(0, self.n_obs_types, occ.sum())
        self.obs_map = obs_map
        u = self.u
        u.reset(seed=self.seed + int(rng.randint(1_000_000)))
        u.max_episode_steps = 10 ** 9         # one continuous episode, no reset
        tokens, is_rev, visited = [], [], []
        seen = set()
        prev = u.agent.pos.copy()
        prev_cell = self._cell(prev)
        # DIRECTED-WALK policy (analogue of the torus "pick a direction, go a
        # run-length"): mostly forward, with occasional turn-bursts to redirect
        # and a wall-bounce (turn-burst) whenever a forward is blocked. Plain
        # per-step random actions with 15-deg turns explore terribly (the agent
        # slides along walls into a corner and covers ~10 cells); this covers
        # the room and produces genuine long-range cross-cell revisits.
        turn_budget, turn_dir = 0, 0
        for _ in range(n_steps):
            if turn_budget > 0:
                a = turn_dir; turn_budget -= 1
            elif rng.random() < 0.12:                       # start a turn-burst
                turn_dir = int(rng.randint(0, 2)); turn_budget = int(rng.randint(2, 13)); a = turn_dir; turn_budget -= 1
            else:
                a = 2                                       # forward
            self._macro(a, rng)
            cur = u.agent.pos.copy()
            cell = self._cell(cur)
            if a == 2 and cell == prev_cell and turn_budget == 0:   # blocked -> bounce
                turn_dir = int(rng.randint(0, 2)); turn_budget = int(rng.randint(4, 13))
            if self.allocentric:
                tokens.append(self._disp_dir(prev, cur) + self.action_offset)
            else:
                tokens.append(a + self.action_offset)
            tokens.append(int(self.obs_map[cell[0], cell[1]]) + self.obs_offset)
            # CROSS-cell revisit: back at a previously-seen cell, and it is not
            # simply the same cell as the previous step (defeats copy-last-obs).
            is_rev.append(cell in seen and cell != prev_cell)
            seen.add(cell); visited.append(cell)
            prev = cur; prev_cell = cell

        self.visited_locations = visited
        tokens = torch.tensor(tokens, dtype=torch.long)
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool); obs_mask[1::2] = True
        rev_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        for i, r in enumerate(is_rev):
            if r:
                rev_mask[2 * i + 1] = True
        return tokens, obs_mask, rev_mask

    def generate_batch(self, batch_size, n_steps=128, p_action_noise=0.0,
                       p_transition_noise=0.0, policy=None, rng=None):
        # p_action_noise / p_transition_noise / policy are accepted for
        # interface compatibility with GridWorld/MiniGridWorld but are IGNORED
        # here (this study does not use action noise on MiniWorld).
        toks, oms, revs, locs = [], [], [], []
        for _ in range(batch_size):
            t, om, rm = self.generate_trajectory(n_steps, rng=rng)
            toks.append(t); oms.append(om); revs.append(rm)
            locs.append(list(self.visited_locations))
        return torch.stack(toks), torch.stack(oms), torch.stack(revs), locs
