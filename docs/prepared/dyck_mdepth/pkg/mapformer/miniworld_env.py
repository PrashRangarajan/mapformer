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
                 allocentric=False, max_episode_steps=2000, fixed_map=False,
                 oracle=False):
        self.env_name = env_name
        self.grid_size = grid_size
        self.n_obs_types = n_obs_types
        self.p_empty = p_empty
        self.n_dir = n_dir
        self.seed = seed
        self.allocentric = allocentric
        # oracle: emit the EXACT integer cell transition (Δgx, Δgz) each clamped to
        # {-1,0,+1} as a 9-class token, instead of allocentric's 24-bin continuous
        # DIRECTION. This makes the commutative cumsum reconstruct the obs-map cell
        # EXACTLY (R²->1, like the torus / MiniGrid ±x/±y), holding the env and the
        # in-context demand fixed -- the decisive test of whether the fresh-map flip
        # is a reconstruction-FIDELITY problem (forensics: 24-bin allo R²=0.55 due
        # to forward-step magnitude variance CV=0.49). Overrides allocentric.
        self.oracle = oracle
        self._oracle_clamped = 0          # count of multi-cell jumps clamped to sign
        self._oracle_steps = 0
        # fixed_map: the obs_map is drawn ONCE (per seed) and every trajectory
        # reuses it, so the task is pure PATH INTEGRATION on a known map (novel
        # walk each episode) rather than IN-CONTEXT map building. The latter
        # needs ~infinite fresh maps to generalise (a 3k-traj buffer memorises
        # it); the fixed-map task is the data-efficient, well-posed test of the
        # raw-vs-allocentric axis -- identical in spirit to the MiniGrid design.
        self.fixed_map = fixed_map

        self.env = gym.make(env_name, render_mode=None,
                            max_episode_steps=max_episode_steps)
        self.env.reset(seed=seed)
        self.u = self.env.unwrapped
        # DISABLE the POV render. MiniWorld renders the agent camera (EGL/GPU) on
        # every u.step, but we DISCARD the image (the observation is the cell's
        # obs_map token, not pixels). Rendering runs AFTER the physics and touches
        # neither agent.pos nor the RNG, so stubbing render_obs is DATA-INVARIANT
        # (verified: positions identical with/without) while ~30x faster per step
        # and immune to the multi-worker single-GPU render serialization that
        # dominated parallel buffer builds. If a future variant needs the pixels
        # (see PERCEPTION_EXPERIMENT_PLAN.md), gate this stub behind a flag.
        _oshape = getattr(self.env.observation_space, "shape", (60, 80, 3))
        self._blank_obs = np.zeros(_oshape, dtype=np.uint8)
        self.u.render_obs = lambda *a, **k: self._blank_obs
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
        self._geometry_bounds()
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
        # oracle 9 = (Δgx,Δgz) each in {-1,0,+1}; allo 1+n_dir; raw 3.
        self.N_ACTIONS = 9 if oracle else ((1 + n_dir) if allocentric else 3)
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.unified_blank = self.obs_offset + self.blank_token
        self.unified_vocab_size = self.N_ACTIONS + n_obs_types + 1
        self.size = grid_size                 # eval-script compatibility

    # -- geometry ----------------------------------------------------------
    def _geometry_bounds(self):
        """Reachable (x, z) extent read DIRECTLY from room geometry, inset by the
        agent radius (the agent centre cannot reach a wall). Exact, zero-cost, and
        identical across processes -- replaces a 4000-step random-walk estimate
        that under-sampled the extent, cost ~1 s per env construction, and varied
        per process (which would give parallel buffer workers inconsistent cell
        discretization). For OneRoom this yields [0.4, 9.6] on both axes, matching
        the old random walk exactly (radius 0.4, room [0, 10])."""
        u = self.u
        rad = float(getattr(u.agent, "radius", 0.0))
        rooms = getattr(u, "rooms", None)
        if rooms:
            self.xmin = min(r.min_x for r in rooms) + rad
            self.xmax = max(r.max_x for r in rooms) - rad
            self.zmin = min(r.min_z for r in rooms) + rad
            self.zmax = max(r.max_z for r in rooms) - rad
        else:                                       # fallback for room-less envs
            self._measure_bounds(np.random.RandomState(self.seed))

    def _measure_bounds(self, rng, n=4000):
        """Random-walk fallback extent estimate (used only when an env exposes no
        `rooms` list). One continuous episode; internal truncation disabled."""
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
        prev = u.agent.pos.copy()
        for _ in range(cap):
            u.step(2)
            cur = u.agent.pos
            if self._cell(cur) != start_cell:
                break
            if abs(float(cur[0] - prev[0])) + abs(float(cur[2] - prev[2])) < 1e-6:
                break                             # wall-blocked: position frozen
            prev = cur.copy()

    # -- trajectory --------------------------------------------------------
    def generate_trajectory(self, n_steps=128, rng=None):
        if rng is None:
            rng = np.random
        # obs_map: FRESH per episode (in-context map building, data-hungry) OR
        # FIXED per seed (path integration on a known map, data-efficient).
        # Fixed reuses the __init__ map so novel walks share one spatial layout;
        # to know the token at step t the model must integrate actions to the
        # current cell -> the clean raw-vs-allocentric test. Stored on self so
        # external oracle checks can recompute obs for the most recent trajectory.
        if not self.fixed_map:
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
            if self.oracle:
                # EXACT cell transition, each axis clamped to {-1,0,+1} -> 9 classes.
                # A fixed per-id 2D vector then cumsums to the exact cell index.
                rdx, rdz = cell[0] - prev_cell[0], cell[1] - prev_cell[1]
                dgx = max(-1, min(1, rdx)); dgz = max(-1, min(1, rdz))
                self._oracle_steps += 1
                if abs(rdx) > 1 or abs(rdz) > 1:
                    self._oracle_clamped += 1
                tokens.append((dgx + 1) * 3 + (dgz + 1) + self.action_offset)
            elif self.allocentric:
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
