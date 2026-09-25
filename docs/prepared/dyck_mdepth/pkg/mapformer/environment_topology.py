"""Cross-topology env infrastructure.

Same vocab and actions as standard GridWorld, but different boundary
behavior:

  - TorusGridWorld: standard wraparound (the existing GridWorld behavior)
  - OpenGridWorld: no wraparound. Actions at boundary become no-ops
    (position stays, obs is from current cell, action is still recorded).
  - WallsGridWorld: 2D grid with internal walls. Some cells are impassable.
    Actions that would enter a wall become no-ops.

MultiTopologyGridWorld pools envs from multiple topology classes for
cross-topology training. Same obs vocab across all classes, so the
model's embedding works for any of them.

The cognitive demand: each topology requires the path-integration
machinery to adapt — on torus, cumulative angle wraps; on open grids,
position clamps; on walls, position depends on local wall layout. The
model has to learn to handle no-op actions (i.e., detect that an action
didn't change the obs and infer the topology).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .environment import GridWorld


class OpenGridWorld(GridWorld):
    """Grid without wraparound. Actions at boundary are no-ops."""

    def __init__(
        self,
        size: int = 64,
        n_obs_types: int = 16,
        p_empty: float = 0.5,
        n_landmarks: int = 0,
        seed: Optional[int] = None,
    ):
        super().__init__(size, n_obs_types, p_empty, n_landmarks, seed)
        self.topology = "open"

    def generate_trajectory(
        self, n_steps: int = 128, start: Optional[tuple[int, int]] = None,
        p_transition_noise: float = 0.0,
    ):
        if start is not None:
            x, y = start
        else:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)

        tokens = []
        self.visited_locations = []
        is_revisit = []
        seen = set()

        t = 0
        while t < n_steps:
            a = np.random.randint(0, self.N_ACTIONS)
            k = np.random.randint(1, 11)
            for _ in range(k):
                if t >= n_steps:
                    break
                a_exec = a
                if p_transition_noise > 0.0 and np.random.random() < p_transition_noise:
                    a_exec = np.random.randint(0, self.N_ACTIONS)
                dx, dy = self.ACTION_DELTAS[a_exec]
                nx = x + dx
                ny = y + dy
                # OPEN: clamp to [0, size); action at boundary is no-op
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    x, y = nx, ny
                # else: position stays unchanged; action is still recorded
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
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool); obs_mask[1::2] = True
        revisit_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        for step_idx, rev in enumerate(is_revisit):
            if rev:
                revisit_mask[2 * step_idx + 1] = True
        return tokens, obs_mask, revisit_mask


class WallsGridWorld(GridWorld):
    """Grid with internal walls. Walls are randomly placed; actions into walls
    are no-ops. Wraps around boundaries (torus topology) but walls block paths.
    """

    def __init__(
        self,
        size: int = 64,
        n_obs_types: int = 16,
        p_empty: float = 0.5,
        n_landmarks: int = 0,
        seed: Optional[int] = None,
        wall_density: float = 0.10,
    ):
        super().__init__(size, n_obs_types, p_empty, n_landmarks, seed)
        self.topology = "walls"
        # Place walls (cells marked as impassable)
        rng = np.random.RandomState(seed + 99999 if seed is not None else None)
        wall_mask = np.zeros((size, size), dtype=bool)
        n_walls = int(wall_density * size * size)
        wall_indices = rng.choice(size * size, size=n_walls, replace=False)
        for idx in wall_indices:
            wx, wy = idx // size, idx % size
            # Don't make landmark cells walls
            if hasattr(self, 'is_landmark_cell') and self.is_landmark_cell[wx, wy]:
                continue
            wall_mask[wx, wy] = True
        self.wall_mask = torch.from_numpy(wall_mask)

    def generate_trajectory(
        self, n_steps: int = 128, start: Optional[tuple[int, int]] = None,
        p_transition_noise: float = 0.0,
    ):
        # Find a valid starting cell (not a wall)
        if start is not None:
            x, y = start
        else:
            while True:
                x = np.random.randint(0, self.size)
                y = np.random.randint(0, self.size)
                if not self.wall_mask[x, y]:
                    break

        tokens = []
        self.visited_locations = []
        is_revisit = []
        seen = set()

        t = 0
        while t < n_steps:
            a = np.random.randint(0, self.N_ACTIONS)
            k = np.random.randint(1, 11)
            for _ in range(k):
                if t >= n_steps:
                    break
                a_exec = a
                if p_transition_noise > 0.0 and np.random.random() < p_transition_noise:
                    a_exec = np.random.randint(0, self.N_ACTIONS)
                dx, dy = self.ACTION_DELTAS[a_exec]
                nx = (x + dx) % self.size  # torus boundary
                ny = (y + dy) % self.size
                # If next cell is a wall, action is no-op
                if not self.wall_mask[nx, ny]:
                    x, y = nx, ny
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
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool); obs_mask[1::2] = True
        revisit_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        for step_idx, rev in enumerate(is_revisit):
            if rev:
                revisit_mask[2 * step_idx + 1] = True
        return tokens, obs_mask, revisit_mask


class MultiTopologyGridWorld:
    """Pool of envs from multiple topology classes.

    Topologies: 'torus' (wraparound), 'open' (no wraparound), 'walls'
    (torus + internal walls). All share the same vocab.
    """

    N_ACTIONS = GridWorld.N_ACTIONS

    def __init__(
        self,
        topologies=("torus", "open", "walls"),
        size: int = 64,
        n_obs_types: int = 16,
        p_empty: float = 0.5,
        n_landmarks: int = 200,
        n_envs_per_topology: int = 20,
        n_test_envs_per_topology: int = 20,
        seed: int = 0,
    ):
        self.topologies = list(topologies)
        self.size = size

        topo_cls = {"torus": GridWorld, "open": OpenGridWorld, "walls": WallsGridWorld}

        self.train_envs_by_topo = {}
        self.test_envs_by_topo = {}
        seed_offset = 1
        for topo in topologies:
            cls = topo_cls[topo]
            envs = []
            for _ in range(n_envs_per_topology):
                env = cls(size=size, n_obs_types=n_obs_types, p_empty=p_empty,
                          n_landmarks=n_landmarks, seed=seed + seed_offset)
                envs.append(env)
                seed_offset += 1
            self.train_envs_by_topo[topo] = envs

            envs_test = []
            for _ in range(n_test_envs_per_topology):
                env = cls(size=size, n_obs_types=n_obs_types, p_empty=p_empty,
                          n_landmarks=n_landmarks, seed=seed + seed_offset + 100000)
                envs_test.append(env)
                seed_offset += 1
            self.test_envs_by_topo[topo] = envs_test

        sample_env = self.train_envs_by_topo[topologies[0]][0]
        self.unified_vocab_size = sample_env.unified_vocab_size

    def generate_trajectory(
        self, n_steps: int = 128, train: bool = True, topology: Optional[str] = None,
        p_transition_noise: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        if rng is None: rng = np.random
        pool = self.train_envs_by_topo if train else self.test_envs_by_topo
        if topology is None:
            topology = self.topologies[int(rng.randint(0, len(self.topologies)))]
        env = pool[topology][int(rng.randint(0, len(pool[topology])))]
        tokens, obs_mask, rev_mask = env.generate_trajectory(
            n_steps, p_transition_noise=p_transition_noise,
        )
        return tokens, obs_mask, rev_mask, topology

    def generate_batch(
        self, batch_size: int, n_steps: int = 128, train: bool = True,
        topology: Optional[str] = None,
        p_transition_noise: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        toks, oms, rms, topos = [], [], [], []
        for _ in range(batch_size):
            t, om, rm, topo = self.generate_trajectory(
                n_steps, train=train, topology=topology,
                p_transition_noise=p_transition_noise, rng=rng,
            )
            toks.append(t); oms.append(om); rms.append(rm); topos.append(topo)
        return (torch.stack(toks), torch.stack(oms), torch.stack(rms), topos)
