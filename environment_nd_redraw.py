"""GridWorldND with the observation map REDRAWN for every trajectory (RANK_NOWRAP Amendment 1, memorisation control).

On the 32-torus the training map is fixed per run seed (1024 cells), so a model can partly memorise it (RANK_ND S3:
failing rank-2 arms scored +0.17 higher on their own map). Redrawing the map per trajectory removes that solution while
keeping everything else of the task: the walk, the torus, the wrap-around revisits, the scoring.

WHAT IS INHERITED, unchanged: GridWorldND.generate_trajectory (the directed walk, wrap, interleaving, revisit mask) and
generate_batch -- this class overrides only generate_trajectory, to replace self.obs_map and then call the parent.

THE MAP DRAW does not touch the global numpy RNG, so the walk (start cell, actions, run lengths) is the one the parent
would produce from the same global RNG state -- at a given seed the action stream is IDENTICAL to the fixed-map arm's;
only the observations differ. The map is drawn by GridWorldND's own map code (`draw_map`, the same three lines as
GridWorldND.__init__, verified equal for a fixed seed) from a RandomState whose seed is a CRC32 of the global RNG state
at the moment the trajectory starts: deterministic, different for every trajectory, and independent of worker count
(data_parallel seeds the global RNG per batch index).
"""
import zlib

import numpy as np
import torch

from mapformer.environment_nd import GridWorldND


def draw_map(rng, dims, size, n_obs_types, p_empty, blank_token):
    """GridWorldND.__init__'s map draw, verbatim."""
    shape = (size,) * dims
    obs_map = np.full(shape, blank_token, dtype=np.int64)
    occupied = rng.random(shape) >= p_empty
    obs_map[occupied] = rng.randint(0, n_obs_types, int(occupied.sum()))
    return torch.from_numpy(obs_map).long()


class GridWorldNDRedraw(GridWorldND):
    REDRAW = True

    def _map_seed(self) -> int:
        st = np.random.get_state()                      # ('MT19937', keys[624] uint32, pos, has_gauss, cached)
        return zlib.crc32(np.asarray(st[1], np.uint32).tobytes() + int(st[2]).to_bytes(4, "little"))

    def generate_trajectory(self, n_steps: int = 128, start=None):
        self.obs_map = draw_map(np.random.RandomState(self._map_seed()), self.dims, self.size, self.n_obs_types,
                                self.p_empty, self.blank_token)
        return super().generate_trajectory(n_steps, start)
