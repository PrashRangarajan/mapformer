"""
MapFormer: Self-Supervised Cognitive Maps with Lie Group Path Integration.

Implementation based on Rambaud et al. (2025), arXiv:2511.19279.
Includes InEKF extension for uncertainty-aware path integration.
"""

from .lie_groups import (
    skew_symmetric_2d,
    exp_map_2d,
    log_map_2d,
    build_block_diagonal_rotations,
    build_block_diagonal_rotations_fast,
    exp_map_so_n,
    log_map_so_n,
    is_orthogonal,
    is_special_orthogonal,
)
from .prefix_scan import (
    parallel_prefix_product,
    sequential_prefix_product,
)
from .environment import GridWorld
from .model import MapFormerWM, MapFormerEM, ActionToLieAlgebra
from .baselines import TransformerRoPE, LSTMBaseline
from .inekf import InEKFLayer
from .train import train
from .evaluate import (
    eval_length_generalisation,
    extract_position_states,
    compute_rate_map,
    plot_length_generalisation,
    plot_position_pca,
    plot_grid_cell_autocorrelation,
    plot_training_curves,
)
