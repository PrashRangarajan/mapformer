import numpy as np, torch
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.model_wrap_exist import action_angles
for d, L in ((128, 1), (256, 2)):
    st, big, n = [], [], []
    for seed in range(1480, 1488):
        torch.manual_seed(seed); np.random.seed(seed)
        env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=seed)
        m = VARIANT_MAP["AbsAO4_r4"](vocab_size=env.unified_vocab_size, d_model=d, n_heads=2, n_layers=L, grid_size=64)
        A = action_angles(m).numpy()          # (4, C)
        with torch.no_grad(): s = m.step_table()[m.act].reshape(4, -1).numpy()
        st.append(np.abs(s).mean()); big.append((A > np.pi).mean()); n.append(A.shape[1])
    print(f"d {d} L {L}: channels {n[0]}, mean |step| {np.mean(st):.3f}, share of (move, channel) angles > pi {np.mean(big):.3f}")
