"""Exact wrap-only revisit counts at the chosen large grid (complements rank_nowrap_gate_n1000_out.txt, which prints the
share to 4 decimals). Uses the gate's floors_and_strata on the same walks: held-out map seed 10000, walk seed 10000,
and the training-map walks (map seed 200, walk seed 200), 1000 trajectories each at T=1024."""
import importlib.util
spec = importlib.util.spec_from_file_location("gate", "/home/prashr/mapformer/docs/audits/2026-10-06/rank_nowrap_gate.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
for N in (192, 256):
    for ms in (10000, 200):
        env, tr = g.rollout(N, 1024, 1000, ms, ms)
        f = g.floors_and_strata(env, tr, 1024)
        print(f"grid {N} map/walk seed {ms}: scored {f['scored']}, wrap-only {round(f['wrap_share'] * f['scored'])} "
              f"(share {f['wrap_share']:.2e}), retrace {f['retrace']:.4f}, blank {f['const_blank']:.4f}", flush=True)
