"""SEARCH_THEORY A7 (eval-only, CPU, post hoc): GAUGE +1/+2 cell. Two conditions make a +1/+2 map: the RELATION
a(+2) = 2 a(+1) (mod 2 pi) and, because every revisit is a lap (displacement 32 k), the LAP LATTICE 32 a(+1) = 0 (mod 2 pi).
Per run: usage-weighted share of channels meeting each (|error| < 0.3 rad), excluding blind channels (|a(+1)| <= 0.05),
plus the usage share on blind channels. Usage as in sign_clock.py.
  python3 lattice_p.py > lattice_p_out.txt"""
import sys, json
from pathlib import Path
sys.path.insert(0, "/home/prashr")
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from sign_clock import usage, angles, wrapa
R = Path("/home/prashr/mapformer/runs/gauge/p0")
for cell in ("SO_p", "AO_p"):
    print(f"-- {cell}")
    for s in range(1300, 1308):
        d = torch.load(R / f"{cell}_s{s}/{cell[:2]}.pt", map_location="cpu", weights_only=False); sd = d["model_state_dict"]
        ev = json.load(open(R / f"{cell}_s{s}/eval.json"))
        A = angles(sd, cell.startswith("SO"), 2); u = usage(sd); w = u / u.sum()
        a1, a2 = A[0], A[1]
        blind = np.abs(wrapa(a1)) <= 0.05
        TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
        rel = np.abs(wrapa(a2 - 2 * a1)) < TOL; lat = np.abs(wrapa(32 * a1)) < TOL
        nb = ~blind; W = w[nb].sum()
        print(f"  s{s} {'SOLV' if ev['final_loss'] < 0.05 else '----'} acc {ev['eval']['128']['acc']:.3f} | usage: blind {w[blind].sum():.2f}; "
              f"of non-blind: relation {w[nb & rel].sum() / W:.2f}, lattice {w[nb & lat].sum() / W:.2f}, both {w[nb & rel & lat].sum() / W:.2f}")
