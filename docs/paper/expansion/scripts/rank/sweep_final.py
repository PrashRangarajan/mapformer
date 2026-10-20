"""ANALYSIS (post hoc, CPU): drift geometry of every stored matched-length rank checkpoint (final weights).

For docs/paper/expansion/RANK_THEORY.md, checks C1-C4. Per head: rank_geom.head_geometry; per run: SOLVED by the
registered rule (stats_core.classify_run: final-5% training loss < 0.05) and the T1 basin (CLEAN / CLOCK / COLLAPSE).
Output: sweep_final_out.txt and sweep_final.json next to this file.
Run: cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/expansion/scripts/rank/sweep_final.py
"""
import glob, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(2)
from rank_geom import head_geometry, basin  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402
from mapformer.stats_core import classify_run  # noqa: E402
from mapformer import train_score_rank  # noqa: E402,F401  (registers MapPoPE-Pair arms)
from mapformer.train_ring_rank import ARMS as RING  # noqa: E402

R = "/home/prashr/mapformer/runs"
# (label, glob of checkpoints, D, per-head rank, kind)
SETS = [
    ("torus G64 shared r2 (rank_mi)", f"{R}/rank_mi/p0/Vanilla_s*/Vanilla.pt", 2, 2, "torus"),
    ("torus G64 per-head r2 (rank_mi)", f"{R}/rank_mi/p0/Vanilla_r2ph_s*/Vanilla_r2ph.pt", 2, 2, "torus"),
    ("torus G64 block-diag r4 (rank_sep)", f"{R}/rank_sep/p0/Vanilla_r4mibd_s*/Vanilla_r4mibd.pt", 2, 2, "torus"),
    ("torus G64 shared r2 (score_rank)", f"{R}/score_rank/p0/Vanilla_s*/Vanilla.pt", 2, 2, "torus"),
    ("torus G64 shared r2 1800ep (loop_rank_e1800)", f"{R}/loop_rank_e1800/p0/Vanilla_s*/Vanilla.pt", 2, 2, "torus"),
    ("torus G64 per-head r3 (rank3)", f"{R}/rank3/p0/Vanilla_r3ph_s*/Vanilla_r3ph.pt", 2, 3, "torus"),
    ("torus G64 shared r4 (rank_mi)", f"{R}/rank_mi/p0/Vanilla_r4mi_s*/Vanilla_r4mi.pt", 2, 4, "torus"),
    ("torus G64 per-head r4 (rank_sep)", f"{R}/rank_sep/p0/Vanilla_r4ph_s*/Vanilla_r4ph.pt", 2, 4, "torus"),
    ("torus G64 shared r4 (score_rank)", f"{R}/score_rank/p0/Vanilla_r4mi_s*/Vanilla_r4mi.pt", 2, 4, "torus"),
    ("ND D2 G32 r2 (rank_nd)", f"{R}/rank_nd/D2/Vanilla_r2ph_s*/Vanilla_r2ph.pt", 2, 2, "nd"),
    ("ND D2 G32 r3 (rank_nd)", f"{R}/rank_nd/D2/Vanilla_r3ph_s*/Vanilla_r3ph.pt", 2, 3, "nd"),
    ("ND D3 G10 r3 (rank_nd)", f"{R}/rank_nd/D3/Vanilla_r3ph_s*/Vanilla_r3ph.pt", 3, 3, "nd"),
    ("ND D3 G10 r4 (rank_nd)", f"{R}/rank_nd/D3/Vanilla_r4ph_s*/Vanilla_r4ph.pt", 3, 4, "nd"),
    ("ND D3 G18 r4 (rank_wrap)", f"{R}/rank_wrap/N18/D3/Vanilla_r4ph_s*/Vanilla_r4ph.pt", 3, 4, "nd"),
    ("ND D3 G10 r4 (rank_wrap)", f"{R}/rank_wrap/N10/D3/Vanilla_r4ph_s*/Vanilla_r4ph.pt", 3, 4, "nd"),
    ("ND D2 G32 r3 (rank_wrap)", f"{R}/rank_wrap/N32/D2/Vanilla_r3ph_s*/Vanilla_r3ph.pt", 2, 3, "nd"),
    ("ring R1 (rank_tests)", f"{R}/rank_tests/ring/p0/R1_s*/R1.pt", 1, 1, "ring"),
    ("ring R2 (rank_tests)", f"{R}/rank_tests/ring/p0/R2_s*/R2.pt", 1, 2, "ring"),
    ("ring R2S shared r2 (rank_tests)", f"{R}/rank_tests/ring/p0/R2S_s*/R2S.pt", 1, 2, "ring"),
    ("ring R1AO (rank_tests)", f"{R}/rank_tests/ring/p0/R1AO_s*/R1AO.pt", 1, 1, "ring"),
    ("escape E lifted r2->r4", f"{R}/rank_tests/escape/E/p0/*_s*/*.pt", 2, 4, "torus"),
    ("escape C r2 continued", f"{R}/rank_tests/escape/C/p0/*_s*/*.pt", 2, 2, "torus"),
    ("escape F fresh r4 step map", f"{R}/rank_tests/escape/F/p0/*_s*/*.pt", 2, 4, "torus"),
    ("paper torus T=128 shared r2 (rank_sweep)", f"{R}/rank_sweep/p0/Vanilla_s*/Vanilla.pt", 2, 2, "torus"),
    ("paper torus T=128 shared r4 (rank_sweep)", f"{R}/rank_sweep/p0/Vanilla_r4_s*/Vanilla_r4.pt", 2, 4, "torus"),
]


def load(ck, kind):
    b = torch.load(ck, map_location="cpu", weights_only=False)
    c = b["config"]; arm = b["variant"]
    if kind == "ring":
        kw = dict(vocab_size=c["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32)
        if arm in ("R1DF", "R1AO"):
            kw["action_ids"] = (0, 1)
        if arm == "R2S":
            kw["bottleneck_r"] = 2
        m = RING[arm](**kw)
    else:
        m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c.get("d_model", 128), n_heads=c.get("n_heads", 2),
                             n_layers=c.get("n_layers", 1), grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); m.eval()
    return m, b


def main():
    out = {}
    lines = []
    for lab, pat, D, r, kind in SETS:
        cks = sorted(glob.glob(pat))
        if not cks:
            lines.append(f"## {lab}: NO CHECKPOINTS ({pat})"); continue
        rows = []
        for ck in cks:
            m, b = load(ck, kind)
            if kind == "ring":
                g = head_geometry(m, D=1, act_pairs=[(0, 1)], obs_offset=2)
            else:
                g = head_geometry(m, D=D)
            cl = classify_run(b["losses"])["registered"]
            rows.append(dict(ck=ck.replace(R + "/", ""), seed=b.get("seed"), solved=cl == "SOLVED", cls=cl,
                             basin=basin(g), heads=g))
        out[lab] = dict(D=D, r=r, rows=rows)
        lines.append(f"## {lab}  (D={D}, per-head rank {r})  SOLVED {sum(x['solved'] for x in rows)}/{len(rows)}")
        for x in rows:
            hs = " | ".join(f"k {h['kappa']:.3f} ind {h['indep']:.2f} rb {h['r_beta']:.2f} off {h['offmap']:.2f} "
                            f"lam {h['lam']:.3f} lp {h['lam_perp']:.3f} rg {h['readgain']:.2f}" for h in x["heads"])
            lines.append(f"  s{x['seed']} {x['cls'][:4]:4s} {x['basin']:8s} {hs}")
    open(f"{HERE}/sweep_final_out.txt", "w").write("\n".join(lines) + "\n")
    json.dump(out, open(f"{HERE}/sweep_final.json", "w"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
