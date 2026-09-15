"""SAMEBLOCK_PREREG.md analysis (written before any final evaluation was read).

    python3 -m mapformer.analyze_sameblock      (from /home/prashr)
"""
import glob, json, os
import numpy as np
import torch

from mapformer.stats_guard import paired, table

REPO = "/home/prashr/mapformer"; RUNS = f"{REPO}/runs/sameblock"
LENS = ("30", "60", "100", "150")


def load():
    R = {}
    for f in glob.glob(f"{RUNS}/*/*_addition.json"):
        j = json.load(open(f)); c = j["config"]
        R[(c["variant"], c["fmt"], c["seed"])] = dict(ev=j["eval"], loss=j["final_loss"], fast=bool(c.get("compile") or c.get("fast_data")),
                                                      pt=f.replace(".json", ".pt"))
    return R


@torch.no_grad()
def role_cosines(pt):
    from mapformer.train_variant import VARIANT_MAP
    from mapformer.environment_addition import D_A, D_B, D_S
    ck = torch.load(pt, map_location="cpu", weights_only=False); c = ck["config"]
    m = VARIANT_MAP[ck["variant"]](34, d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"], max_pos=c["max_pos"])
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    ids = torch.tensor([list(range(D_A, D_A + 10)) + list(range(D_B, D_B + 10)) + list(range(D_S, D_S + 10))])
    d = m.w_out(m.w_in(m.tok(ids))).view(1, 30, m.h, m.nb)[0]
    if ck["variant"].endswith("abs"):
        d = d.abs()
    w = m.omega.abs()
    out = []
    for h in range(m.h):
        a, b, s = d[0:10, h].mean(0), d[10:20, h].mean(0), d[20:30, h].mean(0)
        cos = lambda x, y: float((x * y * w[h]).sum() / ((x * x * w[h]).sum().sqrt() * (y * y * w[h]).sum().sqrt() + 1e-9))
        out.append((cos(s, a), cos(s, b), cos(a, b)))
    return out


def main():
    R = load()
    L = ["# SAMEBLOCK raw report (SAMEBLOCK_PREREG.md; mechanical)\n", "## Every run\n",
         "| arm | format | seed | code | final loss | exact 30 / 60 / 100 / 150 | per-digit 60 / 100 / 150 |", "|---|---|---|---|---|---|---|"]
    for k in sorted(R):
        r = R[k]; e = r["ev"]
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {'fast' if r['fast'] else 'seed-0 code'} | {r['loss']:.4f} | "
                 + " / ".join(f"{e[x]['exact']:.3f}" for x in LENS) + " | " + " / ".join(f"{e[x]['digit']:.3f}" for x in LENS[1:]) + " |")
    g2 = lambda k: R[k]["ev"]["30"]["exact"] >= 0.9
    ctl = [R[k]["ev"]["100"]["exact"] for k in R if k[0] == "ChoPos_coupled" and k[1] == "role"]
    G1 = len(ctl) > 0 and float(np.mean(ctl)) >= 0.9
    L += ["", "## Gates\n", f"- **G1** coupled (role) mean exact at 100 digits = {np.mean(ctl) if ctl else float('nan'):.3f} over {len(ctl)} seeds -> **{'PASS' if G1 else 'FAIL'}**"]
    for arm in ("ChoPos_signed", "ChoPos_abs", "ChoPos_rope", "ChoPos_coupled", "ChoPos_nope"):
        ks = sorted(k for k in R if k[0] == arm and k[1] == "role")
        L.append(f"- **G2** {arm} (role): " + ", ".join(f"s{k[2]} {'pass' if g2(k) else 'FAIL'} ({R[k]['ev']['30']['exact']:.3f})" for k in ks))
    L += ["", "## Contrasts at 100 digits, role format, paired by seed (arm-seeds failing G2 enter as their raw score and are flagged)\n"]
    cs = []
    for a_, b_, lab in (("ChoPos_signed", "ChoPos_abs", "P1 signed - abs"), ("ChoPos_signed", "ChoPos_rope", "P2 signed - rope"),
                        ("ChoPos_signed", "ChoPos_coupled", "signed - coupled (no prediction)")):
        sa = {k[2]: R[k]["ev"]["100"]["exact"] for k in R if k[0] == a_ and k[1] == "role"}
        sb = {k[2]: R[k]["ev"]["100"]["exact"] for k in R if k[0] == b_ and k[1] == "role"}
        common = sorted(set(sa) & set(sb))
        flags = [s for s in common if not (g2((a_, "role", s)) and g2((b_, "role", s)))]
        if len(common) >= 2:
            c = paired({s: sa[s] for s in common}, {s: sb[s] for s in common}, f"{lab} (seeds {common}; G2 fails in seeds {flags})")
            cs.append(c)
        else:
            L.append(f"- {lab}: only {len(common)} common seed(s): " + ", ".join(f"s{s}: {sa[s]:.3f} vs {sb[s]:.3f}" for s in common))
    if cs:
        L.append(table(cs))
    if not G1:
        L.append("\n**G1 failed: by the pre-registration no contrast above is read.**")
    L += ["", "## Mechanism (descriptive): per-head cosine of mean role increments (s vs a, s vs b, a vs b)\n"]
    for k in sorted(R):
        if k[0] in ("ChoPos_signed", "ChoPos_abs"):
            L.append(f"- {k[0]} {k[1]} s{k[2]}: " + "; ".join(f"h{i} ({x:+.2f}, {y:+.2f}, {z:+.2f})" for i, (x, y, z) in enumerate(role_cosines(R[k]['pt']))))
    open(f"{REPO}/SAMEBLOCK_RAW.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
