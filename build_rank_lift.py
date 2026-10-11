"""Build lifted rank-4 checkpoints from stalled rank-2 runs, for the rank escape test (RANK_TESTS_PREREG.md, exp. 2).

For each seed, take `runs/rank_mi/p0/Vanilla_s<S>/Vanilla.pt` (shared rank 2, trained 900 epochs at T = 1024, 0/8 SOLVED)
and write a `Vanilla_r4mi` checkpoint (shared rank 4 built on rank 2's base: `model_rank_perhead.MapFormerWM_r4MatchedInit`)
whose every weight equals the rank-2 model's, with the bottleneck widened by two directions (LoRA-style):
    W_in'  = [W_in ; A]     (4 x d)      A ~ the default nn.Linear init of a (2 x d) layer (the scale rank 4 starts at)
    W_out' = [W_out, 0]     (64 x 4)     the two new output columns start at exactly zero
So the function is unchanged at step 0 (checked in float64: max |logit diff| < 1e-9; in float32 the 4-wide matmul
rounds differently from the 2-wide one and the cumulative sum amplifies it, ~5e-3 in logits), and the new columns get gradient
(A x) from the first step. (A first version with eps = 0.01 on both factors changed logits by up to 0.19: a per-token
change of eps^2 accumulates through the cumulative-sum phase over 511 tokens.) A is drawn from a generator seeded by
7_000_000 + seed (no global RNG use).
Amendment 1 (`--fresh`): the RELEARN reference F -- the same trained rank-2 network with its bottleneck REPLACED by a fresh
rank-4 bottleneck, the exact initial bottleneck `Vanilla_r4mi` draws at seed S (so F starts with trained content weights
and an untrained step map; the function changes). E vs F asks whether the stalled rank-2 step map helps (escape) or
whether only the trained rest of the network does (relearn).
Output (the ckpt_guard layout, so eval_noise_refine reads it): <out>/p0/Vanilla_r4mi_s<S>/Vanilla_r4mi.pt
  python3 -m mapformer.build_rank_lift            # from /home/prashr
"""
import argparse, math, os
import torch

from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"


def lift(sd, seed):
    Win, Wout = sd["action_to_lie.w_in.weight"], sd["action_to_lie.w_out.weight"]      # (2, d), (64, 2)
    g = torch.Generator().manual_seed(7_000_000 + seed)
    d, n_out = Win.shape[1], Wout.shape[0]
    A = (torch.rand(2, d, generator=g) * 2 - 1) / math.sqrt(d)          # nn.Linear default: U(-1/sqrt(fan_in), +)
    out = {k: v.clone() for k, v in sd.items()}
    out["action_to_lie.w_in.weight"] = torch.cat([Win, A.to(Win.dtype)], 0).contiguous()
    out["action_to_lie.w_out.weight"] = torch.cat([Wout, torch.zeros(n_out, 2, dtype=Wout.dtype)], 1).contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{REPO}/runs/rank_mi/p0")
    ap.add_argument("--out", default=f"{REPO}/runs/rank_escape/lifted")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--fresh", action="store_true", help="F: fresh rank-4 bottleneck on the trained network")
    a = ap.parse_args()
    for s in a.seeds:
        src = f"{a.src}/Vanilla_s{s}/Vanilla.pt"
        b = torch.load(src, map_location="cpu", weights_only=False); c = b["config"]
        kw = dict(vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                  grid_size=c["grid_size"])
        if a.fresh:
            torch.manual_seed(s); init = VARIANT_MAP["Vanilla_r4mi"](**kw).state_dict()   # S's initial weights at seed s
            sd4 = {k: v.clone() for k, v in b["model_state_dict"].items()}
            for k in ("action_to_lie.w_in.weight", "action_to_lie.w_out.weight"):
                sd4[k] = init[k].clone()
        else:
            sd4 = lift(b["model_state_dict"], s)
        m4 = VARIANT_MAP["Vanilla_r4mi"](**kw); m4.load_state_dict(sd4)          # strict
        m2 = VARIANT_MAP["Vanilla"](**kw); m2.load_state_dict(b["model_state_dict"])
        m2.eval(); m4.eval()
        g = torch.Generator().manual_seed(123)
        tok = torch.randint(0, c["vocab_size"], (4, 511), generator=g)
        with torch.no_grad():
            diff = float((m4(tok) - m2(tok)).abs().max())                              # float32: rounding, amplified by cumsum
            diff64 = float((m4.double()(tok) - m2.double()(tok)).abs().max())          # float64: the lift itself
        m4.float()
        if not a.fresh:
            assert diff64 < 1e-9, f"lift changed the function: float64 max |logit diff| {diff64}"
        out = f"{a.out}/p0/Vanilla_r4mi_s{s}"; os.makedirs(out, exist_ok=True)
        cfg = dict(c); cfg.update({"lifted_from": src, "lift": ("fresh rank-4 bottleneck (Vanilla_r4mi init at the seed) on the trained network" if a.fresh
                    else "W_in + 2 default-init rows, W_out + 2 zero columns"), "lift_max_logit_diff_fp32": diff, "lift_max_logit_diff_fp64": diff64})
        # 'losses' EMPTY on purpose (build_rank_proj's lesson): this rank-4 model has no training history of its own
        # losses empty (no history of its own); losses_prior is set from source_losses for symmetry with C (audit cosmetic)
        torch.save({"model_state_dict": m4.state_dict(), "losses": [], "source_losses": list(b["losses"]),
                    "variant": "Vanilla_r4mi", "seed": s, "config": cfg}, f"{out}/Vanilla_r4mi.pt")
        print(f"s{s}: {'FRESH bottleneck (F)' if a.fresh else 'lifted rank 2 -> 4'}; max |logit diff| vs the rank-2 source fp64 {diff64:.1e} (fp32 {diff:.1e}); "
              f"source final loss {sum(b['losses'][-45:]) / 45:.4f} -> {out}/Vanilla_r4mi.pt", flush=True)


if __name__ == "__main__":
    main()
