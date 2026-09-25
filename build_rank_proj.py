"""Build rank-2 models from solved r=4 models, for the warm-start stability test
(RANK_PROJ_PREREG.md).

For each seed, take the trained r=4 checkpoint, project its action latent onto the top
two (uncentred) singular directions over the vocabulary, and write an r=2 (`Vanilla`)
checkpoint whose every other weight is the r=4 model's. Same method as
probe_rank_projection.py (s6 0.9995, s2 0.9896 at T=1024).

Output layout is the ckpt_guard one, so eval_noise_refine / eval_rank_strata read it:
<out>/p0/Vanilla_s<S>/Vanilla.pt
"""
import argparse, os
import torch

from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{REPO}/runs/rank_matched_e900/p0")
    ap.add_argument("--out", default=f"{REPO}/runs/rank_proj")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    a = ap.parse_args()
    for s in a.seeds:
        src = f"{a.src}/Vanilla_r4_s{s}/Vanilla_r4.pt"
        b = torch.load(src, map_location="cpu", weights_only=False); c = b["config"]
        kw = dict(vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                  n_layers=c["n_layers"], grid_size=c["grid_size"])
        sd4 = b["model_state_dict"]
        E = sd4["token_emb.weight"]; Win = sd4["action_to_lie.w_in.weight"]
        Wout = sd4["action_to_lie.w_out.weight"]
        H = E @ Win.T                                   # (V, 4) latents over the vocabulary
        S = torch.linalg.svdvals(H); _, _, Vh = torch.linalg.svd(H, full_matrices=False)
        P = Vh[:2].T                                    # (4, 2)
        sd = {k: v.clone() for k, v in sd4.items()}
        sd["action_to_lie.w_in.weight"] = (P.T @ Win).contiguous()
        sd["action_to_lie.w_out.weight"] = (Wout @ P).contiguous()
        m = VARIANT_MAP["Vanilla"](**kw); m.load_state_dict(sd)       # strict
        D, D2 = H @ Wout.T, (H @ P @ P.T) @ Wout.T
        rel = float((D - D2).norm() / D.norm()); top2 = float((S[:2] ** 2).sum() / (S ** 2).sum())
        out = f"{a.out}/p0/Vanilla_s{s}"; os.makedirs(out, exist_ok=True)
        cfg = dict(c); cfg.update({"projected_from": src, "proj_top2_energy": top2, "proj_delta_rel_err": rel})
        # "losses" is EMPTY on purpose: this r=2 model was never trained. Copying the r=4
        # source's curve here made the projected checkpoint (and, via --init-from, the
        # trained arm's losses_prior) carry an r=4 run's history under variant "Vanilla",
        # which classify()/experiment_audit/compare_checkpoints would read as r=2's own.
        torch.save({"model_state_dict": m.state_dict(), "losses": [],
                    "source_losses": list(b["losses"]),
                    "variant": "Vanilla", "seed": s, "config": cfg}, f"{out}/Vanilla.pt")
        print(f"s{s}: latent sv {[round(float(x), 3) for x in S]} top2 energy {top2:.5f} "
              f"Delta rel err {rel:.4f} -> {out}/Vanilla.pt", flush=True)


if __name__ == "__main__":
    main()
