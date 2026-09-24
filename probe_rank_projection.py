"""(Committed 2026-09-23; written by the rank review, re-run by the main session: s6 0.9995, s2 0.9896.)
Existence check: can an r=2 model reproduce a trained r=4 model's function?
Project the r=4 latent onto its top-2 (uncentred) singular directions, load into a
Vanilla (r=2) model, evaluate with eval_noise_refine.evaluate on the held-out map."""
import sys, json, time, numpy as np, torch
torch.set_num_threads(8)
from mapformer.train_variant import VARIANT_MAP
from mapformer.environment import GridWorld
from mapformer.eval_noise_refine import evaluate
R = "/home/prashr/mapformer/runs"   # absolute: -m resolves relative paths to the parent dir
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REF = json.load(open("/home/prashr/mapformer/RANK_MATCHED.json"))
ref = {s: a for s, a, n in REF["0.0|Vanilla_r4|1024"]}
dev = torch.device("cpu")
for run, seeds in (("rank_matched", [6, 5, 2, 1, 0]), ("rank_sweep", [0, 1])):
    for s in seeds:
        b = torch.load(f"{R}/{run}/p0/Vanilla_r4_s{s}/Vanilla_r4.pt", map_location="cpu", weights_only=False)
        c = b["config"]
        kw = dict(vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                  n_layers=c["n_layers"], grid_size=c["grid_size"])
        m4 = VARIANT_MAP["Vanilla_r4"](**kw); m4.load_state_dict(b["model_state_dict"]); m4.eval()
        E = m4.token_emb.weight.detach()                    # (V, d)
        Win = m4.action_to_lie.w_in.weight.detach()          # (4, d)
        Wout = m4.action_to_lie.w_out.weight.detach()        # (H*nb, 4)
        H = E @ Win.T                                        # (V, 4) latents
        # weight rows by how often each token occurs: 4 actions + 17 obs, one of each per step;
        # use the actual Delta_out energy in the SVD so the projection keeps what matters
        D = H @ Wout.T                                       # (V, H*nb)
        sv_lat = torch.linalg.svdvals(H)
        _, S, Vh = torch.linalg.svd(H, full_matrices=False)
        P = Vh[:2].T                                         # (4, 2)
        sd = {k: v.clone() for k, v in b["model_state_dict"].items()}
        sd["action_to_lie.w_in.weight"] = (P.T @ Win).contiguous()
        sd["action_to_lie.w_out.weight"] = (Wout @ P).contiguous()
        m2 = VARIANT_MAP["Vanilla"](**kw); m2.load_state_dict(sd); m2.eval()
        D2 = (H @ P @ P.T) @ Wout.T
        rel = float((D - D2).norm() / D.norm())
        env = GridWorld(size=c["grid_size"], n_obs_types=c.get("n_obs_types", 16),
                        p_empty=c.get("p_empty", 0.5), seed=10000)
        t0 = time.time()
        T = 1024
        a4, n4 = evaluate(m4, env, T, NT, 0.0, dev, seed=1234 + s)
        a2, n2 = evaluate(m2, env, T, NT, 0.0, dev, seed=1234 + s)
        frac = (S[:2] ** 2).sum() / (S ** 2).sum()
        print(f"{run} r4 s{s}: latent sv {[round(float(x),3) for x in sv_lat]} top2 energy {float(frac):.5f} "
              f"| Delta rel err after rank-2 proj {rel:.4f} | T=1024 acc r4 {a4:.4f} (json {ref.get(s) if run=='rank_matched' else 'n/a'}) "
              f"-> projected r2 {a2:.4f}; nll {n4:.3f} -> {n2:.3f}  [{time.time()-t0:.0f}s, {NT} trials]", flush=True)
