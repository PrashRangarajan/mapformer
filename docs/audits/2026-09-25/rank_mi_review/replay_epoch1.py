"""Review check: replay EPOCH 1 of every rank_mi run with the current code and compare the
first-epoch loss with the stored checkpoint's losses[0] (bitwise). Same schedule length
(900 epochs), same data stream, same init; only the epoch loop is cut after one epoch.
Also: construction-level matched-init checks and initial Delta scale per arm."""
import builtins, sys, json, torch, numpy as np
import importlib
TR = importlib.import_module('mapformer.train')
import mapformer.train_variant as TV
from mapformer.environment import GridWorld

SCR = "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/review"
REPO = "/home/prashr/mapformer"
_range = builtins.range
assert TR.__name__ == 'mapformer.train' and TR.train.__globals__ is vars(TR)
TR.range = lambda *a: _range(1) if a == (900,) else _range(*a)

def build(variant, s):
    torch.manual_seed(s); np.random.seed(s)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=s,
                    action_mode="translate", obs_mode="allo", boundary="torus",
                    score_moves_only=False, action_record="commanded", n_headings=4, heading_noise=0.0)
    m = TV.VARIANT_MAP[variant](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    return env, m

def main():
    out = {"construct": {}, "replay": {}}
    arms = ["Vanilla", "Vanilla_r2ph", "Vanilla_r4mi"]
    for s in range(8):
        ms = {v: build(v, s)[1].state_dict() for v in arms}
        shared = [k for k in ms["Vanilla"] if not k.startswith("action_to_lie")]
        d = {f"{v} vs Vanilla": max(float((ms[v][k] - ms["Vanilla"][k]).abs().max()) for k in shared) for v in arms[1:]}
        d["r2ph.w_in == r4mi.w_in"] = float((ms["Vanilla_r2ph"]["action_to_lie.w_in.weight"] - ms["Vanilla_r4mi"]["action_to_lie.w_in.weight"]).abs().max())
        d["keys equal"] = all(set(k for k in ms[v] if not k.startswith("action_to_lie")) == set(shared) for v in arms)
        # initial Delta std per arm on the 21 token embeddings
        for v in arms:
            env, m = build(v, s)
            with torch.no_grad():
                dl = m.action_to_lie(m.token_emb.weight[None])[0]   # (V, nh, nb)
            d[f"{v} delta_std"] = float(dl.std())
            d[f"{v} n_params"] = sum(p.numel() for p in m.parameters())
        out["construct"][s] = d
        print(s, json.dumps(d))

    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    todo = [(f"{REPO}/runs/rank_mi/p0/{v}_s{s}/{v}.pt", v, s) for v in arms for s in range(8)]
    todo.append((f"{REPO}/runs/rank_mi_repro/p0/Vanilla_s3/Vanilla.pt", "Vanilla", 3))
    todo.append((f"{REPO}/runs/rank_perhead_pilot/p0/Vanilla_s0/Vanilla.pt", "Vanilla", 0))
    for ck, v, s in todo:
        stored = torch.load(ck, map_location="cpu", weights_only=False)["losses"]
        od = f"{SCR}/replay/{v}_s{s}"
        sys.argv = ["x", "--variant", v, "--seed", str(s), "--epochs", "900", "--lr", "1e-3", "--n-batches", "98",
                    "--batch-size", "16", "--n-steps", "1024", "--n-layers", "1", "--n-heads", "2", "--d-model", "128",
                    "--n-landmarks", "0", "--schedule", "cosine", "--data-workers", "3", "--device", dev, "--output-dir", od]
        TV.main()
        assert len(torch.load(f"{od}/{v}.pt", map_location='cpu', weights_only=False)['losses']) == 1, 'patch failed'
        mine = torch.load(f"{od}/{v}.pt", map_location="cpu", weights_only=False)["losses"]
        r = {"stored": stored[0], "replay": mine[0], "equal": stored[0] == mine[0], "ckpt": ck}
        out["replay"][ck] = r
        print("REPLAY", v, s, r, flush=True)
    json.dump(out, open(f"{SCR}/replay_epoch1.json", "w"), indent=1, default=str)
    print("DONE")

if __name__ == '__main__':
    main()
