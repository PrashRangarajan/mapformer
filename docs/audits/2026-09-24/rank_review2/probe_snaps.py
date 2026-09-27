"""Action-code geometry + T=1024 strata accuracy for snapshot checkpoints and reference runs."""
import sys, glob, os, numpy as np, torch
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.eval_rank_strata import evaluate as strata_eval
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
D = "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/review_rank2"
R = "/home/prashr/mapformer/runs"
envg = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=1)
envh = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
def build(v, sd):
    m = VARIANT_MAP[v](vocab_size=envg.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(sd); return m.to(dev).eval()
def geom(m, ref=None):
    with torch.no_grad():
        ids = torch.arange(envg.unified_vocab_size, device=dev)
        Z = m.action_to_lie.w_in(m.token_emb(ids)); Dl = m.action_to_lie.w_out(Z)
    Z = Z.cpu().numpy(); Dl = Dl.cpu().numpy()
    A = Z[envg.action_offset:envg.action_offset + envg.N_ACTIONS]; O = Z[envg.obs_offset:]
    dl = [envg.ACTION_DELTAS[i] for i in range(envg.N_ACTIONS)]
    pairs = [(i, j) for i in range(4) for j in range(i+1, 4) if dl[i][0] == -dl[j][0] and dl[i][1] == -dl[j][1]]
    opp = np.mean([np.linalg.norm(A[i]+A[j]) / ((np.linalg.norm(A[i])+np.linalg.norm(A[j]))/2) for i, j in pairs])
    u, v = A[pairs[0][0]], A[pairs[1][0]]; cs = abs(u@v)/(np.linalg.norm(u)*np.linalg.norm(v))
    on = np.linalg.norm(O, axis=1).mean()/np.linalg.norm(A, axis=1).mean()
    # per-token angle increments (Delta) for actions, relative change vs reference
    DA = Dl[envg.action_offset:envg.action_offset+4]
    rel = None if ref is None else float(np.linalg.norm(DA-ref)/np.linalg.norm(ref))
    om = m.path_integrator.omega.detach().cpu().numpy().reshape(-1)      # (H*nb,)
    ph = 64 * DA * om[None, :]; ph = np.angle(np.exp(1j*ph))            # 64-step phase per action/block, wrapped
    global WRAPERR, NORMS; WRAPERR = float(np.abs(ph).mean()/np.pi); NORMS = np.linalg.norm(DA, axis=1)
    # opposition measured on Delta (basis-free output of the bottleneck)
    oppD = np.mean([np.linalg.norm(DA[i]+DA[j]) / ((np.linalg.norm(DA[i])+np.linalg.norm(DA[j]))/2) for i, j in pairs])
    return opp, cs, on, oppD, rel, DA
def row(tag, v, sd, ref=None, ntr=40, seed=1234):
    m = build(v, sd); opp, cs, on, oppD, rel, DA = geom(m, ref)
    st = strata_eval(m, envh, 1024, ntr, seed, dev)
    print(f"{tag:28s} opp {opp:.3f} |cos| {cs:.3f} obs/act {on:.3f} oppDelta {oppD:.3f} relDeltaVsStart {'' if rel is None else f'{rel:.3f}'} wrapErr {WRAPERR:.3f} |D| {np.round(NORMS,2)} | "
          f"acc all {st['all']['acc']:.3f} nll {st['all']['nll']:.3f} <128 {st['plain_lag<128']['acc']:.3f} >=128 {st['plain_lag>=128']['acc']:.3f} wrap {st['wrap']['acc']:.3f}", flush=True)
    return DA
which = sys.argv[2] if len(sys.argv) > 2 else "all"
if which in ("ref", "all"):
    print("== references: from-scratch r=2 (e900 / e900c) and r=4 e900 ==")
    for tag in ("rank_matched_e900", "rank_matched_e900c"):
        for s in range(8):
            b = torch.load(f"{R}/{tag}/p0/Vanilla_s{s}/Vanilla.pt", map_location="cpu", weights_only=False)
            row(f"{tag[-5:]} r2 s{s}", "Vanilla", b["model_state_dict"], seed=1234+s)
for s in (0, 6):
    if which not in (f"s{s}", "all"): continue
    files = sorted(glob.glob(f"{D}/snap_proj_s{s}/ep*.pt"))
    if not files: continue
    print(f"== snapshots, trainable r=2 from projection, seed {s} ==")
    ref = None
    for f in files:
        b = torch.load(f, map_location="cpu", weights_only=False)
        l = b["losses"]; ep = b["epoch"]
        tail = np.mean(l[-5:]) if l else float('nan')
        DA = row(f"s{s} ep{ep:4d} trainloss~{tail:.3f}", "Vanilla", b["model_state_dict"], ref, seed=1234+s)
        if ref is None: ref = DA
    b = torch.load(f"{R}/rank_proj_train/p0/Vanilla_s{s}/Vanilla.pt", map_location="cpu", weights_only=False)
    row(f"s{s} ep 900 (committed final)", "Vanilla", b["model_state_dict"], ref, seed=1234+s)
