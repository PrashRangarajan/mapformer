"""CPU bitwise gate for the model scale-fold and the sync-free training loop.
(The GPU must be re-gated with verify_gpu_bitexact.py -- GEMM kernels differ.)"""
import sys, copy
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_efficiency")
import importlib; OM = importlib.import_module("mapformer.model"); OT = importlib.import_module("mapformer.train")
NM = importlib.import_module("mfprop.model"); NT = importlib.import_module("mfprop.train")
from mapformer.environment import GridWorld

def same_params(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))

res = []
# 1. model forward + backward, train mode (dropout on), d_head 64 (pow2) and 48 (fallback)
for d_model in (128, 96):
    torch.manual_seed(0); mo = OM.MapFormerWM(vocab_size=21, d_model=d_model, n_heads=2)
    mn = NM.MapFormerWM(vocab_size=21, d_model=d_model, n_heads=2); mn.load_state_dict(mo.state_dict())
    x = torch.randint(0, 21, (3, 301))
    torch.manual_seed(5); lo = mo(x); lo.square().mean().backward()
    torch.manual_seed(5); ln = mn(x); ln.square().mean().backward()
    g = all(torch.equal(p.grad, q.grad) for p, q in zip(mo.parameters(), mn.parameters()))
    res.append((f"model d_head={d_model//2} logits+grads bitwise", torch.equal(lo, ln) and g))
# 2. training loop, 2 epochs x 3 batches, serial data, cosine; then a no-revisit config
for n_steps, sched in ((64, "cosine"), (48, "linear"), (2, "cosine")):
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=0)
    torch.manual_seed(1); mo = OM.MapFormerWM(vocab_size=env.unified_vocab_size)
    mn = copy.deepcopy(mo)
    torch.manual_seed(2); np.random.seed(2)
    lo = OT.train(mo, env, n_epochs=2, n_batches=3, batch_size=4, n_steps=n_steps, lr=1e-3, schedule=sched, verbose=False)
    torch.manual_seed(2); np.random.seed(2)
    ln = NT.train(mn, env, n_epochs=2, n_batches=3, batch_size=4, n_steps=n_steps, lr=1e-3, schedule=sched, verbose=False)
    res.append((f"train loop n_steps={n_steps} {sched}: losses {lo} vs {ln}, params bitwise",
                lo == ln and same_params(mo, mn)))
# 3. new loop + new model vs old loop + old model (the full proposed stack)
env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=0)
torch.manual_seed(1); mo = OM.MapFormerWM(vocab_size=env.unified_vocab_size)
mn = NM.MapFormerWM(vocab_size=env.unified_vocab_size); mn.load_state_dict(mo.state_dict())
torch.manual_seed(2); np.random.seed(2)
lo = OT.train(mo, env, n_epochs=2, n_batches=3, batch_size=4, n_steps=96, lr=1e-3, schedule="cosine", verbose=False)
torch.manual_seed(2); np.random.seed(2)
ln = NT.train(mn, env, n_epochs=2, n_batches=3, batch_size=4, n_steps=96, lr=1e-3, schedule="cosine", verbose=False)
res.append(("full stack (new loop + new model)", lo == ln and same_params(mo, mn)))
for name, ok in res:
    print("PASS" if ok else "FAIL", name)
