"""THEORY REVIEW (CPU, exploratory): the loss along the collective route from the |.| start ('absmag', balanced gauge) to the
signed cancelling code. In the balanced gauge z(+) = c e1, z(-) = c e2 and W_out columns carry the angles; the route
z(-)(tau) = c((1 - tau) e2 - tau e1) moves the minus move's angle in EVERY channel from a(-) to -a(+) at once (each channel
crosses zero at its own tau). Same route under |.|: a(-) = |.| never goes negative; at tau = 1 a(-) = a(+) (a pure clock).
Loss is toy_wrap's retrieval loss on fixed walks; beta = exp(log_beta) * softplus(u) weights as at init."""
import sys, math
sys.path.insert(0, "/home/prashr")
import numpy as np, torch
from mapformer.toy_wrap import Toy, walks, loss_acc, set_init
torch.set_num_threads(4)
mv, pos = walks(256, 128, rng=np.random.RandomState(123))
for seed in (10, 11, 12):
    for lb in (0.0, -4.0):
        line = []
        for mode in ("signed", "abs"):
            torch.manual_seed(seed); rng = np.random.RandomState(seed); m = Toy(mode)
            with torch.no_grad():
                m.angles_init = (m.w_out(m.z) * m.omega).detach(); m.angles_init_abs = m.angles_init.abs()
            set_init(m, "absmag", rng)
            with torch.no_grad():
                m.log_beta.fill_(lb); c = float(m.z[0, 0])
            vals = []
            for tau in np.linspace(0, 1, 11):
                with torch.no_grad():
                    m.z[1].zero_(); m.z[1, 1] = c * (1 - tau); m.z[1, 0] = -c * tau
                    l, a = loss_acc(m, mv, pos)
                vals.append(f"{float(l):.2f}/{float(a):.2f}")
            line.append(f"{mode}: " + " ".join(vals))
        print(f"seed {seed} log_beta {lb:+.0f} | loss/acc at tau = 0, 0.1, ..., 1\n   " + "\n   ".join(line), flush=True)
