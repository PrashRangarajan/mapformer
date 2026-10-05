"""MapPoPE-Pair == MapPoPE-Flat whose 64 angles are tied in adjacent pairs: copy every shared weight, set the Flat
model's step map and omega to the pairwise model's duplicated rows; logits must agree (output: _out.txt, 0.0)."""
import torch
from mapformer.model_pope_pair import MapFormerWM_PoPEPair
from mapformer.train_variant import VARIANT_MAP
torch.manual_seed(1); P = MapFormerWM_PoPEPair(vocab_size=22).eval()
F_ = VARIANT_MAP["MapPoPE-Flat"](vocab_size=22).eval()
with torch.no_grad():
    for p in P.parameters(): p.add_(0.3 * torch.randn_like(p))
    P.layers[0].pope_delta.copy_(-torch.rand_like(P.layers[0].pope_delta))
    sd = {k: v for k, v in P.state_dict().items() if not k.startswith(("action_to_lie.w_out", "path_integrator"))}
    F_.load_state_dict(sd, strict=False)
    H, nb, r = 2, 32, 2
    Wo = P.action_to_lie.w_out.weight.view(H, nb, r)
    F_.action_to_lie.w_in.weight.copy_(P.action_to_lie.w_in.weight)
    F_.action_to_lie.w_out.weight.copy_(Wo.repeat_interleave(2, dim=1).reshape(H * 2 * nb, r))
    F_.path_integrator.omega.copy_(P.path_integrator.omega.repeat_interleave(2, dim=1))
    tok = torch.randint(0, 22, (3, 128))
    print("max |logit diff| MapPoPE-Pair vs pair-tied MapPoPE-Flat:", (P(tok) - F_(tok)).abs().max().item())
