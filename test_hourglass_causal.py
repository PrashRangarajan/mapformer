"""Correctness tests for the Hourglass-MapFormer.

1. CAUSALITY: perturbing token t must not change any logit at position < t.
   This is the load-bearing property of the causal shift-then-pool shortening;
   if it fails, the model is cheating by seeing the future through the pool.
2. SHAPES / FORWARD: forward runs and returns (B, L, vocab).
3. FLAT CONTROL: HourglassFlat3 (k=1) is also causal and same param count.
"""

import torch
from mapformer.model_hourglass import (
    MapFormerWM_Hourglass_k2, MapFormerWM_Hourglass_k4,
    MapFormerWM_Hourglass_k2_deep, MapFormerWM_HourglassFlat3,
)
from mapformer.model import MapFormerWM


def _causality(model, L=63, vocab=20, seed=0, k=None):
    """Return (max_future_leak, min_response).

    max_future_leak: largest change at positions STRICTLY BEFORE a perturbed
        token (must be ~0 for a causal model).
    min_response: smallest change at the perturbed position and after (must be
        > 0, else the model is degenerate/dead and the leak test is vacuous).
    Perturbs interior positions AND, if k is given, every group-boundary pair
    (last-of-group b+k-1 must not move first-of-group b) — the exact indices a
    shift bug breaks.
    """
    torch.manual_seed(seed)
    model.eval()
    base = torch.randint(0, vocab, (1, L))
    with torch.no_grad():
        logits0 = model(base)
    positions = {7 % L, 20 % L, 33 % L, L // 2, L - 1}
    if k and k > 1:
        for b in range(k, L, k):            # first-of-group indices
            positions.add(min(b + k - 1, L - 1))  # last-of-group (a shift bug leaks to b)
    max_leak, min_resp = 0.0, float("inf")
    for t in sorted(positions):
        pert = base.clone()
        pert[0, t] = (pert[0, t].item() + 1) % vocab
        with torch.no_grad():
            logits1 = model(pert)
        if t > 0:
            max_leak = max(max_leak, (logits1[:, :t] - logits0[:, :t]).abs().max().item())
        min_resp = min(min_resp, (logits1[:, t:] - logits0[:, t:]).abs().max().item())
    return max_leak, min_resp


def main():
    vocab = 20
    configs = {
        "Hourglass_k2": MapFormerWM_Hourglass_k2,
        "Hourglass_k4": MapFormerWM_Hourglass_k4,
        "Hourglass_k2_deep": MapFormerWM_Hourglass_k2_deep,
        "HourglassFlat3": MapFormerWM_HourglassFlat3,
    }
    for name, cls in configs.items():
        m = cls(vocab_size=vocab, d_model=64, n_heads=2)
        k = m.k
        # forward / shape
        out = m(torch.randint(0, vocab, (4, 63)))
        assert out.shape == (4, 63, vocab), (name, out.shape)
        # sweep lengths covering every pad residue AND the no-pad branch (pad=0)
        worst_leak, worst_resp = 0.0, float("inf")
        for L in (60, 61, 62, 63, 64):
            leak, resp = _causality(m, L=L, vocab=vocab, k=k)
            worst_leak = max(worst_leak, leak)
            worst_resp = min(worst_resp, resp)
        # also confirm the coarse-OFF skip path is independently causal
        m._use_coarse = False
        leak_off, _ = _causality(m, L=63, vocab=vocab, k=k)
        m._use_coarse = True
        n_params = sum(p.numel() for p in m.parameters())
        cc = m.coarse_contribution(torch.randint(0, vocab, (4, 63)))
        assert worst_leak < 1e-5, (name, "FUTURE LEAK", worst_leak)
        assert worst_resp > 1e-4, (name, "DEGENERATE (no response)", worst_resp)
        assert leak_off < 1e-5, (name, "coarse-off leak", leak_off)
        print(f"{name:22s} params={n_params:>7,d}  max_future_leak={worst_leak:.2e}  "
              f"min_response={worst_resp:.2e}  coarse_off_leak={leak_off:.2e}  "
              f"coarse_contrib={cc:.3f}  OK")

    # param-count parity check: Hourglass_k2 vs a flat 3-layer MapFormerWM
    hg = MapFormerWM_Hourglass_k2(vocab_size=vocab, d_model=64, n_heads=2)
    flat = MapFormerWM(vocab_size=vocab, d_model=64, n_heads=2, n_layers=3)
    ph = sum(p.numel() for p in hg.parameters())
    pf = sum(p.numel() for p in flat.parameters())
    print(f"\nparam parity: Hourglass_k2={ph:,}  flat-3-layer MapFormerWM={pf:,}  "
          f"(diff={ph - pf})")
    print("\nAll shape/causality assertions passed." )


if __name__ == "__main__":
    main()
