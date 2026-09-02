"""
Training loop for MapFormer and baseline models.

Self-supervised objective: predict next observation given the interleaved
token stream s = (a1, o1, a2, o2, ..., aT, oT).

Loss is computed ONLY on observation predictions (after action tokens),
since actions are random and unpredictable.

Matches paper (Rambaud et al., 2025, Appendix B):
- AdamW optimizer, lr=3e-4, weight_decay=0.05
- Linear learning rate decay
- Batch size 128, 200K sequences total
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from typing import Optional

from .environment import GridWorld


def train(
    model: nn.Module,
    env: GridWorld,
    n_epochs: int = 50,
    lr: float = 3e-4,
    batch_size: int = 128,
    n_steps: int = 128,
    n_batches: int = 100,
    device: str = "cpu",
    verbose: bool = True,
    weight_decay: float = 0.05,
    p_action_noise: float = 0.0,
    p_transition_noise: float = 0.0,
    aux_coef: float = 0.0,
    schedule: str = "linear",
    data_workers: int = 0,
) -> list[float]:
    """Full training loop with observation-only loss.

    If ``aux_coef > 0`` and the model exposes ``prediction_error_loss()``
    (e.g. PC, GridL15PC), the auxiliary loss is added to the next-token
    loss as ``total = next_token_loss + aux_coef * model.prediction_error_loss()``.

    Returns:
        List of per-epoch average losses (next-token loss only; aux is
        included in the gradient step but logged separately when present).
    """
    has_aux = aux_coef > 0.0 and hasattr(model, "prediction_error_loss")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = n_epochs * n_batches
    if schedule == "cosine":
        # 5% warmup then cosine to 10% of peak. The "linear" default is
        # LinearLR(1.0 -> 0.0), which decays from STEP ONE with no warmup: on a
        # plateau-then-cliff landscape a run can never escape the plateau late, so
        # the budget measures "did the transition fire early", not "can this model
        # solve the task". Switching to this schedule moved one MiniWorld arm from
        # 0.448 to 0.990 on the SAME task and INVERTED the sign of the headline
        # effect (standing rule 10). Default kept as linear so every previously
        # trained checkpoint reproduces.
        import math as _math
        warm = max(1, int(0.05 * total_steps))
        def _lr(step):
            if step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, total_steps - warm)
            return 0.1 + 0.9 * 0.5 * (1.0 + _math.cos(_math.pi * min(p, 1.0)))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr)
    else:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
        )

    criterion = nn.CrossEntropyLoss()

    # Optional parallel trajectory generation. Generation is 79-95% of an
    # epoch at the standard config and is single-threaded, so this is where
    # the wall time is. OFF by default: the parallel path seeds each batch by
    # its INDEX and so draws a DIFFERENT sample from the same generator than
    # the serial path does. Same distribution, not the same stream -- a
    # parallel run therefore will not reproduce a stored serial checkpoint.
    wants_positions = hasattr(model, "_batch_positions")
    gen = None
    if data_workers > 0:
        from .data_parallel import ParallelBatchGenerator
        gen = ParallelBatchGenerator(
            env, batch_size, n_steps, n_workers=data_workers,
            base_seed=torch.initial_seed() % (2 ** 31),
            p_transition_noise=p_transition_noise,
            want_locations=wants_positions)

    losses = []
    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for _ in range(n_batches):
            # tokens: (B, 2*n_steps) interleaved [a1, o1, a2, o2, ...]
            # obs_mask: True at observation positions
            # revisit_mask: True at observation positions AT REVISITED cells
            if gen is not None:
                tokens, obs_mask, revisit_mask, all_locations = gen.next_batch()
            else:
                tokens, obs_mask, revisit_mask, all_locations = env.generate_batch(
                    batch_size, n_steps, p_transition_noise=p_transition_noise,
                )
            tokens = tokens.to(device)
            revisit_mask = revisit_mask.to(device)

            # Stash ground-truth positions on the model for variants whose
            # auxiliary loss needs them (e.g., DoG aux on Level15_DoG).
            # Position at input index 2t+1 (obs token at step t) = location[t].
            # Action positions (even indices) are placeholders; aux losses
            # mask them out. Vectorised: build on CPU as numpy, transfer once.
            if wants_positions:
                import numpy as _np
                B = len(all_locations)
                L_in = 2 * n_steps - 1
                loc_arr = _np.asarray(all_locations, dtype=_np.float32)  # (B, n_steps, 2)
                positions = torch.zeros(B, L_in, 2, dtype=torch.float32)
                n_odd = L_in // 2  # = n_steps - 1 for odd L_in
                positions[:, 1::2] = torch.from_numpy(loc_arr[:, :n_odd, :])
                model._batch_positions = positions.to(device, non_blocking=True)

            # Optional action noise: corrupt random action tokens at even positions
            if p_action_noise > 0:
                # Actions are at even positions (0, 2, 4, ...)
                even_mask = torch.zeros_like(tokens, dtype=torch.bool)
                even_mask[:, 0::2] = True
                noise_mask = (torch.rand_like(tokens, dtype=torch.float) < p_action_noise) & even_mask
                random_actions = torch.randint(0, env.N_ACTIONS, tokens.shape, device=device)
                tokens = torch.where(noise_mask, random_actions, tokens)

            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            # Paper: "predict observation each time it comes back to a previously
            # visited location" — loss only on REVISITS, not first visits
            target_mask = revisit_mask[:, 1:]

            logits = model(input_tokens)

            # Skip batches with no revisits (rare at start of training)
            if target_mask.sum() == 0:
                continue

            logits_masked = logits[target_mask]
            targets_masked = target_tokens[target_mask]

            loss = criterion(logits_masked, targets_masked)
            if has_aux:
                aux = model.prediction_error_loss()
                total_loss = loss + aux_coef * aux
            else:
                total_loss = loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            dt = time.time() - t0
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.2e} | {dt:.1f}s")

    if gen is not None:
        gen.close()

    return losses
