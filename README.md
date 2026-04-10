# MapFormer: Self-Supervised Cognitive Maps with Lie Group Path Integration

Implementation of MapFormer ([Rambaud et al., 2025](https://arxiv.org/abs/2511.19279)) with an Invariant Extended Kalman Filter (InEKF) extension for uncertainty-aware path integration.

## What is MapFormer?

MapFormer is a Transformer that learns a **cognitive map** from sequences of (action, observation) pairs. The key idea: separate **structure** ("where" — updated by actions via SO(n) rotations) from **content** ("what" — updated by observations). Path integration uses a parallel prefix scan over rotation matrices in O(log T), enabling length generalisation that standard Transformers cannot achieve.

## Architecture

| Component | Description |
|-----------|-------------|
| `MapFormer-WM` | Working Memory variant. Generalises RoPE: rotation comes from actions, not token index. Relative positional encoding. |
| `MapFormer-EM` | Episodic Memory variant. Absolute positional encoding with parallel content + structure attention streams. |
| `InEKF` | Invariant Extended Kalman Filter layer. Tracks position uncertainty (covariance) alongside the mean, with landmark-based correction. |

**Baselines** (for comparison):
- `Transformer+RoPE` — fixed index-based rotation (fails OOD)
- `LSTM` — fixed hidden state bottleneck (fails on long sequences)

## Project Structure

```
mapformer/
  __init__.py          # Package exports
  lie_groups.py        # SO(2)/SO(n) exp/log maps, block-diagonal rotations
  prefix_scan.py       # O(log T) parallel prefix product
  environment.py       # 2D GridWorld with non-unique observations
  model.py             # MapFormer-WM and MapFormer-EM
  baselines.py         # Transformer+RoPE and LSTM
  inekf.py             # Invariant EKF layer
  train.py             # Self-supervised training (next-obs prediction)
  evaluate.py          # Evaluation + figure generation
  main.py              # Full experiment pipeline
```

## Quick Start

```bash
pip install -r requirements.txt

# Run with defaults (small scale, ~12 min on MPS)
python3 -m mapformer.main --device mps

# Scale up for paper-quality results (needs GPU)
python3 -m mapformer.main \
  --d-model 128 --n-heads 8 --n-layers 4 \
  --epochs 200 --n-batches 500 \
  --grid-size 16 --n-obs-types 8 \
  --device cuda

# MapFormer only (skip baselines)
python3 -m mapformer.main --skip-baselines --device cuda
```

Figures are saved to `mapformer/figures/`.

## What the Pipeline Produces

1. **Training curves** — cross-entropy loss for all models
2. **Length generalisation** — train on T=64, test on T=128, 256, 512, 1024
3. **Position state PCA** — coloured by true (x, y) grid location
4. **Grid cell autocorrelation** — rate maps + 2D autocorrelation (look for 6 peaks at 60 degrees)

## Key Math

| Quantity | Formula |
|----------|---------|
| Lie algebra element (SO(2)) | A = theta * J, J = [[0,-1],[1,0]] |
| Rotation matrix | R = exp(A) = [[cos, -sin], [sin, cos]] |
| Position state | P_t = M_t * M_{t-1} * ... * M_1 (prefix scan) |
| WM attention | Uses relative P_t * P_s^{-1} |
| EM attention | Additive: content_score + structure_score |
| InEKF predict | Sigma_t = F * Sigma_{t-1} * F^T + Q |
| InEKF update | K = Sigma * H^T * (H * Sigma * H^T + R)^{-1} |

## References

- Rambaud, V., Mascarenhas, S., & Lakretz, Y. (2025). *MapFormer: Self-Supervised Learning of Cognitive Maps with Input-Dependent Positional Embeddings.* [arXiv:2511.19279](https://arxiv.org/abs/2511.19279)
- Barrau, A., & Bonnabel, S. (2017). *The Invariant Extended Kalman Filter as a Stable Observer.* IEEE TAC, 62(4).
