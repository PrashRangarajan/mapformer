# Measured floors for the paper-task OOD protocol

Constant predictors scored on exactly the events `revisit_accuracy` scores (fresh obs_map, env seed 10000; 8 x 32). No model.

| condition | always-blank | best constant (marginal) | 1/vocab | scored n |
|---|---|---|---|---|
| IID  l=128 g=64  pe=0.5 | 0.522 | 0.522 | 0.0476 | 7,342 |
| OOD-d l=64 g=32  pe=0.2 | 0.216 | 0.216 | 0.0476 | 3,212 |
| OOD-s l=256 g=128 pe=0.8 | 0.803 | 0.803 | 0.0476 | 16,663 |
| OOD-s l=512 g=128 pe=0.8 | 0.799 | 0.799 | 0.0476 | 34,793 |
| ext-s l=1024 g=128 pe=0.8 | 0.801 | 0.801 | 0.0476 | 74,919 |
| ext-s l=2048 g=128 pe=0.8 | 0.802 | 0.802 | 0.0476 | 162,768 |
