# READING

**Why the metrics differ, and which claim survives which.**
- The ORDERING is identical under every metric: MapPoPE-1L > MapWM-1L ~ MapEM-1L > CoPE/PoPE/RoPE >
  n-gram. That part of the paper is robust.
- "MapFormers with a single layer perfectly generalize OOD" survives NO standard metric. Under the
  published set-prediction criterion (B2, correct only if every step of the sequence is correct)
  MapWM-1L goes 0.672 -> 0.003 and MapPoPE-1L 0.661 -> 0.042 from the training cell to L128 D12.
  Under Hewitt's bracket-closing memory (A2) they go 0.992 -> 0.638 and 0.994 -> 0.719.
- What DOES survive, and is the paper's real claim: a ONE-LAYER MapFormer learns the stack at the
  training distribution (A2 0.98-0.99) where one-layer index models do not (0.63-0.65). Two-layer
  index models do learn it IID (0.90-0.92), exactly as Yao et al. 2021 predict -- and lose it OOD
  (0.57-0.65). The paper's own F1 hides the 1-layer distinction: it gives RoPE-1L 0.912 IID.
- Only on the paper's F1 (C1) does the no-stack n-gram outscore every index model (0.857 at L128 D12
  vs 0.472-0.704). On every stack-sensitive metric the n-gram is at chance, as it should be.
- Aggregation matters as much as the metric: the same probabilities give RoPE-1L 0.871 under A1
  (mean over positions) and 0.627 under A2 (mean over distances), because long-distance cases are
  rare and A1 lets the abundant easy ones dominate.

**Caveat on the B rows.** Suzgun et al. / Bhattamishra et al. train with per-symbol sigmoids and an
MSE loss against k-hot valid-set labels, then threshold at 0.5. These models are trained with
softmax next-token cross-entropy, as MapFormer does, so they are not optimised for set prediction;
the threshold-free form used here is the closest comparable, and the absolute values are not
comparable to published NCP numbers.

# Dyck-2 under every standard metric in the literature

Same checkpoints and same 512 evaluation sequences per cell throughout; cells are seed means (n=8 per model, 1 for the n-grams). Trained on L=32 D=4 only. Definitions and citations in `eval_dyck_literature.py`.

## A1 close accuracy, mean over positions -- Yao et al. 2021 (ACL), following Hewitt et al. 2020. p(legal closer) renormalised over closing brackets. Chance 0.500.

| model | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| n-gram k=1 | 0.796 | 0.789 | 0.771 | 0.768 |
| n-gram k=3 | 0.871 | 0.862 | 0.814 | 0.832 |
| MapPoPE-1L | 0.998 | 0.989 | 0.997 | 0.976 |
| MapWM-1L | 0.997 | 0.961 | 0.974 | 0.924 |
| MapEM-1L | 0.993 | 0.956 | 0.976 | 0.931 |
| PoPE-1L | 0.872 | 0.847 | 0.825 | 0.818 |
| PoPE-2L | 0.959 | 0.846 | 0.868 | 0.800 |
| RoPE-1L | 0.871 | 0.820 | 0.819 | 0.793 |
| RoPE-2L | 0.963 | 0.853 | 0.861 | 0.812 |
| CoPE-1L | 0.861 | 0.833 | 0.820 | 0.809 |
| CoPE-2L | 0.951 | 0.867 | 0.862 | 0.826 |

## A2 close accuracy, mean over DISTANCES j -- Hewitt et al. 2020's 'bracket-closing memory'. Chance 0.500.

| model | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| n-gram k=1 | 0.531 | 0.511 | 0.531 | 0.508 |
| n-gram k=3 | 0.562 | 0.524 | 0.563 | 0.516 |
| MapPoPE-1L | 0.994 | 0.793 | 0.993 | 0.719 |
| MapWM-1L | 0.992 | 0.772 | 0.932 | 0.638 |
| MapEM-1L | 0.980 | 0.701 | 0.946 | 0.620 |
| PoPE-1L | 0.646 | 0.585 | 0.627 | 0.551 |
| PoPE-2L | 0.919 | 0.623 | 0.765 | 0.578 |
| RoPE-1L | 0.627 | 0.554 | 0.612 | 0.535 |
| RoPE-2L | 0.914 | 0.623 | 0.740 | 0.574 |
| CoPE-1L | 0.629 | 0.579 | 0.620 | 0.549 |
| CoPE-2L | 0.899 | 0.652 | 0.752 | 0.591 |

## A3 the same comparison as 0/1 -- is the legal closer ranked above the illegal one. Chance 0.500.

| model | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| n-gram k=1 | 0.797 | 0.791 | 0.769 | 0.770 |
| n-gram k=3 | 0.874 | 0.864 | 0.818 | 0.832 |
| MapPoPE-1L | 1.000 | 0.991 | 0.999 | 0.978 |
| MapWM-1L | 0.999 | 0.965 | 0.978 | 0.928 |
| MapEM-1L | 0.997 | 0.962 | 0.982 | 0.938 |
| PoPE-1L | 0.940 | 0.904 | 0.861 | 0.864 |
| PoPE-2L | 0.980 | 0.936 | 0.898 | 0.891 |
| RoPE-1L | 0.936 | 0.851 | 0.858 | 0.822 |
| RoPE-2L | 0.979 | 0.887 | 0.888 | 0.856 |
| CoPE-1L | 0.919 | 0.889 | 0.857 | 0.854 |
| CoPE-2L | 0.972 | 0.886 | 0.891 | 0.853 |

## B1 valid-set prediction, per step -- Gers & Schmidhuber 2001 / Suzgun et al. 2019 / Bhattamishra et al. 2020 / Ebrahimi et al. 2020, threshold-free form.

| model | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| n-gram k=1 | 0.828 | 0.820 | 0.786 | 0.785 |
| n-gram k=3 | 0.894 | 0.882 | 0.831 | 0.843 |
| MapPoPE-1L | 0.982 | 0.907 | 0.969 | 0.901 |
| MapWM-1L | 0.981 | 0.885 | 0.923 | 0.840 |
| MapEM-1L | 0.979 | 0.870 | 0.928 | 0.873 |
| PoPE-1L | 0.898 | 0.527 | 0.818 | 0.451 |
| PoPE-2L | 0.924 | 0.253 | 0.471 | 0.202 |
| RoPE-1L | 0.894 | 0.411 | 0.707 | 0.377 |
| RoPE-2L | 0.924 | 0.268 | 0.480 | 0.210 |
| CoPE-1L | 0.886 | 0.626 | 0.798 | 0.605 |
| CoPE-2L | 0.911 | 0.452 | 0.459 | 0.285 |

## B2 valid-set prediction, PER SEQUENCE -- the published criterion: correct only if every step is correct.

| model | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| n-gram k=1 | 0.020 | 0.000 | 0.000 | 0.000 |
| n-gram k=3 | 0.029 | 0.000 | 0.002 | 0.000 |
| MapPoPE-1L | 0.661 | 0.108 | 0.552 | 0.042 |
| MapWM-1L | 0.672 | 0.065 | 0.229 | 0.003 |
| MapEM-1L | 0.615 | 0.001 | 0.255 | 0.000 |
| PoPE-1L | 0.000 | 0.000 | 0.000 | 0.000 |
| PoPE-2L | 0.084 | 0.000 | 0.000 | 0.000 |
| RoPE-1L | 0.000 | 0.000 | 0.000 | 0.000 |
| RoPE-2L | 0.097 | 0.000 | 0.000 | 0.000 |
| CoPE-1L | 0.004 | 0.000 | 0.000 | 0.000 |
| CoPE-2L | 0.097 | 0.000 | 0.000 | 0.000 |

## C1 F1 valid continuation -- Goodale et al. 2025, the metric MapFormer reports.

| model | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| n-gram k=1 | 0.807 | 0.800 | 0.871 | 0.857 |
| n-gram k=3 | 0.850 | 0.830 | 0.905 | 0.886 |
| MapPoPE-1L | 0.987 | 0.923 | 0.976 | 0.926 |
| MapWM-1L | 0.985 | 0.900 | 0.943 | 0.868 |
| MapEM-1L | 0.985 | 0.879 | 0.949 | 0.888 |
| PoPE-1L | 0.912 | 0.647 | 0.884 | 0.616 |
| PoPE-2L | 0.948 | 0.505 | 0.679 | 0.472 |
| RoPE-1L | 0.912 | 0.575 | 0.824 | 0.567 |
| RoPE-2L | 0.947 | 0.521 | 0.691 | 0.497 |
| CoPE-1L | 0.903 | 0.703 | 0.855 | 0.704 |
| CoPE-2L | 0.937 | 0.617 | 0.668 | 0.541 |

## C2 strict F1 -- C1 with a min over Val(s) in place of BT's mean.

| model | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| n-gram k=1 | 0.683 | 0.670 | 0.717 | 0.701 |
| n-gram k=3 | 0.773 | 0.745 | 0.783 | 0.772 |
| MapPoPE-1L | 0.974 | 0.887 | 0.945 | 0.871 |
| MapWM-1L | 0.970 | 0.867 | 0.900 | 0.808 |
| MapEM-1L | 0.973 | 0.856 | 0.914 | 0.855 |
| PoPE-1L | 0.869 | 0.491 | 0.798 | 0.414 |
| PoPE-2L | 0.918 | 0.250 | 0.466 | 0.199 |
| RoPE-1L | 0.867 | 0.386 | 0.681 | 0.352 |
| RoPE-2L | 0.917 | 0.262 | 0.474 | 0.206 |
| CoPE-1L | 0.847 | 0.608 | 0.780 | 0.592 |
| CoPE-2L | 0.902 | 0.439 | 0.452 | 0.275 |

## invalid mass -- probability on ungrammatical brackets. Uniform guesser ~0.25.

| model | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| n-gram k=1 | 0.125 | 0.130 | 0.108 | 0.116 |
| n-gram k=3 | 0.084 | 0.094 | 0.081 | 0.083 |
| MapPoPE-1L | 0.002 | 0.028 | 0.003 | 0.018 |
| MapWM-1L | 0.003 | 0.047 | 0.016 | 0.054 |
| MapEM-1L | 0.005 | 0.069 | 0.015 | 0.058 |
| PoPE-1L | 0.078 | 0.217 | 0.078 | 0.195 |
| PoPE-2L | 0.021 | 0.277 | 0.121 | 0.281 |
| RoPE-1L | 0.078 | 0.241 | 0.085 | 0.220 |
| RoPE-2L | 0.020 | 0.232 | 0.113 | 0.229 |
| CoPE-1L | 0.084 | 0.189 | 0.096 | 0.174 |
| CoPE-2L | 0.026 | 0.195 | 0.122 | 0.193 |

## A3 by distance to the bracket that must be closed (L=128, D=12)

| model | d 1-2 | d 3-8 | d 9-32 | d 33+ |
|---|---|---|---|---|
| n-gram k=1 | 0.505 | 0.498 | 0.508 | 0.511 |
| n-gram k=3 | 1.000 | 0.495 | 0.502 | 0.500 |
| MapPoPE-1L | 0.997 | 0.996 | 0.971 | 0.730 |
| MapWM-1L | 0.942 | 0.906 | 0.799 | 0.611 |
| MapEM-1L | 0.951 | 0.933 | 0.851 | 0.576 |
| PoPE-1L | 0.801 | 0.740 | 0.645 | 0.569 |
| PoPE-2L | 0.900 | 0.767 | 0.694 | 0.638 |
| RoPE-1L | 0.693 | 0.641 | 0.570 | 0.539 |
| RoPE-2L | 0.834 | 0.683 | 0.602 | 0.592 |
| CoPE-1L | 0.847 | 0.644 | 0.618 | 0.587 |
| CoPE-2L | 0.800 | 0.666 | 0.616 | 0.626 |

## D cross-entropy against the sampler's own entropy (the achievable floor)

| model | L32 D4 | L128 D4 | L32 D12 | L128 D12 |
|---|---|---|---|---|
| MapPoPE-1L | 0.865 | 1.125 | 1.865 | 1.673 |
| MapWM-1L | 0.865 | 1.221 | 1.892 | 1.791 |
| MapEM-1L | 0.879 | 1.161 | 1.720 | 1.547 |
| PoPE-1L | 0.993 | 2.170 | 1.802 | 2.326 |
| PoPE-2L | 0.848 | 3.646 | 2.740 | 3.852 |
| RoPE-1L | 0.989 | 2.273 | 2.112 | 2.444 |
| RoPE-2L | 0.844 | 3.573 | 2.987 | 3.830 |
| CoPE-1L | 1.007 | 1.371 | 1.637 | 1.492 |
| CoPE-2L | 0.856 | 2.660 | 2.681 | 3.280 |
| *floor (sampler entropy)* | *0.802* | *0.854* | *0.520* | *0.904* |
