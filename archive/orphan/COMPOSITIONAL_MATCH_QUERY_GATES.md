# Compositional Match-Query -- pre-flight gates (CPU, no training)

Blind continuation in a repeated-motif world. `exact` = exact cell seen in explore (path-integration matching). `cross` = exact cell NOT seen, room explored + motif seen in another copy (path-integration AND motif abstraction). Non-blank only; chance = 1/16 = 0.0625. Oracle = 1.000 for both by construction.

| TE | TQ | cat | chance | marginal | ng1 | ng3 | ng5 | never-moved | n |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 64 | exact | 0.0625 | 0.0823 | 0.0445 | 0.0716 | 0.0697 | 0.1777 | 899 |
| 128 | 64 | cross | 0.0625 | 0.0768 | 0.0800 | 0.0846 | 0.0786 | 0.0893 | 951 |
| 128 | 128 | exact | 0.0625 | 0.0751 | 0.0606 | 0.0781 | 0.0836 | 0.1272 | 1158 |
| 128 | 128 | cross | 0.0625 | 0.0663 | 0.0580 | 0.0651 | 0.0681 | 0.0952 | 1449 |
| 256 | 64 | exact | 0.0625 | 0.0877 | 0.0411 | 0.0844 | 0.0960 | 0.1559 | 1072 |
| 256 | 64 | cross | 0.0625 | 0.0717 | 0.0568 | 0.0745 | 0.0654 | 0.0837 | 1938 |
| 256 | 128 | exact | 0.0625 | 0.0774 | 0.0591 | 0.0776 | 0.0805 | 0.1142 | 1525 |
| 256 | 128 | cross | 0.0625 | 0.0712 | 0.0628 | 0.0641 | 0.0743 | 0.0709 | 3187 |
| 512 | 64 | exact | 0.0625 | 0.0752 | 0.0476 | 0.0689 | 0.0784 | 0.1237 | 1516 |
| 512 | 64 | cross | 0.0625 | 0.0692 | 0.0543 | 0.0561 | 0.0619 | 0.0829 | 3466 |
| 512 | 128 | exact | 0.0625 | 0.0701 | 0.0631 | 0.0687 | 0.0704 | 0.0924 | 2538 |
| 512 | 128 | cross | 0.0625 | 0.0712 | 0.0674 | 0.0638 | 0.0757 | 0.0766 | 6589 |

| TE | TQ | answerable rate | consistency fails |
|---|---|---|---|
| 128 | 64 | 0.072 | 0 |
| 128 | 128 | 0.051 | 0 |
| 256 | 64 | 0.118 | 0 |
| 256 | 128 | 0.092 | 0 |
| 512 | 64 | 0.195 | 0 |
| 512 | 128 | 0.178 | 0 |

**Reading it.** Every baseline column should sit at chance (0.0625) for BOTH categories. `cross` marginal/n-gram above chance would mean the compositional answer is guessable without the map. `consistency fails` must be 0 (else the cross target is ill-defined). `answerable rate` sets the effective sample size.
