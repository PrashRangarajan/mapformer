# Compositional-motif results

Cross-instance = compositional target (motif seen elsewhere). cross_nb = non-blank subset. Fresh env, OOD length.


## T=256

| variant | exact_acc | cross_acc | cross_nb_acc | cross_nll |
|---|---|---|---|---|
| PlainHourglass | 0.925 | 0.635 | 0.357 | 1.342 |
| PlainFlat | 0.905 | 0.568 | 0.212 | 1.603 |

## T=512

| variant | exact_acc | cross_acc | cross_nb_acc | cross_nll |
|---|---|---|---|---|
| PlainHourglass | 0.866 | 0.586 | 0.250 | 1.559 |
| PlainFlat | 0.793 | 0.529 | 0.103 | 1.820 |

## T=1024

| variant | exact_acc | cross_acc | cross_nb_acc | cross_nll |
|---|---|---|---|---|
| PlainHourglass | 0.728 | 0.536 | 0.119 | 1.790 |
| PlainFlat | 0.634 | 0.509 | 0.039 | 1.983 |

## T=2048

| variant | exact_acc | cross_acc | cross_nb_acc | cross_nll |
|---|---|---|---|---|
| PlainHourglass | 0.612 | 0.520 | 0.055 | 1.952 |
| PlainFlat | 0.524 | 0.487 | 0.020 | 2.141 |
