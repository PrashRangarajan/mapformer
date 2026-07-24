# Compositional-motif results

Cross-instance = compositional target (motif seen elsewhere). cross_nb = non-blank subset. Fresh env, OOD length.


## T=256

| variant | exact_acc | cross_acc | cross_nb_acc | cross_nll |
|---|---|---|---|---|
| Vanilla | 0.929 | 0.571 | 0.297 | 1.343 |
| VanillaEM | 0.947 | 0.516 | 0.089 | 1.494 |
| Hourglass_k2 | 0.939 | 0.611 | 0.344 | 1.409 |
| HourglassFlat3 | 0.920 | 0.578 | 0.251 | 1.600 |

## T=512

| variant | exact_acc | cross_acc | cross_nb_acc | cross_nll |
|---|---|---|---|---|
| Vanilla | 0.893 | 0.532 | 0.188 | 1.587 |
| VanillaEM | 0.909 | 0.500 | 0.046 | 1.696 |
| Hourglass_k2 | 0.911 | 0.559 | 0.221 | 1.581 |
| HourglassFlat3 | 0.866 | 0.526 | 0.144 | 2.117 |

## T=1024

| variant | exact_acc | cross_acc | cross_nb_acc | cross_nll |
|---|---|---|---|---|
| Vanilla | 0.721 | 0.503 | 0.075 | 2.493 |
| VanillaEM | 0.776 | 0.483 | 0.024 | 1.896 |
| Hourglass_k2 | 0.826 | 0.518 | 0.095 | 2.046 |
| HourglassFlat3 | 0.674 | 0.498 | 0.052 | 3.993 |

## T=2048

| variant | exact_acc | cross_acc | cross_nb_acc | cross_nll |
|---|---|---|---|---|
| Vanilla | 0.597 | 0.505 | 0.034 | 2.998 |
| VanillaEM | 0.618 | 0.488 | 0.010 | 1.971 |
| Hourglass_k2 | 0.658 | 0.511 | 0.045 | 2.622 |
| HourglassFlat3 | 0.568 | 0.496 | 0.025 | 4.594 |
