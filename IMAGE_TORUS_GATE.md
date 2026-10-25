# Image torus: gate, floors, perception ceiling (gate_image_torus.py; no trained model)

Bank md5 `72cdafaf92a9a42b1b6266ff72423e09`. Stream check: train_image's environment equals GridWorld(64, 16, 0.5, seed) and emits the same batch under the same np seed for every seed below; max token < 21.

| seed | blank | retrace | n-gram (token, best of 0-5) | n-gram (actions only, best of 1-5) | best trivial | retrace short / long / wrap | ceil_last | ceil_vote |
|---|---|---|---|---|---|---|---|---|---|
| 3100 | 0.5147 | 0.8419 | 0.5245 | 0.5147 | 0.8419 | 0.908 / 0.516 / 0.522 | 0.9903 | 0.9928 |
| 3101 | 0.5052 | 0.8377 | 0.5084 | 0.5052 | 0.8377 | 0.908 / 0.505 / 0.505 | 0.9894 | 0.9928 |
| 3102 | 0.5076 | 0.8440 | 0.5164 | 0.5076 | 0.8440 | 0.913 / 0.506 / 0.509 | 0.9894 | 0.9921 |
| 3103 | 0.5037 | 0.8406 | 0.5148 | 0.5037 | 0.8406 | 0.907 / 0.495 / 0.504 | 0.9900 | 0.9927 |
| 3104 | 0.5141 | 0.8461 | 0.5248 | 0.5141 | 0.8461 | 0.913 / 0.505 / 0.510 | 0.9904 | 0.9931 |
| 3105 | 0.5040 | 0.8434 | 0.5148 | 0.5040 | 0.8434 | 0.911 / 0.522 / 0.505 | 0.9897 | 0.9923 |
| 3106 | 0.5046 | 0.8476 | 0.5102 | 0.5046 | 0.8476 | 0.915 / 0.513 / 0.517 | 0.9905 | 0.9931 |
| 3107 | 0.5094 | 0.8440 | 0.5190 | 0.5094 | 0.8440 | 0.912 / 0.514 / 0.495 | 0.9899 | 0.9930 |

Standalone classifier (encoder 79,968 params + linear head, 10 epochs): test 0.9801, train 0.9861. Per class (test): 0 0.994, 1 0.993, 2 0.986, 3 0.988, 4 0.994, 5 0.970, 6 0.985, 7 0.985, 8 0.987, 9 0.970, trouser 0.988, dress 0.967, sandal 0.966, sneaker 0.963, bag 0.985, ankle_boot 0.958

## Init scale (Amendment 1, M2): image MapWM (fixed encoder gain 30.3729) vs token twin, at initialisation, 10 evaluation walks

| seed | obs | emb norm action / object / blank | rms step object / action | within-class (instance) / action |
|---|---|---|---|---|
| 3100 | token | 11.29 / 11.23 / 11.13 | 1.314 | 0.000 |
| 3100 | image | 11.29 / 13.01 / 11.74 | 1.709 | 0.261 |
| 3101 | token | 11.50 / 11.29 / 10.52 | 1.281 | 0.000 |
| 3101 | image | 11.50 / 11.40 / 10.45 | 0.438 | 0.292 |
| 3102 | token | 11.53 / 11.15 / 11.04 | 1.080 | 0.000 |
| 3102 | image | 11.53 / 11.63 / 11.23 | 1.078 | 0.161 |
| 3103 | token | 11.11 / 11.22 / 10.62 | 1.223 | 0.000 |
| 3103 | image | 11.11 / 11.51 / 9.84 | 0.706 | 0.247 |
| 3104 | token | 11.54 / 11.39 / 11.09 | 0.909 | 0.000 |
| 3104 | image | 11.54 / 10.79 / 10.17 | 1.390 | 0.174 |
| 3105 | token | 11.77 / 10.96 / 9.75 | 0.594 | 0.000 |
| 3105 | image | 11.77 / 10.45 / 9.68 | 0.497 | 0.115 |
| 3106 | token | 11.44 / 11.20 / 11.72 | 0.989 | 0.000 |
| 3106 | image | 11.44 / 10.74 / 9.53 | 0.422 | 0.146 |
| 3107 | token | 10.79 / 11.25 / 12.60 | 1.047 | 0.000 |
| 3107 | image | 10.79 / 11.71 / 10.79 | 1.016 | 0.195 |

token: object/action step ratio mean 1.055 (range 0.594-1.314); instance/action mean 0.000

image: object/action step ratio mean 0.907 (range 0.422-1.709); instance/action mean 0.199
