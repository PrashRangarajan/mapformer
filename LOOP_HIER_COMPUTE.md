# Loop x hierarchy: what each saves, measured

Forward+backward, batch 64, timed alone on an idle device after warmup with explicit synchronisation.

The loop saves PARAMETERS and costs COMPUTE; hierarchy saves COMPUTE and
costs PARAMETERS. The question is whether combining them gives both.

| length | arm | params | ms/step | peak MiB | vs flat-unshared |
|---|---|---|---|---|---|
| 16 | HourglassFlat3 | 596,034 | 9.83 | 58.6 | +0.0% time, +0.0% mem |
| 16 | Hourglass_k2 | 596,034 | 11.03 | 53.5 | +12.2% time, -8.7% mem |
| 16 | LoopedHourglassFlat | 199,490 | 10.33 | 54.1 | +5.1% time, -7.7% mem |
| 16 | LoopedHourglass | 199,490 | 10.71 | 49.0 | +9.0% time, -16.4% mem |
| 128 | HourglassFlat3 | 596,034 | 11.01 | 345.9 | +0.0% time, +0.0% mem |
| 128 | Hourglass_k2 | 596,034 | 11.23 | 298.0 | +2.0% time, -13.8% mem |
| 128 | LoopedHourglassFlat | 199,490 | 10.54 | 341.3 | -4.3% time, -1.3% mem |
| 128 | LoopedHourglass | 199,490 | 11.05 | 293.5 | +0.4% time, -15.1% mem |
| 512 | HourglassFlat3 | 596,034 | 26.28 | 2050.6 | +0.0% time, +0.0% mem |
| 512 | Hourglass_k2 | 596,034 | 20.56 | 1702.1 | -21.8% time, -17.0% mem |
| 512 | LoopedHourglassFlat | 199,490 | 26.30 | 2046.0 | +0.1% time, -0.2% mem |
| 512 | LoopedHourglass | 199,490 | 20.58 | 1697.6 | -21.7% time, -17.2% mem |
| 2048 | HourglassFlat3 | 596,034 | 280.97 | 20551.6 | +0.0% time, +0.0% mem |
| 2048 | Hourglass_k2 | 596,034 | 216.62 | 16467.7 | -22.9% time, -19.9% mem |
| 2048 | LoopedHourglassFlat | 199,490 | 281.20 | 20547.1 | +0.1% time, -0.0% mem |
| 2048 | LoopedHourglass | 199,490 | 216.77 | 16463.1 | -22.8% time, -19.9% mem |

## The pairing

- **L=16**: hierarchy alone +12.2% time; sharing alone +5.1%; **both +9.0% time and -66.5% parameters**.
- **L=128**: hierarchy alone +2.0% time; sharing alone -4.3%; **both +0.4% time and -66.5% parameters**.
- **L=512**: hierarchy alone -21.8% time; sharing alone +0.1%; **both -21.7% time and -66.5% parameters**.
- **L=2048**: hierarchy alone -22.9% time; sharing alone +0.1%; **both -22.8% time and -66.5% parameters**.

Sharing is expected to cost roughly nothing in time -- it changes which weights are read, not how many block applications run -- so any time saving in the combined arm comes from the hierarchy and any parameter saving comes from the sharing. That is the point: the two resources are independent, so the savings should compose exactly.
