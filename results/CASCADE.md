# Second opinion: which boxes are worth a cloud call

Generated 2026-11-26 00:53. 594 test photos (public), never used for training or for choosing the confidence levels (those come from the 588 validation photos). Costs assume 30% damaged boxes, 100 for a shipped damaged box and 10 for a wasted good one.

- The gateway's and the cloud's model give the same verdict on 89.7% of photos.
- The cloud corrects the gateway on 7.9% and makes it worse on 2.4%. Both are wrong on 2.2%.

## How sure the gateway is, and what a cloud call is worth

| Gateway is | Share of boxes | Gateway right | Cloud right | Cost if kept | Cost if cloud asked | Gain per cloud call |
|---|---|---|---|---|---|---|
| very unsure | 12% | 49.0% | 80.4% | 8.13 | 3.59 | +4.54 |
| unsure | 16% | 79.5% | 90.4% | 4.91 | 1.87 | +3.04 |
| fairly sure | 27% | 93.3% | 97.0% | 1.40 | 1.07 | +0.33 |
| sure | 45% | 97.0% | 98.2% | 0.65 | 0.39 | +0.26 |

## Send only the least-sure boxes to the cloud

| Sent to the cloud | Correct | Cost per box | Same share picked at random: correct | cost |
|---|---|---|---|---|
| 0% | 89.9% | 2.52 | 89.9% | 2.52 |
| 5% | 91.9% | 2.00 | 90.2% | 2.45 |
| 10% | 92.8% | 1.80 | 90.5% | 2.39 |
| 15% | 93.3% | 1.66 | 90.7% | 2.32 |
| 20% | 93.9% | 1.43 | 91.1% | 2.24 |
| 25% | 94.3% | 1.37 | 91.3% | 2.18 |
| 30% | 94.4% | 1.40 | 91.5% | 2.16 |
| 40% | 94.6% | 1.37 | 92.2% | 1.98 |
| 50% | 94.8% | 1.34 | 92.7% | 1.88 |
| 75% | 95.3% | 1.26 | 94.1% | 1.54 |
| 100% | 95.5% | 1.23 | 95.5% | 1.23 |

With 594 photos a difference of about 1.5 points in 'correct' is within chance.
