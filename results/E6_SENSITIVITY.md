# E6 - Does the conclusion survive other assumptions? (simulated)

Generated: 2026-11-25 23:53. For every setting the model thresholds were re-picked, the agent re-trained and the rules re-tuned. Scores are average reward per box over the situations of E1 (higher is better; compare within a row only, the cost scale changes between rows).

Image sets used for the thresholds: 588 validation, 594 test images. AUC (how well each model separates good from damaged, 1.0 = perfectly, 0.5 = guessing): esp32 0.723, gateway 0.967, cloud 0.984.

| setting | Always-Edge | Always-Gateway | Always-Cloud | Threshold rule | Deadline rule | Adaptive rule | Ask the cloud always | Confidence rule | Confidence rule + reserve | Q-learning (frozen) | Q-learning (online) | best | Q online minus best rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1% damaged | -1.22 | -1.40 | -1.81 | -1.64 | -1.69 | -0.69 | -0.68 | -0.61 | -0.61 | **-0.56** | -0.57 | Q-learning (frozen) | +0.04 |
| 5% damaged | -4.30 | -2.09 | -2.85 | -2.52 | -2.55 | -1.40 | -1.37 | **-1.31** | -1.34 | -1.34 | -1.33 | Confidence rule | -0.02 |
| 30% damaged (base) | -6.39 | -5.18 | -7.65 | -5.18 | -6.42 | -3.05 | -2.79 | -2.74 | -2.72 | **-2.53** | -2.58 | Q-learning (frozen) | +0.14 |
| base, wasted good box costs 3x | -17.19 | -6.54 | -8.77 | -7.66 | -7.66 | -5.42 | -5.13 | -4.92 | -4.89 | -4.87 | **-4.85** | Q-learning (online) | +0.03 |
| base, escaped defect costs 3x | -7.49 | -11.09 | -18.25 | -11.09 | -14.58 | -5.20 | -4.63 | -4.52 | -4.41 | **-4.11** | -4.13 | Q-learning (frozen) | +0.28 |

**Model operating points picked for each setting** (threshold / damaged caught / good wrongly rejected)

| setting | ESP32 model | gateway model | cloud model | reward: ESP32 in time | gateway in time | cloud in time | missed |
|---|---|---|---|---|---|---|---|
| 1% damaged | 0.95 / 11.5% / 4.9% | 0.91 / 86.0% / 5.3% | 0.94 / 89.7% / 2.9% | -1.37 | -0.67 | -0.39 | -6.00 |
| 5% damaged | 0.84 / 35.8% / 13.5% | 0.75 / 89.4% / 6.5% | 0.82 / 94.0% / 4.1% | -4.49 | -1.15 | -0.69 | -10.00 |
| 30% damaged (base) | 0.05 / 98.0% / 81.2% | 0.06 / 96.0% / 18.8% | 0.16 / 97.7% / 7.8% | -6.29 | -2.52 | -1.23 | -35.00 |
| base, wasted good box costs 3x | 0.07 / 97.1% / 75.5% | 0.51 / 91.7% / 7.8% | 0.35 / 97.4% / 5.7% | -16.72 | -4.12 | -1.97 | -35.00 |
| base, escaped defect costs 3x | 0.05 / 98.0% / 81.2% | 0.05 / 96.6% / 20.0% | 0.16 / 97.7% / 7.8% | -7.49 | -4.49 | -2.61 | -95.00 |

**Missed boxes % (average over the situations)**

| setting | Always-Edge | Always-Gateway | Always-Cloud | Threshold rule | Deadline rule | Adaptive rule | Ask the cloud always | Confidence rule | Confidence rule + reserve | Q-learning (frozen) | Q-learning (online) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1% damaged | 0.0 | 7.3 | 16.2 | 11.5 | 12.7 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5% damaged | 0.0 | 7.3 | 16.2 | 12.8 | 12.4 | 0.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |
| 30% damaged (base) | 0.0 | 7.3 | 16.2 | 7.3 | 11.7 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| base, wasted good box costs 3x | 0.0 | 7.3 | 16.2 | 11.5 | 11.7 | 0.8 | 0.0 | 0.3 | 0.3 | 0.3 | 0.3 |
| base, escaped defect costs 3x | 0.0 | 7.3 | 16.2 | 7.3 | 11.7 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
