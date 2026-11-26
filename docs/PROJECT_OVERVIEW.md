# IoT based edge-gateway-cloud orchestration using reinforcement learning
### Phase II application: adaptive three-tier offloading for real-time visual quality inspection on a smart conveyor

---

## 1. Problem

Cameras + AI are used to catch damaged products on moving belts. The AI can run in three places:

* **on the belt (edge)**: an ESP32, the same board that runs the motor, the sensors and the reject arm, with
  a tiny model. Always there, but the least accurate.
* **next to the belt (gateway)**: a laptop with the webcam and a small model. Fast and fairly accurate, unless
  the laptop is busy.
* **on a cloud server (Render, Singapore)** with a big model: the most accurate, but the answer travels over the
  Internet and can arrive **after the product has already passed the reject point**.

Fixed choices fail: *always-edge* throws away too many good boxes and lets more damaged ones through;
*always-cloud* misses boxes whenever the network degrades; hand-written rules break when something changes
that they were not written for.

## 2. Objectives

1. Build a three-node test-bed: **ESP32 edge node** (belt, two IR sensors, reject arm, tiny model),
   **gateway laptop** (webcam, communication with both sides, Q-learning), **cloud node on Render** (big model).
2. Train three damage-detection models of increasing size and **measure the speed-accuracy trade-off between
   edge, gateway and cloud processing** on the same photos.
3. Formulate the per-box choice among the three places as a reinforcement-learning problem with an
   **expected-cost reward**.
4. Pre-train a tabular Q-learning agent in a simulator, then keep it learning **online** on the gateway.
5. Compare against six baselines (always-edge, always-gateway, always-cloud, threshold, deadline and adaptive
   rules) under steady, changing, mismatched and never-seen conditions.
6. Provide a live dashboard showing every decision, the measured conditions and the agent's current preferences.

## 3. What changed from Phase I

| | Phase I | Phase II (this project) |
|---|---|---|
| Application | IR event duration → LOW/MED/HIGH "task" | real visual inspection of boxes with three CNNs |
| Nodes | one board, simulated cloud delay | ESP32 (edge) + laptop (gateway) + real cloud server on Render (cloud) |
| State | 3 states (task complexity) | 432 states from measurements (spare time, cloud time, ping, gateway speed, queue) |
| Actions | edge / cloud | ESP32 / gateway / cloud |
| Reward | weighted latency/CPU/"wrongness" | **expected cost of the box** (escape / false reject / unchecked) from measured model quality |
| Training | online only, ε = 0.3 | simulator pre-training (900,000 boxes) + safe online learning |
| Baselines | static allocation | 6 baselines, tuned on the same conditions as the agent |
| Evaluation | single run | speed-accuracy measurement on real photos; steady, change-in-the-middle, mismatch, unseen fault (simulated) |

## 4. System architecture

```
 ┌──────────── ESP32 (edge, on the belt) ────────────┐
 │ IR-1: box under the camera ─────────────────────────────► event
 │ IR-2: box at the reject arm ────────────────────────────► event
 │ belt motor (PWM), reject arm (servo)          ◄────────── commands
 │ tiny CNN: 32-48 px grey photo ─► chance damaged ◄──────── photo / ──► answer
 │ safety: gateway silent for 3 s ─► belt stops       │
 └──────────────────────┬─────────────────────────────┘
                        │ USB cable (serial, 460,800 baud; model file sent at every start)
 ┌──────────────────────┴───────── laptop (gateway) ──────────────────────────────────────────────┐
 │ webcam ─► photo ─► STATE (spare time, cloud time, ping, can the gateway finish in time, queue)  │
 │                         │                                                                       │
 │                  Q-learning agent ──► EDGE   : send the shrunken photo to the ESP32             │
 │                         ├──────────► GATEWAY: small CNN here (MobileNetV3-S, 128 px)            │
 │                         └──────────► CLOUD  : JPEG ── Internet ──┐                              │
 │ box at the arm: verdict ready? ─► damaged → tell the ESP32 to push it off                       │
 │                               └─► no answer yet → MISSED         │                              │
 │ reward ─► Q-table update (online) ─► Q-table backup every 60 s ──┤                              │
 │ ping every 0.5 s, test image every 2 s ──────────────────────────┤                              │
 └──────────────────────────────────────────────────────────────────┼──────────────────────────────┘
                              every request carries the API key      ▼
 ┌──────────────────────── Render, Singapore (cloud) ──────────────────────────────────────────────┐
 │ web service (https): /infer (big CNN, ONNX), /ping, dashboard, control buttons, box log (SQLite) │
 │ instant updates (long polling) to the dashboard and the gateway; /api/edge: the ESP32 over Wi-Fi │
 │ (alternative included: AWS EC2 + S3 + CloudWatch + SSM, cloud/aws/cloudformation.yaml)            │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 5. Method (short; details in docs/ALGORITHM.md)

1. **Vision.** Gateway and cloud: "DefectNet", CNN backbone → damage heat-map → box score = max over regions
   (MobileNetV3-Small at 128 px, MobileNetV3-Large at 160 px, ImageNet start). ESP32: three or four 3 × 3
   convolutions on a 32 to 48 px grey photo, about 6,000 to 12,000 numbers, trained from zero. Each model's
   cut-off minimises the same expected cost the agent uses.
2. **RL formulation (second opinion).** The gateway's model checks every box first; its verdict is the
   fallback. State: 7 measured features → 10,800 states, among them how sure the gateway's model is about this
   box and how many cloud calls are left. Actions: keep the gateway's verdict / ask the cloud too / ESP32 (only
   when the gateway is too slow to go first). Reward: −expected cost of the box given who answered and how
   sure the gateway was − energy. Q-learning with delayed updates, γ = 0.9.
3. **Simulator.** Event-based model of capture, USB link and ESP32, gateway queue and load, JPEG, network
   (RTT, jitter, loss with TCP retransmission timeouts, bandwidth), cloud queue, belt deadline.
4. **Pre-training.** 2.4 million simulated boxes under randomised and changing conditions. Every simulated
   box is a real photo with the three models' real verdicts (`tools/photo_scores.py`): validation photos for
   training, test photos for judging.
5. **Online learning.** α = 0.1, safe ε-greedy (explore only when options are close), fading of untried values.
   **Spot checks:** when the true state of a box is known afterwards, the agent compares the gateway's and the
   cloud's verdict on it and keeps a running average of what the cloud call saved, per confidence level. That
   replaces the number from the training photos, so the agent follows the line when the photos change.
6. **Evaluation.** Speed against accuracy on the same photos at the three places (`tools/tradeoff.py`);
   what a second opinion is worth on real photos (`results/CASCADE.md`);
   E1 steady (10 situations), E2 change-in-the-middle (6 changes), E3 pre-training value, E4 learned policy
   map, E5 simulator/hardware mismatch, E6 other damage rates and costs, E7 cloud-call budget, E8 the photos
   get darker / blurred in the middle of a run, E9 every gap with its range over 200 re-drawn test sets; a
   second public benchmark.

## 6. Results: see results/TRADEOFF.md and results/SIM_RESULTS.md

<!-- RESULTS_SUMMARY -->
**Speed against accuracy** (the same 594 photos at all three places, measured 2026-11-26 01:14)

| Place | Correct verdicts | Time per photo (median) | How it was measured |
|---|---|---|---|
| ESP32 (edge) | 65.3% | 77.4 ms | EMULATED (no board connected): cable time from the baud rate, compute time from 40 ns per multiply |
| Gateway (laptop) | 89.9% | 1.5 ms | this laptop, CPU |
| Cloud server (on this PC for now) | 95.1% | 5.7 ms | server on THIS PC: no Internet in the path, so the real cloud will be slower |

- Most accurate first: Cloud server (on this PC for now) 95.1%, Gateway (laptop) 89.9%, ESP32 (edge) 65.3%.
- Fastest first: Gateway (laptop) 1 ms, Cloud server (on this PC for now) 6 ms, ESP32 (edge) 77 ms.
- The ESP32 is both slower (77 ms against 1 ms) and less accurate than the gateway, because the photo has to be sent to it over the cable first. It is not there to win this table. It is there for when the gateway is too busy and the Internet is too slow: then it is the only place that still answers before the box reaches the arm.
- Going from the gateway to the cloud buys 5.2 points of accuracy for 4 ms more per box.
- The ESP32 time is an estimate until this is run with the board plugged in. Its accuracy is exact: the same arithmetic as the firmware.
- The cloud server ran on this same PC, so its time has no Internet in it. Run this again with --server http://<AWS address>:8000 for the real number.

**Second opinion on real photos** (594 test photos, public; measured 2026-11-26 00:53)

- The gateway's and the cloud's model give the same verdict on 89.7% of photos. The cloud corrects the gateway on 7.9% and makes it worse on 2.4%.
- Sending only the least-sure 25% of boxes to the cloud: 94.3% correct (all boxes: 95.5%, none: 89.9%; the same 25% picked at random: 91.3%).

| Gateway's model is | Share of boxes | Gateway right | Cloud right | Cost saved per cloud call |
|---|---|---|---|---|
| very unsure | 12.2% | 49.0% | 80.4% | +4.54 |
| unsure | 16.5% | 79.5% | 90.4% | +3.04 |
| fairly sure | 26.8% | 93.3% | 97.0% | +0.33 |
| sure | 44.6% | 97.0% | 98.2% | +0.26 |

**The three vision models** (held-out test photos: real photos from a public set, not this project's belt)

| Model | Backbone | Accuracy | Damaged caught | Good wrongly rejected | Threshold |
|---|---|---|---|---|---|
| ESP32 (edge) | TinyCNN: 4 x 3x3 conv (8, 16, 24, 32), average, 1 output @ 48 px | 65.3% | 98.0% | 81.2% | 0.05 |
| Gateway laptop | mobilenet_v3_small @ 128 px | 89.9% | 96.0% | 18.8% | 0.06 |
| Cloud server | mobilenet_v3_large @ 160 px | 95.5% | 97.7% | 7.8% | 0.16 |

**E1 — 10 network/device situations, 20 runs × 400 boxes each (simulation; every box is a real test photo, reward = minus the real cost of what happened to it, cloud calls limited)**

| Policy | Avg reward / box (higher = better) | Boxes missed | Boxes sent to the cloud | Damaged got through | Good boxes wrongly rejected |
|---|---|---|---|---|---|
| Q-learning (frozen) | -2.60 | 0.0% | 18.0% | 3.3% | 21.8% |
| Q-learning (online) | -2.64 | 0.0% | 19.6% | 3.4% | 21.8% |
| Confidence rule | -2.83 | 0.0% | 8.1% | 3.7% | 23.3% |
| Confidence rule + reserve | -2.91 | 0.0% | 9.8% | 4.1% | 23.2% |
| Ask the cloud always | -2.95 | 0.0% | 23.5% | 4.0% | 23.5% |
| Adaptive rule | -3.25 | 0.8% | 10.3% | 5.0% | 23.7% |
| Always-Gateway | -5.46 | 7.3% | 0.0% | 11.6% | 17.3% |
| Always-Edge | -6.32 | 0.0% | 0.0% | 1.7% | 81.6% |
| Threshold rule | -6.74 | 11.6% | 9.3% | 15.5% | 16.0% |
| Deadline rule | -6.79 | 11.8% | 10.3% | 15.7% | 15.9% |
| Always-Cloud | -8.04 | 16.3% | 26.1% | 19.9% | 14.1% |

**E9 — which gaps are real** (the comparison repeated on 200 re-drawn test sets; a gap is real only when its 95% range does not cross zero)

| Cost per box, first minus second | Gap | 95% range | Verdict |
|---|---|---|---|
| Q-learning (online) vs Adaptive rule | -0.58 | -1.00 to -0.26 | real gap |
| Q-learning (online) vs Ask the cloud always | -0.28 | -0.57 to -0.02 | real gap |
| Q-learning (online) vs Confidence rule | -0.17 | -0.42 to +0.08 | level, within chance |
| Q-learning (frozen) vs Confidence rule | -0.20 | -0.48 to +0.05 | level, within chance |
| Confidence rule vs Adaptive rule | -0.41 | -0.67 to -0.16 | real gap |

**E7 — cost per box with fewer cloud calls allowed (good network, simulation, lower = better)**

| Policy | no limit | 30/min | 15/min | 8/min | 4/min |
|---
|---
|---
|---
|---
|---|
| Adaptive rule | 1.23 | 1.78 | 2.14 | 2.24 | 2.33 |
| Ask the cloud always | 1.29 | 1.88 | 2.04 | 2.23 | 2.30 |
| Confidence rule | 1.45 | 1.45 | 1.57 | 1.91 | 2.08 |
| Confidence rule + reserve | 1.40 | 1.45 | 1.70 | 1.87 | 2.14 |
| Q-learning (frozen) | 1.40 | 1.44 | 1.55 | 1.87 | 2.08 |
| Q-learning (online) | 1.42 | 1.48 | 1.62 | 1.94 | 2.13 |
| Best possible (knows the answers) | 0.52 | 0.52 | 0.52 | 0.58 | 1.14 |

**E8 — the photos get worse in the middle of a run (simulation; cost per box 2001-3500 boxes after the change, lower = better)**

The same real test photos, made darker / blurred by software and scored again by the same models. Nothing is re-tuned. Only the "spot checks" rows learn the true state of boxes afterwards.

| Policy | blurred, 15/min | blurred, no limit | darker, 15/min | darker, no limit | darker + blurred, 15/min | darker + blurred, no limit |
|---
|---
|---
|---
|---
|---
|---|
| Adaptive rule | 4.09 | 3.95 | 2.51 | 2.14 | 6.26 | 10.58 |
| Ask the cloud always | 4.18 | 4.02 | 2.53 | 2.20 | 6.29 | 10.65 |
| Confidence rule | 3.25 | 3.00 | 2.10 | 1.97 | 5.16 | 5.19 |
| Confidence rule + reserve | 3.79 | 3.66 | 2.10 | 2.13 | 5.52 | 6.93 |
| Q-learning (frozen) | 3.26 | 3.66 | 2.03 | 2.13 | 5.11 | 6.93 |
| Q-learning (online) | 3.28 | 3.65 | 2.06 | 2.13 | 5.21 | 6.97 |
| Q-learning (online + spot checks) | 3.34 | 3.11 | 2.11 | 2.09 | 5.05 | 5.10 |
| Q-learning (online + 1 box in 5 checked) | 3.39 | 3.19 | 2.10 | 2.06 | 5.22 | 5.32 |
| Best possible (knows the answers) | 1.58 | 1.58 | 0.79 | 0.79 | 2.22 | 1.47 |

**E2 — problem appears in the middle (boxes 301–600), nothing re-tuned**

| Change | Policy | Missed during the problem | Boxes to recover |
|---|---|---|---|
| Cloud server overloaded (invisible to the gateway) | Threshold rule | 13.0% | >300 |
| Cloud server overloaded (invisible to the gateway) | Deadline rule | 13.0% | >300 |
| Cloud server overloaded (invisible to the gateway) | Adaptive rule | 0.0% | 25 |
| Cloud server overloaded (invisible to the gateway) | Confidence rule + reserve | 0.0% | 25 |
| Cloud server overloaded (invisible to the gateway) | Q-learning (online) | 0.0% | 25 |
| Gateway overloaded while the network is bad | Threshold rule | 72.7% | >300 |
| Gateway overloaded while the network is bad | Deadline rule | 72.7% | >300 |
| Gateway overloaded while the network is bad | Adaptive rule | 0.3% | 25 |
| Gateway overloaded while the network is bad | Confidence rule + reserve | 0.3% | 25 |
| Gateway overloaded while the network is bad | Q-learning (online) | 0.2% | 25 |
| Wi-Fi bandwidth capped (invisible in ping) | Threshold rule | 25.0% | >300 |
| Wi-Fi bandwidth capped (invisible in ping) | Deadline rule | 25.0% | >300 |
| Wi-Fi bandwidth capped (invisible in ping) | Adaptive rule | 0.1% | 25 |
| Wi-Fi bandwidth capped (invisible in ping) | Confidence rule + reserve | 0.0% | 25 |
| Wi-Fi bandwidth capped (invisible in ping) | Q-learning (online) | 0.0% | 25 |
| Wi-Fi stalls appear (NEVER seen in training) | Threshold rule | 2.5% | 25 |
| Wi-Fi stalls appear (NEVER seen in training) | Deadline rule | 4.9% | 299 |
| Wi-Fi stalls appear (NEVER seen in training) | Adaptive rule | 4.5% | 295 |
| Wi-Fi stalls appear (NEVER seen in training) | Confidence rule + reserve | 0.0% | 25 |
| Wi-Fi stalls appear (NEVER seen in training) | Q-learning (online) | 0.0% | 25 |

**E5 — 'real' hardware 2–3× slower than the simulator assumed:** Threshold rule -5.67, Deadline rule -6.20, Adaptive rule -3.23, Q-learning (frozen) -2.79, Q-learning (online) -2.76 (reward/box). Measured features (ping, test-image time, time left) absorb such differences, so no method needs re-tuning; the fixed rules are hurt most.

Average over all 10 situations: Q-learning (online) -2.64 against the best hand-written rule (Confidence rule) -2.83: **Q-learning ahead by 0.19 on these photos; over 200 re-drawn test sets the gap is 0.17 (95% range -0.08 to +0.42): level, within chance.** Against the best rule that picks ONE place per box (Adaptive rule, -3.25) Q-learning is ahead by 0.61. Figures: `results/figures/E1_steady_comparison.png`, `E2_change_in_middle.png`, `E3_pretraining_value.png`, `E4_policy_map.png`, `E5_sim_to_real_mismatch.png`.
<!-- /RESULTS_SUMMARY -->

## 7. Honest limitations

* The conveyor is a **test-bed**: a real factory would wire a strong computer at the line. The method targets
  wireless / mobile / many-cheap-device settings.
* **The ESP32 is not a good inspector.** A 32 to 48 px grey photo does not show enough: its model is right
  about two times out of three. And because the webcam is on the gateway, a photo must be sent to the ESP32
  first, so in normal conditions it is slower than the gateway as well. It earns its place only when the
  gateway is overloaded and the Internet is slow.
* Rows of the speed-accuracy table marked "emulated" or "on this PC" are not final until the measurement is
  repeated with the board plugged in. The cloud server is now on Render and its time over the Internet was
  measured from the gateway laptop: 83 ms per box photo (95th percentile 97 ms), 5.2 ms of it the model.
* The models are trained on real photographs from a **public** set, not on photos from this project's belt.
* A hand-tuned **confidence rule** that uses the same confidence signal is **level** with the agent: over 200
  re-drawn test sets the gap between them is within chance (E9). The proven gain is the second-opinion design
  itself, against every rule that picks one place per box. What the agent adds is that nobody tunes it and that
  it corrects itself.
* **The agent's table is tied to the photos it was trained on.** On darker or blurred photos the cloud's model
  loses more than the gateway's, and the trained agent then asks the cloud too often (E8). Spot checks fix that,
  but they need the true state of some boxes. The darker / blurred photos in E8 are made by software from the
  public photos; they are not taken on a belt.
* With a tight cloud-call limit, and whenever only the limit changes, the agent and the confidence rule do the
  same thing. With no limit and clean photos, simply asking the cloud for every box is as good as anything.
* The decision experiments are **simulated**. **The belt has not run on real hardware yet:** the belt is not built
  (wiring guide and circuit diagram are ready and checked against the firmware, `docs/hardware/`), and firmware 4
  (with Wi-Fi) still has to be flashed and tried on the board. The cloud server is deployed (Render).
* Render's free plan sleeps after 15 minutes without requests (about a minute to wake up) and keeps the box log
  only until the next restart.

## 8. References

1. Phase-I report: "IoT Based Edge–Cloud Orchestration using Reinforcement Learning", Dept. of E&TE, SIT Tumakuru, 2025–26.
2. A. Kouloumpris, G. L. Stavrinides, M. K. Michael, T. Theocharides, "An optimization framework for task allocation
   in the edge/hub/cloud paradigm", *Future Generation Computer Systems*, vol. 155, pp. 354–366, 2024.
3. Datasets of synthetic task flow graphs …, Zenodo, 2024, doi:10.5281/zenodo.10654551.
4. R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018 (Q-learning, ε-greedy).
5. A. Howard et al., "Searching for MobileNetV3", ICCV 2019. K. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
6. Linux `tc-netem(8)` manual — network emulation (delay, loss, rate, Gilbert–Elliott burst loss).
