# Smart Conveyor QC: edge, gateway and cloud orchestration with reinforcement learning

Boxes travel on a conveyor. A webcam photographs each one. **Three places** can judge the photo:

| Node | Hardware | Model | Strength | Weakness |
|---|---|---|---|---|
| **Edge** | ESP32 on the belt | tiny CNN, about 6,000 numbers, grey 32 or 48 px photo | needs no Internet and no laptop CPU; also runs the belt | least accurate; the photo must be sent to it over USB |
| **Gateway** | laptop next to the belt | MobileNetV3-Small, 128 px | fast, good accuracy | slow when the laptop is overloaded |
| **Cloud** | AWS EC2 (Mumbai) | MobileNetV3-Large, 160 px | most accurate | the photo crosses the Internet; the answer can come too late |

They work as a **second opinion** system. The gateway's model checks **every box first** (about 2 ms). A
**Q-learning agent on the gateway** then decides whether the cloud should check it too, using how sure the
gateway's model is, the time left, the network, and how many **cloud calls** are left (they are limited). A
cloud answer that comes too late changes nothing: the gateway's verdict is used, so the box is still checked.
When the laptop is too busy to check boxes in time, the agent hands them to the ESP32.

```
 ESP32 (edge)  <---- USB cable ---->  laptop (gateway)  <---- Internet ---->  AWS (cloud)
 belt motor, 2 IR sensors,            webcam, Q-learning agent,               big model,
 reject arm, LEDs, tiny model         small model, link to both sides         dashboard, box log
```

The project's experiment is the **speed-accuracy trade-off between the three places**
(`MEASURE_SPEED_VS_ACCURACY.bat`, table below), what a cloud second opinion is worth on real photos
(`results/CASCADE.md`), and how well the agent uses both when the network, the gateway or the cloud-call
budget gets tight.

## Fastest start (Windows, nothing to type)

Double-click **`START_DEMO.bat`**: no hardware needed. All double-click files and the hardware order are in
**[START_HERE.md](START_HERE.md)**. Wiring and flashing the ESP32: **[docs/ESP32_WIRING.md](docs/ESP32_WIRING.md)**.

## The same by hand

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements-laptop.txt

python -m server.app                       # terminal 1: the cloud node, here on this PC -> http://localhost:8000
python -m gateway_node.main --mock         # terminal 2: gateway with a pretend belt and a pretend ESP32
python -m gateway_node.main --mock --esp32 auto    # ... pretend belt, tiny model on a real ESP32 (USB)
python -m gateway_node.main --server http://<ec2-ip>:8000   # the real line: ESP32 + webcam + AWS
python -m tools.tradeoff --esp32 auto --server http://<ec2-ip>:8000   # speed against accuracy, all real
```

## Put the cloud node on AWS

Easiest: AWS CloudShell, 3 steps, in [START_HERE.md](START_HERE.md). From a PC with the AWS CLI:

```powershell
.\cloud\deploy_aws.ps1               # EC2 + Elastic IP + S3 + CloudWatch + API key, about 5-8 min
python -m tools.check_server --server http://<elastic-ip>:8000
.\cloud\teardown_aws.ps1             # remove it all when done
```
Full guide: **[docs/CLOUD_AWS.md](docs/CLOUD_AWS.md)**.

## Folder map

| Folder | What is inside |
|---|---|
| `esp32_node/` | **edge node firmware** (`esp32_node.ino`): belt, IR sensors, reject arm, LEDs, tiny-model engine, 3-second safety stop. No libraries needed. |
| `gateway_node/` | **gateway program**: `esp32_link.py` (USB link, pretend ESP32), `camera.py` (webcam), `gateway_worker.py` (small model), `cloud_client.py`, `netmon.py`, **`pipeline.py`** (box-by-box loop), `main.py` |
| `server/` | **cloud node**: `/infer`, `/ping`, dashboard, API-key auth, SQLite box log, model registry, S3/CloudWatch |
| `common/` | shared: state, reward, **Q-learning agent** (`rl.py`), baseline policies, **`tiny.py`** (the ESP32 model's file format and exact arithmetic), image I/O, fast algorithms |
| `sim/` | simulator of belt + ESP32 + gateway + network + cloud, **Q pre-training** (`train_q.py`), experiments E1-E6 |
| `vision/` | `train.py` (gateway and cloud models), **`train_tiny.py`** (ESP32 model), synthetic box generator |
| `models/` | `edge_tiny.bin` (ESP32), `gateway_model.onnx`, `cloud_model.onnx`, `q_table.npz`, `rule_params.json`, `model_card.json` |
| `tools/` | `go.py` (the double-click launcher), **`tradeoff.py`**, `esp32_check.py`, `run_experiment.py`, `report_real.py`, `check_server.py`, `capture_dataset.py`, `roi_check.py`, `calibrate.py` |
| `cloud/` | AWS deployment: `deploy_cloudshell.sh`, `deploy_aws.ps1`, `aws/cloudformation.yaml`, `Dockerfile` |
| `tests/` | automatic tests, including the **real firmware file compiled for the PC** and driven through the gateway's link code |
| `benchmark_stfg/` | second, public test on the Synthetic Task Flow Graphs dataset |
| `results/` | `TRADEOFF.md`, `SIM_RESULTS.md`, `E6_SENSITIVITY.md`, `STFG_RESULTS.md`, figures |
| `docs/` | ESP32_WIRING, ALGORITHM, VIVA_QA, DEMO_SCRIPT, PROJECT_OVERVIEW, CLOUD_AWS, EFFICIENT_ALGORITHMS |

## Re-train / re-run anything

| Command | Time |
|---|---|
| `python -m vision.train_tiny --data data/public` : ESP32 model | 1-2 min (CPU) |
| `python -m vision.train --role gateway --data data/public --init imagenet` : gateway model | about 3 min (CPU) |
| `python -m vision.train --role cloud --data data/public --init imagenet --img 160` : cloud model | about 13 min (CPU) |
| `python -m sim.train_q` : pre-train the 3-way agent + tune the baseline rules | about 1 min |
| `python -m sim.experiments` : simulation experiments E1-E5 | about 2 min |
| `python -m sim.sensitivity --data data/public` : other damage rates and costs (E6) | about 4 min |
| `python -m tools.tradeoff` : speed against accuracy at the three places | under 1 min |
| `python -m benchmark_stfg.bench` : task-graph benchmark | about 1 min |
| `python -m pytest -q tests` : all automatic tests | about 10 s |

## Results

<!-- README_RESULTS -->
**Speed against accuracy** (the same 594 photos at all three places, measured 2026-11-25 23:54)

| Place | Correct verdicts | Time per photo (median) | How it was measured |
|---|---|---|---|
| ESP32 (edge) | 65.3% | 77.5 ms | EMULATED (no board connected): cable time from the baud rate, compute time from 40 ns per multiply |
| Gateway (laptop) | 89.9% | 2.5 ms | this laptop, CPU |
| Cloud server (on this PC for now) | 95.1% | 8.1 ms | server on THIS PC: no Internet in the path, so the real cloud will be slower |

- Most accurate first: Cloud server (on this PC for now) 95.1%, Gateway (laptop) 89.9%, ESP32 (edge) 65.3%.
- Fastest first: Gateway (laptop) 3 ms, Cloud server (on this PC for now) 8 ms, ESP32 (edge) 77 ms.
- The ESP32 is both slower (77 ms against 3 ms) and less accurate than the gateway, because the photo has to be sent to it over the cable first. It is not there to win this table. It is there for when the gateway is too busy and the Internet is too slow: then it is the only place that still answers before the box reaches the arm.
- Going from the gateway to the cloud buys 5.2 points of accuracy for 6 ms more per box.
- The ESP32 time is an estimate until this is run with the board plugged in. Its accuracy is exact: the same arithmetic as the firmware.
- The cloud server ran on this same PC, so its time has no Internet in it. Run this again with --server http://<AWS address>:8000 for the real number.

**Second opinion on real photos** (594 test photos, public; measured 2026-11-25 23:37)

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
| Q-learning (frozen) | -2.57 | 0.0% | 19.0% | 3.2% | 21.9% |
| Q-learning (online) | -2.63 | 0.0% | 19.4% | 3.4% | 21.9% |
| Confidence rule | -2.83 | 0.0% | 8.1% | 3.7% | 23.3% |
| Confidence rule + reserve | -2.91 | 0.0% | 9.8% | 4.1% | 23.2% |
| Ask the cloud always | -2.95 | 0.0% | 23.5% | 4.0% | 23.5% |
| Adaptive rule | -3.25 | 0.8% | 10.3% | 5.0% | 23.7% |
| Always-Gateway | -5.46 | 7.3% | 0.0% | 11.6% | 17.3% |
| Always-Edge | -6.32 | 0.0% | 0.0% | 1.7% | 81.6% |
| Threshold rule | -6.74 | 11.6% | 9.3% | 15.5% | 16.0% |
| Deadline rule | -6.79 | 11.8% | 10.3% | 15.7% | 15.9% |
| Always-Cloud | -8.04 | 16.3% | 26.1% | 19.9% | 14.1% |

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
| Q-learning (frozen) | 1.40 | 1.40 | 1.56 | 1.98 | 2.16 |
| Q-learning (online) | 1.42 | 1.44 | 1.63 | 2.00 | 2.11 |
| Best possible (knows the answers) | 0.52 | 0.52 | 0.52 | 0.58 | 1.14 |

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
| Gateway overloaded while the network is bad | Q-learning (online) | 0.5% | 26 |
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

Average over all 10 situations: Q-learning (online) -2.63 against the best hand-written rule (Confidence rule) -2.83: **Q-learning ahead by 0.21**: real, but small. Against the best rule that picks ONE place per box (Adaptive rule, -3.25) Q-learning is ahead by 0.62. Figures: `results/figures/E1_steady_comparison.png`, `E2_change_in_middle.png`, `E3_pretraining_value.png`, `E4_policy_map.png`, `E5_sim_to_real_mismatch.png`.

Full tables: [results/TRADEOFF.md](results/TRADEOFF.md), [results/SIM_RESULTS.md](results/SIM_RESULTS.md)
<!-- /README_RESULTS -->

## Honest notes

- The ESP32 cannot run the gateway's model: it has about 0.3 MB of free memory. Its own tiny model is the
  least accurate of the three by design.
- The webcam is on the gateway, so a photo has to be sent to the ESP32 first. In normal conditions the
  ESP32 is therefore slower *and* less accurate than the gateway. It earns its place only when the gateway
  is overloaded and the Internet is slow: then it is the only checker that answers in time.
- Rows marked "emulated" or "on this PC" in the speed-accuracy table are not final: plug the ESP32 in and
  set up AWS, then run the measurement again.
- The decision experiments (E1-E6) are simulated. The real belt has not been measured.
- On average the Q-learning agent is level with a well-tuned adaptive rule. It does not beat it by a wide
  margin. Its advantage is that nobody has to tune it.
- Offloading with Q-learning is a known idea. The contribution here is the deadline-driven, cost-based
  design over three tiers and the test against a strong hand-written rule.
