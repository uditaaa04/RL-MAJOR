# Smart Conveyor QC: edge, gateway and cloud orchestration with reinforcement learning

Boxes travel on a conveyor. A webcam photographs each one. **Three places** can judge the photo:

| Node | Hardware | Model | Strength | Weakness |
|---|---|---|---|---|
| **Edge** | ESP32 on the belt | tiny CNN, about 6,000 numbers, grey 32 or 48 px photo | needs no Internet and no laptop CPU; also runs the belt | least accurate; the photo must be sent to it over USB |
| **Gateway** | laptop next to the belt | MobileNetV3-Small, 128 px | fast, good accuracy | slow when the laptop is overloaded |
| **Cloud** | Render web service (Singapore), live at https://smart-conveyor-cloud.onrender.com | MobileNetV3-Large, 160 px | most accurate | the photo crosses the Internet; the answer can come too late |

They work as a **second opinion** system. The gateway's model checks **every box first** (about 2 ms). A
**Q-learning agent on the gateway** then decides whether the cloud should check it too, using how sure the
gateway's model is, the time left, the network, and how many **cloud calls** are left (they are limited). A
cloud answer that comes too late changes nothing: the gateway's verdict is used, so the box is still checked.
When the laptop is too busy to check boxes in time, the agent hands them to the ESP32.

What a cloud answer is worth was first measured on the training photos. A real line drifts away from those,
so when the true state of a box is known afterwards (a test run with a known box order, a person checking
some boxes) the agent does a **spot check** and keeps its own tally from the line.

**What is claimed, and what is not.** Over 200 re-drawn test sets the second-opinion design is clearly cheaper
than every rule that picks one place per box. Q-learning against a hand-tuned confidence rule is **level**:
ahead on average, inside the range of chance. Everything about the belt, the network and the timing is
simulated; the photos and the models' verdicts are real. Nothing has run on real hardware yet.

```
 ESP32 (edge)  <---- USB cable ---->  laptop (gateway)  <---- Internet (https) ---->  Render (cloud)
 belt motor, 2 IR sensors,            webcam, Q-learning agent,                       big model,
 reject arm, tiny model               small model, link to both sides                 dashboard, box log
      \_________________ ESP32's own Wi-Fi (firmware 4): sensors, belt, arm ________/
```
Every finished box, sensor change and dashboard button travels at once (long polling, no 1-second timer):
about 0.1 s from the belt to the dashboard over the Internet.

The project's experiment is the **speed-accuracy trade-off between the three places**
(`MEASURE_SPEED_VS_ACCURACY.bat`, table below), what a cloud second opinion is worth on real photos
(`results/CASCADE.md`), and how well the agent uses both when the network, the gateway or the cloud-call
budget gets tight.

## Fastest start (Windows, nothing to type)

Double-click **`START_DEMO.bat`**: no hardware needed. All double-click files and the hardware order are in
**[START_HERE.md](START_HERE.md)**. Building the belt: **[docs/hardware/](docs/hardware/)** (step-by-step guide and
circuit diagram, checked against the firmware). Flashing the ESP32: **[docs/ESP32_WIRING.md](docs/ESP32_WIRING.md)**.

## The same by hand

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements-laptop.txt

python -m server.app                       # terminal 1: the cloud node, here on this PC -> http://localhost:8000
python -m gateway_node.main --mock         # terminal 2: gateway with a pretend belt and a pretend ESP32
python -m gateway_node.main --mock --esp32 auto    # ... pretend belt, tiny model on a real ESP32 (USB)
python -m gateway_node.main --server https://<name>.onrender.com           # the real line: ESP32 + webcam + cloud
python -m gateway_node.main --manual --server https://<name>.onrender.com  # hand test: belt off, boxes moved by hand
python -m tools.tradeoff --esp32 auto --server https://<name>.onrender.com # speed against accuracy, all real
python -m tools.flash_esp32                                                # put firmware 4 (Wi-Fi) on the ESP32
```

## The cloud node (live on Render)

The cloud server runs on a free Render web service in Singapore: **https://smart-conveyor-cloud.onrender.com**
(dashboard; every API call needs the key). It deploys itself from this repository (`render.yaml`, branch
`claude/smart-conveyor-project`). The laptop keeps the address and key in `cloud/.secrets/aws_server.json`
(never uploaded). Measured from the gateway laptop in India: ping 71 ms, a box photo checked in the cloud
83 ms (95th percentile 97 ms), of which 5.2 ms is the model itself. Free-plan limits (sleeps after 15 minutes
without requests, box log wiped on restart) and the full guide: **[docs/CLOUD_RENDER.md](docs/CLOUD_RENDER.md)**.

The AWS version (EC2 + S3 + CloudWatch, `cloud/deploy_cloudshell.sh`, `cloud/deploy_aws.ps1`) is still included
as an alternative: **[docs/CLOUD_AWS.md](docs/CLOUD_AWS.md)**.

## Live data and hand tests

- `FLASH_ESP32.bat`: puts firmware 4 on the ESP32 (no Arduino IDE clicking). The board then joins the Wi-Fi in
  `esp32_node/secrets.h` (never uploaded; copy `secrets.example.h`), or opens its own setup hotspot with a setup
  page at http://192.168.4.1 when no Wi-Fi works. Online, it has its own live page (`http://<name>.local`) and
  sends its sensors, belt and arm to the cloud dashboard by itself.
- `MANUAL_TEST.bat`: real ESP32, sensors, arm and webcam, belt motor off. Move each box by hand past IR sensor 1
  (photo + decision) and IR sensor 2 (the arm pushes a damaged box). Every box appears on the dashboard live.
- Dashboard panel **The ESP32 on the belt, live**: both IR sensors light up as they see a box, belt, arm pushes,
  USB and Wi-Fi state, and the delay from the gateway to the screen. Buttons: belt on/off, push the arm, check a
  box now.

## Folder map

| Folder | What is inside |
|---|---|
| `esp32_node/` | **edge node firmware** (`esp32_node.ino`, version 4): belt, IR sensors, reject arm, tiny-model engine, 3-second safety stop; Wi-Fi with setup hotspot, live page and readings to the cloud (`secrets.example.h`, `root_ca.h`). No extra libraries needed. |
| `gateway_node/` | **gateway program**: `esp32_link.py` (USB link, pretend ESP32), `camera.py` (webcam), `gateway_worker.py` (small model), `cloud_client.py`, `netmon.py`, **`pipeline.py`** (box-by-box loop), `main.py` |
| `server/` | **cloud node**: `/infer`, `/ping`, dashboard with instant updates (long polling), API-key auth, SQLite box log, model registry, `/api/edge` (the ESP32 over Wi-Fi), S3/CloudWatch on AWS |
| `common/` | shared: state, reward, **Q-learning agent** (`rl.py`), baseline policies, **`tiny.py`** (the ESP32 model's file format and exact arithmetic), image I/O, fast algorithms |
| `sim/` | simulator of belt + ESP32 + gateway + network + cloud, **Q pre-training** (`train_q.py`), experiments E1-E9 |
| `vision/` | `train.py` (gateway and cloud models), **`train_tiny.py`** (ESP32 model), synthetic box generator |
| `models/` | `edge_tiny.bin` (ESP32), `gateway_model.onnx`, `cloud_model.onnx`, `q_table.npz`, `rule_params.json`, `model_card.json` |
| `tools/` | `go.py` (the double-click launcher), **`tradeoff.py`**, `esp32_check.py`, `run_experiment.py`, `report_real.py`, `check_server.py`, `capture_dataset.py`, `roi_check.py`, `calibrate.py` |
| `cloud/` | Render (`requirements-render.txt`; the Blueprint is `render.yaml` in the top folder) and AWS deployment: `deploy_cloudshell.sh`, `deploy_aws.ps1`, `aws/cloudformation.yaml`, `Dockerfile` |
| `tests/` | automatic tests, including the **real firmware file compiled for the PC** and driven through the gateway's link code |
| `benchmark_stfg/` | second, public test on the Synthetic Task Flow Graphs dataset |
| `results/` | `TRADEOFF.md`, `SIM_RESULTS.md`, `E6_SENSITIVITY.md`, `STFG_RESULTS.md`, figures |
| `docs/` | ESP32_WIRING (also flashing and Wi-Fi), ALGORITHM, VIVA_QA, DEMO_SCRIPT, PROJECT_OVERVIEW, CLOUD_RENDER, CLOUD_AWS, EFFICIENT_ALGORITHMS, and `hardware/` (build guide PDF + circuit diagram) |

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
- The cloud server ran on this same PC, so its time has no Internet in it. Measured since then through the Internet on Render: 83 ms per box photo from the gateway laptop (`tools.check_server`). Run this again with the board plugged in for the full real table.

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

Average over all 10 situations: Q-learning (online) -2.64 against the best hand-written rule (Confidence rule) -2.83: **Q-learning ahead by 0.19 on these photos; over 200 re-drawn test sets the gap is 0.17 (95% range -0.08 to +0.42): level, within chance.** Against the best rule that picks ONE place per box (Adaptive rule, -3.25) Q-learning is ahead by 0.61. Figures: `results/figures/E1_steady_comparison.png`, `E2_change_in_middle.png`, `E3_pretraining_value.png`, `E4_policy_map.png`, `E5_sim_to_real_mismatch.png`.

Full tables: [results/TRADEOFF.md](results/TRADEOFF.md), [results/SIM_RESULTS.md](results/SIM_RESULTS.md)
<!-- /README_RESULTS -->

## Honest notes

- The ESP32 cannot run the gateway's model: it has about 0.3 MB of free memory. Its own tiny model is the
  least accurate of the three by design.
- The webcam is on the gateway, so a photo has to be sent to the ESP32 first. In normal conditions the
  ESP32 is therefore slower *and* less accurate than the gateway. It earns its place only when the gateway
  is overloaded and the Internet is slow: then it is the only checker that answers in time.
- Rows marked "emulated" or "on this PC" in the speed-accuracy table are not final: plug the ESP32 in and run
  the measurement again (the cloud row then goes over the Internet to Render).
- The decision experiments (E1-E6) are simulated. The real belt has not been measured.
- On average the Q-learning agent is level with a well-tuned adaptive rule. It does not beat it by a wide
  margin. Its advantage is that nobody has to tune it.
- Offloading with Q-learning is a known idea. The contribution here is the deadline-driven, cost-based
  design over three tiers and the test against a strong hand-written rule.
