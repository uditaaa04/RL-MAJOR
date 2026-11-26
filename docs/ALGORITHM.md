# The algorithm, in plain words

## 1. The decision

Three places can check the photo of a box. They are used together, as a **second opinion** system:

1. **The gateway's small model checks every box first** (about 2 ms). Its verdict is kept.
2. Its score also says **how sure it is** (far from the cut-off = sure, close to it = unsure).
3. **The Q-learning agent then decides:** keep that verdict, or also **ask the cloud**.
4. A cloud answer that arrives before the arm replaces the verdict. **A late cloud answer changes nothing:
   the gateway's verdict is used, so the box is still checked.**
5. Cloud calls are **limited** (so many per minute, `cloud_budget` in the settings). The agent has to spend
   them on the boxes that need them.
6. Only when the laptop is too busy to check boxes in time does the agent pick one place outright, and then
   the **ESP32** is a choice too.

Why this order: on 594 real test photos the gateway's and the cloud's model give the same verdict on about 90 %
of boxes. Sending every box to the cloud wastes nine calls in ten. The boxes where the cloud helps are almost
all boxes where the gateway's model is unsure (`results/CASCADE.md`).

The three places:

| Action | What happens | Good | Bad |
|---|---|---|---|
| **EDGE** (0) | the gateway shrinks the photo to a tiny grey picture and sends it over the USB cable to the **ESP32**, which runs its **tiny model** (three or four small convolutions, about 6,000 to 12,000 numbers) | needs no Internet and no laptop CPU; time is almost constant | least accurate; slower than the gateway because of the cable |
| **GATEWAY** (1) | the laptop checks the photo with its **small model** (MobileNetV3-Small, 128 px) | a few milliseconds, good accuracy | slow when the laptop is overloaded |
| **CLOUD** (2) | the photo goes to the cloud server's **big model** (MobileNetV3-Large, 160 px) | most accurate | crosses the Internet; the answer can arrive **too late** |

The answer must arrive **before the box reaches the reject arm** (about 0.3 to 0.7 s, depending on belt speed).
If it is late, the box passes **unchecked** ("missed"). With this project's costs a missed box is worse than a
weak answer, so "weak but in time" beats "perfect but late".

Why the ESP32 has its own model: an ESP32 has about 0.3 MB of free memory. The gateway's model needs many
megabytes, so it cannot run there. The tiny model fits in 24 to 47 kB.

## 2. What the agent sees: the state (10,800 situations)

The gateway measures seven things and puts each into a few bins:

| # | Feature | How it is measured | Bins |
|---|---|---|---|
| 1 | **Spare time if sent to the cloud** = time left before the arm − predicted cloud answer time | time left: belt speed (measured from IR-1 to IR-2 transit times). Predicted cloud time = max(measured cloud time, ping + 30 ms) | 8: `<-50`, `-50..0`, `0..50`, `50..100`, `100..150`, `150..250`, `250..400`, `>400` ms |
| 2 | **Measured cloud answer time** | a small **test image every 2 s** + every real request, smoothed | 3: fast (<150 ms) / slow / very slow (>400 ms) |
| 3 | **Ping time** | `GET /ping` every 0.5 s, smoothed | 3: good (<50 ms) / busy / bad (>200 ms) |
| 4 | **Can the gateway's own model finish in time?** | (boxes waiting + 1) × the gateway model's measured time, divided by the time left. The gateway times its own model on every box and on a blank photo every 2 s when idle | 3: fits easily (<0.4) / tight / too slow (≥1.0) |
| 5 | **Cloud queue** | requests already in flight | 2: 0 / 1+ |
| 6 | **How sure the gateway's model is about this box** | distance of its score from the cut-off. The borders between the levels come from the validation photos: the least-sure 10 % are "very unsure", the next 15 % "unsure", the next 25 % "fairly sure", the rest "sure" | 5: not run / sure / fairly sure / unsure / very unsure |
| 7 | **Cloud calls left right now** | the cloud-call budget (a bucket that refills at the allowed rate and holds up to 5 unused calls) | 5: none / 1 / 2 / 3-4 / 5 or more. "5 or more" is also what the agent sees when there is no limit: a call spent now is then not missed later |

8 × 3 × 3 × 3 × 2 × 5 × 5 = **10,800 states**, 3 actions. Many of them can never happen (for example "sure" together
with "gateway's model did not run"); about 3,700 are visited in training.

Feature 6 is what makes the decision **per box**: two boxes that arrive in the same network conditions can
get different decisions. Feature 7 is what makes it a real **sequence** problem: a call spent on this box is
missing for a harder box a few seconds later. That is why the next state matters (γ = 0.9).

The ESP32 needs no feature of its own: its answer time is nearly constant (cable time + a fixed amount of
arithmetic), and if the board is unplugged the gateway simply does not offer it as a choice.

Why "spare time" and not "time left" and "cloud time" separately? Because the decision depends on the
**difference**. With one feature for it, the agent learns for itself *how much spare time is safe enough*,
the number a hand-made rule needs a human to tune (its "safety margin").

Why the **test image**? A ping is tiny, so it cannot show that the cloud server is overloaded or that the
bandwidth collapsed. A real photo-sized test every 2 s can. (Cost: about 8 KB every 2 s.)

## 3. The reward: "minus the expected cost of this box"

```
                 ┌ on time:  - [ p_def × (1 − recall) × C_escape  +  (1 − p_def) × false_reject × C_false_reject ]
reward(box)  =   │
                 └ missed :  - [ p_def × C_escape + C_unchecked ]
                 − energy_weight × joules
```

| Symbol | Meaning | Value (config) |
|---|---|---|
| `p_def` | share of boxes expected to be damaged | 0.30 |
| `C_escape` | cost of a damaged box being shipped | 100 |
| `C_false_reject` | cost of throwing away a good box | 10 |
| `C_unchecked` | extra cost of a box passing unchecked | 5 |
| `recall`, `false_reject` | each of the three models' measured quality, **read from `models/model_card.json`** | from the test photos |
| `energy_weight` | per joule | 1.0 |

**With a second opinion** the same formula is used, but with the error rates **for boxes of that confidence
level**, and for the model that actually **answered**. Example from the validation photos (30 % damaged
boxes): where the gateway's model is "very unsure" its verdict costs about 4.7 per box and the cloud's about
1.8, so a cloud call is worth about 2.8 there. Where it is "sure" a cloud call is worth nothing (slightly
negative). These per-level numbers are stored in the model card (section `cascade`) by `tools/photo_scores.py`.
If the cloud was asked but answered too late, the reward is the gateway's cost for that level: the call
was spent for nothing, and the agent learns that.

The reward uses only what the gateway knows **at run time** (did an answer arrive in time? which model
answered? how sure was the gateway's model?). It does **not** need to know if the box was really damaged.
That is what makes online learning possible on a real line.

In the **experiments** the score that is reported is not this expected cost but the **real cost of what
happened to each box**, counted from the photo's true label: 100 if a damaged box was shipped, 10 if a good
box was thrown away, 5 plus the damage if it passed unchecked.

### Spot checks: the line corrects the model card

The per-level numbers above come from the validation photos. A real line drifts away from them: a lamp fails,
the belt runs faster and the photo smears, a new kind of box arrives. On such photos the cloud's model can lose
more than the gateway's (measured: `results/shift_scores.json`), and then the card's "a cloud call is worth 1.2
here" is simply wrong. Nothing in the photo itself says so.

So whenever the **true state of a box becomes known afterwards** (a test run with a known box order, or a person
checks some boxes by hand) and that box had **both verdicts**, the saving of that one cloud call is known exactly:

```
saving = cost of the gateway's verdict − cost of the cloud's verdict        (0 when they agree)
```

`common/cascade.py → LiveGain` keeps the running average of that saving per confidence level. It starts at the
card's value (the card counts as 60 checked boxes, `agent.spot_prior_n`) and old boxes fade (half their weight
after 300 newer ones, `agent.spot_half_life`). When the agent decides, it adds

```
(saving measured on this line − saving on the card) × (share of recent cloud calls answered in time)
```

to the value of "ask the cloud". The Q-table keeps what it learned about **timing and the price of a call**; what
an **answer is worth** now comes from the line. Without any known truth the tally never moves and the agent
behaves exactly as before. Other settings of the two numbers (30/200, 120/600) gave the same result on the
validation photos.

The same costs also choose each model's **decision cut-off** on the validation photos, so the vision side and
the RL side optimise the same goal. Because a shipped defect costs ten times a wasted good box, every
cut-off leans toward rejecting. For the weak ESP32 model this means: it catches almost every damaged box
but also throws away most good ones. That is still cheaper than letting boxes pass unchecked, which is
exactly why the agent uses it only as a last resort.

## 4. The learning rule: Q-learning

```
Q(s, a)  ←  Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

* `s` = state when box *i* arrived, `a` = the place chosen, `r` = its reward,
  `s'` = the state when box *i+1* arrived (the next decision).
* The reward of a box is known only when its answer arrives (or it passes the arm), so the update happens
  **as soon as both `r` and `s'` are known**, whichever comes last (`common/policies.py → QPolicy`).
* `γ = 0.9`: the next boxes matter, because a cloud call spent now is missing for a harder box a few
  seconds later, and a choice changes the queues the next box will find.
* If the agent picks the ESP32 but the board is not ready, the gateway checks the box and the agent learns
  from what was really done.

## 5. Two-stage training

**Stage 1: pre-training in the simulator** (`python -m sim.train_q`, about 2 minutes)
* 16,000 episodes × 150 boxes = **2.4 million simulated boxes**.
* **Every simulated box is one real photo.** `tools/photo_scores.py` scores all validation and test photos with
  the three models and stores the scores (`models/photo_scores.npz`). In the simulator each model's verdict on
  a box is its real verdict on that photo. The agent trains on the validation photos and is judged on the test
  photos, and every policy is judged on the very same sequence of boxes.
* Every episode gets random conditions (good/busy/bad/terrible network, belt speed, box spacing, gateway
  load from normal to choked, a cloud-call limit from none to very tight, hidden faults such as an overloaded
  cloud server or a bandwidth cap), sometimes changing halfway through ("domain randomisation").
* Learning rate = 1/(visits)^0.6 per (state, action), which averages out simulator noise.
* Exploration starts at 50 % random and decays to 5 %.
* Before learning, the gateway is assumed to be the safe choice ("safe prior"): the cloud has to *earn*
  trust, and the ESP32 starts at what its lower accuracy is worth.

**Stage 2: online learning on the conveyor** (always on, `agent.learn_online: true`)

| Setting | Value | Meaning |
|---|---|---|
| α | 0.1 | learning rate |
| ε | 0.02 | 2 % of the time, try another place ... |
| explore_gap | 3.0 | ... but **only one whose value is close to the best** (*safe exploration*: never gamble a box on a place it is sure is bad) |
| forget | 0.02 | places *not* taken drift back 2 % toward their pre-trained value each visit ("bad memories fade"), so after a fault clears the agent re-checks them |

## 6. The baselines it is compared with (all in `common/policies.py`)

| Policy | Logic | Tuned numbers |
|---|---|---|
| Always-Edge | always the ESP32 | none |
| Always-Gateway | always the gateway | none |
| Always-Cloud | always the cloud | none |
| Threshold rule | cloud if ping < T, else gateway | T |
| Deadline rule | fixed time estimates for all three places; the most accurate place that fits in the time left (+ margin) | margin |
| Adaptive rule | the same, but with the **measured** cloud, gateway and ESP32 times | margin |

Those six pick **one place per box** (the earlier design). Three hand-written rules work the second-opinion
way, like the agent:

| Policy | Logic | Tuned numbers |
|---|---|---|
| Ask the cloud always | gateway first, then always ask the cloud while calls are left | none |
| Confidence rule | gateway first; ask the cloud when its model is at least "unsure" (tuned level) and the measured cloud time fits | level, margin |
| Confidence rule + reserve | the same, but when the budget runs low only "very unsure" boxes are sent | level, reserve, margin |
| Best possible (experiments only) | knows which boxes a cloud answer would fix. No real policy can know that; it shows how much room is left | none |

All rules are tuned **on the same randomised conditions as the agent's pre-training**
(`sim/train_q.py → tune_rules`), so the comparison is fair. The tuned values are in `models/rule_params.json`.

The confidence rules are deliberately strong: they use the same confidence signal as the agent. In the
experiments the agent ends up **slightly ahead of them on average and clearly ahead of every pick-one rule**.
On pure budget questions the confidence rule and the agent are level. The agent's advantage is that nobody
has to choose its level, margin and reserve, and that it keeps adjusting them on the line.

What the agent learns about spending calls (good network, from the trained table):

| Gateway's model is | no call left | 1 left | 2 left | 3-4 left | 5 or more left |
|---|---|---|---|---|---|
| sure | keep | keep | keep | keep | keep |
| fairly sure | keep | keep | keep | keep | ask |
| unsure | keep | keep | ask | ask | ask |
| very unsure | keep | ask | ask | ask | ask |

Nobody wrote that staircase. It comes out of the rewards. "Sure" boxes are never sent: on the validation
photos a cloud answer there is worth slightly less than nothing.

### What the two newest experiments say (`results/SIM_RESULTS.md`, E8 and E9)

* **E9, how sure the gaps are.** The whole comparison was repeated on 200 re-drawn test sets. The gap between
  the second-opinion way and the best pick-one rule is real (its 95 % range does not cross zero). The gap between
  Q-learning and the confidence rule is **not**: Q-learning is cheaper in about nine sets out of ten, but the
  range crosses zero. Say "level, slightly ahead on average", never "better".
* **E8, the photos get worse in the middle of a run.** On darker or blurred photos, asking the cloud for every
  box becomes the most expensive choice, even with no call limit, and the agent trained on clean photos asks too
  often as well. With spot checks it corrects itself within about a thousand boxes and ends level with the
  confidence rule, which was already safe because it only asks for unsure boxes. With a tight call limit
  nobody over-asks, so spot checks change little there. The darker / blurred photos are made by software from
  the public photos; they stand in for a real change, they are not measurements from the belt.

## 7. The tiny model on the ESP32 (`vision/train_tiny.py`, `common/tiny.py`, `esp32_node/esp32_node.ino`)

* Input: the box photo shrunk to 32 × 32 or 48 × 48 grey pixels (1 to 2.3 kB), sent over USB at 460,800 baud
  (about 22 to 50 ms).
* Network: 3 × 3 convolutions with ReLU, an average over the picture, one output. Batch-norm is folded into
  the convolutions before export, so the ESP32 only multiplies and adds.
* The model is **not inside the firmware**. The gateway sends the file to the board at every start, with a
  checksum, so re-training never needs a re-flash.
* `common/tiny.py` does the same arithmetic on the PC. The automatic tests compile the real firmware file
  for the PC and check that both give the same answer to four decimals.
* Ten shapes were tried on the public photos. Wider layers and colour did not help. The limit is the
  picture size: about 65 to 72 % correct, against about 90 % for the gateway model and 95 % for the cloud
  model. That gap is the edge end of the speed-accuracy trade-off.

## 8. Why tabular Q-learning (and not DQN / PPO)?

* The state is small (10,800 situations, about a third of them ever reached): a table is enough and
  converges with 2.4 million simulated boxes in about three minutes.
* A table lookup takes microseconds and needs no GPU.
* It is **explainable**: you can open the table and show *why* it chose a place (the dashboard's
  "What the agent has learned" panel and `results/figures/E4_policy_map.png`).
* Phase-I already used tabular Q-learning; this phase makes the state, the reward and the training rigorous
  and adds the third place, instead of changing the algorithm family.
