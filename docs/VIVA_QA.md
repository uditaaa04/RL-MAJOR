# Viva questions, with honest answers (second-opinion version)

How to read the numbers:

* **(photos)** = measured on real box photos the models never saw during training (public BoxCondition set).
* **(sim)** = from the simulator. Every simulated box is one of the 594 real test photos with the models' real
  verdicts, but the belt, the network and the timing are simulated.
* **(est.)** = an estimate, not a measurement.

Nothing in this file was measured on a real conveyor. Say that first if you are asked.

---

## A. The project in one minute

### 1. "What does your project do, in one sentence?"
A laptop next to a conveyor checks every box's photo for damage in about 2 ms, and a Q-learning agent then
decides, box by box, whether a bigger model in the cloud should check it too, with a limited number of cloud
calls; an ESP32 runs the belt and takes over the checking when the laptop is overloaded.

### 2. "What are the three nodes and what does each one do?"
| Node | Hardware | Job | Model |
|---|---|---|---|
| Edge | ESP32 on the belt | two IR sensors, belt motor, reject arm, safety stop, **and** a last-resort damage check | tiny CNN, 11,705 numbers, 48 x 48 grey picture |
| Gateway | laptop | USB webcam, talks to both sides, runs the Q-learning agent, **checks every box first** | MobileNetV3-Small, 128 px |
| Cloud | Render web service (Singapore) | **second opinion** on the boxes it is asked about, dashboard | MobileNetV3-Large, 160 px |

### 3. "What do you mean by second opinion?"
1. The laptop's model checks every box first. Its verdict is kept.
2. Its score shows how sure it is (far from the cut-off = sure).
3. The agent decides: keep the verdict, or also ask the cloud.
4. If the cloud answers before the box reaches the arm, its verdict is used. If it is late, the laptop's
   verdict stands. **The box is checked either way.**
5. Cloud calls are limited (15 a minute by default), so they have to go to the boxes that need them.

### 4. "Why not send every box to the best model?"
Because on 594 real test photos the laptop's and the cloud's model give **the same verdict on
89.7 %** of boxes (photos). The cloud corrects the laptop on 7.9 % and makes it worse on
2.4 %. Sending everything wastes about nine calls in ten, and each call costs money, bandwidth and time.

### 5. "What changed from the earlier version, where the agent picked one of three places?"
* Before: one place per box. If the cloud was picked and answered late, the box passed unchecked.
* Now: the laptop always checks first, so a slow cloud can no longer cause a missed box.
* New in the state: how sure the laptop's model is about this box, and how many cloud calls are left.
* Kept: Q-learning, the cost-based reward, the three models, the ESP32 firmware, the dashboard, the experiments.
* The pick-one policies are still in the experiments, as the comparison.

---

## B. Speed against accuracy

### 6. "Show me the trade-off."
Same 594 photos through all three places (photos; `results/TRADEOFF.md`):

| Place | Correct | Damaged boxes caught | Good boxes wrongly rejected | Time per photo |
|---|---|---|---|---|
| ESP32 (edge) | 65.3 % | 98.0 % | 81.2 % | about 77 ms (ESTIMATE, no board was plugged in) |
| Gateway (laptop) | 89.9 % | 96.0 % | 18.8 % | 1.5 ms (measured on this laptop) |
| Cloud | 95.1 % | 98.0 % | 9.0 % | 5.7 ms (server on the same PC, so NO Internet in this number) |

### 7. "The ESP32 is slower AND less accurate than the laptop. That is not a trade-off."
Correct, and we say so. The only camera is a USB webcam on the laptop, so the photo has to travel to the ESP32
over a cable first, and the chip is small. No model can fix that. The real trade-off in our numbers is
**laptop against cloud**. The ESP32's value is different: it does not depend on the laptop's processor or on the
Internet. When the laptop is overloaded and the network is bad together, it is the only place that still
answers in time (see "When is the ESP32 actually chosen?").

### 8. "65.3 % correct is close to guessing. Is the ESP32 model any use?"
It is weak: 48 x 48 grey pixels do not show small damage. Even on its most confident boxes it is wrong about one time
in five, so it cannot be used as a first filter. Because a shipped damaged box costs ten times a wasted good
one, its cut-off is set very low: it catches 98.0 % of damaged boxes but throws away 81.2 % of good ones.
That is still cheaper than no check at all: about 6.3 per box against 35 for a box that passes unchecked.
So it is a last resort and nothing more.

### 9. "Are the ESP32 time and the cloud time real?"
* ESP32: no. No board was connected. The time is worked out from the cable speed (460,800 baud) and 40 ns per multiply. The accuracy is exact: the PC does the same arithmetic as the firmware and a test checks they agree. `MEASURE_SPEED_VS_ACCURACY.bat` with the board plugged in replaces the estimate.
* Cloud: the 5.7 ms in the table, no: the server ran on the same laptop. Since then it runs on Render (Singapore) and was measured from the gateway laptop over the Internet: 83 ms per box photo (95th percentile 97 ms), of which 5.2 ms is the model and the rest the network.

---

## C. The second opinion on real photos

### 10. "Which boxes are worth a cloud call?"
The 594 test photos, sorted by how sure the laptop's model is (photos; levels set on 588 other photos; `results/CASCADE.md`):

| Laptop's model is | Share of boxes | Laptop right | Cloud right | Cost saved per cloud call |
|---|---|---|---|---|
| very unsure | 12 % | 49.0 % | 80.4 % | +4.54 |
| unsure | 16 % | 79.5 % | 90.4 % | +3.04 |
| fairly sure | 27 % | 93.3 % | 97.0 % | +0.33 |
| sure | 45 % | 97.0 % | 98.2 % | +0.26 |

Almost all of the benefit is in the two unsure levels.

### 11. "How much cloud traffic does that save?"
Send only the least-sure boxes to the cloud (photos, no belt and no agent here):

| Sent to the cloud | 0 % | 10 % | 25 % | 50 % | 100 % |
|---|---|---|---|---|---|
| Correct verdicts | 89.9 % | 92.8 % | 94.3 % | 94.8 % | 95.5 % |
| Same share picked at random | 89.9 % | 90.5 % | 91.3 % | 92.7 % | 95.5 % |

A quarter of the calls gives almost all of the cloud's benefit. With 594 photos a gap of about 1.5 points is
within chance, so "almost all" is the honest wording, not "all".

### 12. "Why limit the cloud calls? Is that realistic?"
A cloud call costs money (the server), bandwidth (a shared uplink, several belts) and energy. The limit is a
setting (15 a minute by default, changeable on the dashboard while the line runs). It is our choice, not a
law of nature, and we say so.

### 13. "And with no limit?"
On the normal photos the simple policy "always ask the cloud too" is then about as good as anything, because
the laptop's verdict is always there as a fallback. In E7 with no limit on a good network it costs 1.29 per box
and Q-learning 1.42 (sim). That stops being true when the photos get harder: on blurred photos
"always ask" costs 4.02 and the confidence rule 3.00 (sim, E8), because there the cloud's model
loses more than the laptop's. **So "always ask" is only safe while the cloud's model really is the better one
on every kind of box.**

---

## D. How the nodes talk

### 14. "How does the laptop talk to the ESP32?"
One USB cable, serial at 460,800 baud, short text lines. The laptop sends: keep-alive with belt speed, push
the arm, settings, the model file, a photo. The ESP32 sends: "box under the camera", "box at the arm",
and the answer to each photo. Every block of bytes has a checksum.

### 15. "What if the laptop crashes, or the ESP32 resets?"
The ESP32 stops the belt after 3 seconds without a message. If the ESP32 resets, the laptop notices, sends the
model file again and carries on; while the board is not ready it is not offered as a choice.

### 16. "How does the laptop talk to the cloud?"
HTTP with an API key on every request: the photo as a JPEG to `/infer`, a ping every 0.5 s, a photo-sized test
every 2 s, a Q-table backup every 60 s.

### 17. "Is the data on the dashboard real time?"
Yes. Every box is reported when it passes the arm: what the laptop's model said, how sure it was, whether the
cloud was asked, whether it answered in time, the verdict, the timing. On the real line these come from the
ESP32's sensors, the webcam and the cloud. Anything pretend (no board plugged in, pretend belt) is labelled.

### 18. "How is the belt wired?"
ESP32 DevKit V1 on the laptop's USB (power and data). Two IR sensors on GPIO 32 and 33, the reject servo on
GPIO 18 with its own 5 V from a buck converter, the 12 V belt motor through an HW-039 (BTS7960) driver whose
speed input is GPIO 25. There are no good / damaged lamps: the dashboard shows every verdict (the blue light
on the board shows the link to the laptop). Sensors and driver logic run on 3.3 V, every ground is
joined, 12 V never touches the ESP32. The wiring guide was checked pin by pin against the firmware
(`docs/hardware/`). With only one IR sensor the arm can be fired by time instead (`conveyor.arm_sensor: false`).
Not yet built: say so.

---

## E. The Q-learning part

### 19. "What are the state, the actions and the reward?"
* **Actions (3):** keep the laptop's verdict, ask the cloud too, ESP32 (only when the laptop is too slow to go first).
* **State (10,800 situations):** spare time if sent to the cloud (8 levels), measured cloud answer time (3),
  ping (3), can the laptop finish in time (3), cloud queue (2), **how sure the laptop's model is about this
  box (5)**, **cloud calls left right now (5: none, 1, 2, 3-4, 5 or more; "5 or more" also stands for "no limit")**.
* **Reward:** minus the expected cost of the box, for the model that answered and for that confidence level.
  A shipped damaged box costs 100, a wasted good box 10, a box that passes unchecked the expected damage
  plus 5. A cloud call is worth about 2.8 on a "very unsure" box, 1.2 on an "unsure" one and about -0.3 on a "sure" one.

### 20. "Where is the sequential decision? Each box looks independent."
The cloud-call budget. A call spent on this box is missing for a harder box a few seconds later. That is why
the next state matters (discount 0.9) and why this is Q-learning and not a one-shot choice. Without a budget
it would be nearly a one-shot problem, and we say so.

### 21. "What did it learn?"
This is the trained table for a good network (rows: how sure the laptop's model is; columns: cloud calls left):

| Laptop's model is | none left | 1 left | 2 left | 3-4 left | 5 or more left |
|---|---|---|---|---|---|
| very unsure | keep | **ask** | **ask** | **ask** | **ask** |
| unsure | keep | keep | **ask** | **ask** | **ask** |
| fairly sure | keep | keep | keep | keep | **ask** |
| sure | keep | keep | keep | keep | keep |

Nobody wrote that pattern. It comes out of the rewards: the scarcer the calls, the more unsure the laptop has
to be before a call is spent.

### 22. "The laptop does not know if a box was really damaged. How can it learn online?"
It does not need to. The reward only uses what is known at run time: which model answered, whether it was
in time, and how sure the laptop's model was. Each model's quality per confidence level is measured once on
the validation photos and stored in the model card.

That has a weak spot, and we closed it: the card is only right for photos like the training photos. So when the
true state of a box IS known afterwards (a test run with a known box order, or a person checks some boxes),
the agent does a **spot check**: it compares the laptop's and the cloud's verdict on that box and keeps its own
tally of what a cloud call saved, per confidence level. That tally replaces the card's number. Without known
truth it changes nothing.

### 23. "How was it trained?"
First in a simulator: 16,000 episodes of 150 boxes (2.4 million boxes, about two minutes). Every simulated box
is a real validation photo with the models' real verdicts; the network, belt speed, laptop load, cloud-call
limit and hidden faults are random. It visited 3,660 of the 10,800 situations (many cannot happen). Then it
keeps learning on the line, carefully: 2 % exploration, and only among options whose values are close.

### 24. "Why a table and not a neural network (DQN, PPO)?"
Seven measured features with a few levels each fit in a table. A table trains in two minutes, decides in
microseconds, and can be read: the dashboard shows what it would do in each situation.

### 25. "Why can a simple rule not do this?"
A good rule nearly can, and we tested that. The **confidence rule** ("ask the cloud when the laptop is at least
unsure and the measured cloud time fits") uses the same confidence signal as the agent.
* Over 10 situations it costs 2.83 per box, Q-learning 2.64 (sim). Q-learning is ahead by 0.19.
* Against the best hand-written policy overall (Confidence rule): clearly ahead in 6 situations (busy, fast+busy, gateway-busy, server-slow, lossy, wifi-stalls), level in 3, behind in 1 (good).
* Is that gap real? We repeated the comparison on 200 re-drawn test sets (E9). Q-learning was cheaper in
  91 % of them, but the 95 % range of the gap is -0.42 to +0.08: it crosses zero. **So the honest
  word is "level", not "better".**
* On pure budget questions (E7) the two are level as well.
* What Q-learning adds: nobody has to tune its confidence level, time margin and reserve (the rule's were found
  by search), and it keeps adjusting on the line.

---

## F. Results

### 26. "What are your main results?"
All simulated, 10 situations, 15 cloud calls a minute, every box a real test photo, cost = what really
happened to the box (lower is better):

| Method | Cost per box | Boxes missed | Boxes sent to the cloud | Correct verdicts |
|---|---|---|---|---|
| Q-learning frozen | 2.60 | 0.0 % | 18 % | 83.6 % |
| Q-learning learning online | 2.64 | 0.0 % | 20 % | 83.6 % |
| Confidence rule | 2.83 | 0.0 % | 8 % | 82.4 % |
| Confidence rule + reserve | 2.91 | 0.0 % | 10 % | 82.4 % |
| Ask the cloud always | 2.95 | 0.0 % | 24 % | 82.2 % |
| Adaptive rule | 3.25 | 0.8 % | 10 % | 81.2 % |
| Always-Gateway | 5.46 | 7.3 % | 0 % | 79.2 % |
| Always-Edge | 6.32 | 0.0 % | 0 % | 41.8 % |
| Threshold rule | 6.74 | 11.6 % | 9 % | 75.9 % |
| Deadline rule | 6.79 | 11.8 % | 10 % | 75.8 % |
| Always-Cloud | 8.04 | 16.3 % | 26 % | 72.6 % |

* Q-learning against the best rule that picks **one place per box** (adaptive rule): 3.25 down to 2.64, 19 % lower.
* Q-learning against the best hand-written policy of any kind (Confidence rule): ahead by 0.19, 7 % lower.
* Missed boxes: 0.8 % with the adaptive rule, 0.0 % with any second-opinion policy.

### 27. "What does the fallback buy?"
Wi-Fi that stalls now and then, a fault never seen in training: the adaptive pick-one rule missed
4.5 % of boxes, the second-opinion policies 0.0 % (sim). A late cloud answer simply does not matter any more.

### 28. "When is the ESP32 actually chosen?"
Only when the laptop is overloaded. In that test (bad network too) the agent sent 100 % of boxes to the
ESP32 and missed 0.2 %; the threshold and deadline rules, which cannot use it, missed 73 % (sim).
The adaptive rule also switches to the ESP32 (0.3 % missed). This case needs the third node, not Q-learning.

### 29. "What happens with fewer cloud calls?"
Good network, about 60 boxes a minute, cost per box (sim, E7):

| Policy | no limit | 30 a minute | 15 a minute | 8 a minute | 4 a minute |
|---|---|---|---|---|---|
| Adaptive rule | 1.23 | 1.78 | 2.14 | 2.24 | 2.33 |
| Ask the cloud always | 1.29 | 1.88 | 2.04 | 2.23 | 2.30 |
| Confidence rule | 1.45 | 1.45 | 1.57 | 1.91 | 2.08 |
| Confidence rule + reserve | 1.40 | 1.45 | 1.70 | 1.87 | 2.14 |
| Q-learning (frozen) | 1.40 | 1.44 | 1.55 | 1.87 | 2.08 |
| Q-learning (online) | 1.42 | 1.48 | 1.62 | 1.94 | 2.13 |
| Best possible (knows the answers) | 0.52 | 0.52 | 0.52 | 0.58 | 1.14 |

"Best possible" knows which boxes a cloud answer would fix. No real policy can know that; it shows the room
that is left. Between the confidence rules and Q-learning the winner changes from column to column: level.

### 30. "30 % damaged boxes is unrealistic. Other costs?"
Tested (`results/E6_SENSITIVITY.md`, sim). For each setting every model got a new cut-off, the agent was
re-trained and the rules were re-tuned.

| Setting | Q-learning online | Best rule |
|---|---|---|
| 1% damaged | 0.58 | Confidence rule: 0.61 |
| 5% damaged | 1.33 | Confidence rule: 1.31 |
| 30% damaged (base) | 2.58 | Confidence rule + reserve: 2.72 |
| base, wasted good box costs 3x | 4.84 | Confidence rule + reserve: 4.89 |
| base, escaped defect costs 3x | 4.23 | Confidence rule + reserve: 4.41 |

Q-learning was ahead in 3 of 5 settings, level in 2, behind in 0.

### 31. "What if the simulator is wrong about the real hardware?"
Tested with hardware made 2 to 3 times slower than the simulator assumed (E5, sim): adaptive rule
3.23, Q-learning online 2.76. Nothing was re-tuned. Both use live measurements, so neither breaks;
the gap is mostly the second-opinion fallback again.

### 32. "Your gaps are small. How do you know they are not luck?"
We measured it (E9, sim). The 594 test photos are one draw out of many possible ones, so we re-drew the test
set 200 times (same size, photos picked with repeats) and repeated the whole 10-situation comparison each time.

| Cost per box, first minus second | Gap | 95 % range | Verdict |
|---|---|---|---|
| Q-learning (online) vs Adaptive rule | -0.58 | -1.00 to -0.26 | real |
| Q-learning (online) vs Ask the cloud always | -0.28 | -0.57 to -0.02 | real |
| Q-learning (online) vs Confidence rule | -0.17 | -0.42 to +0.08 | level, within chance |
| Q-learning (frozen) vs Confidence rule | -0.20 | -0.48 to +0.05 | level, within chance |
| Confidence rule vs Adaptive rule | -0.41 | -0.67 to -0.16 | real |

What we claim: the second-opinion design beats every rule that picks one place per box. What we do not
claim: that Q-learning beats the confidence rule.

### 33. "The models were trained on clean photos. What if the photos on the line get worse?"
Tested (E8, sim). We made the same test photos darker, blurred, or both by software and scored them again with
the same three models (photos):

| Photos | ESP32 | Laptop | Cloud |
|---|---|---|---|
| normal | 65.3 % | 89.9 % | 95.5 % |
| blurred | 66.2 % | 86.0 % | 89.6 % |
| darker | 58.8 % | 89.1 % | 92.1 % |
| darker and blurred | 58.8 % | 74.4 % | 75.9 % |

The cloud's model loses more than the laptop's. Then, in the simulator, the photos change after 500 boxes and
nothing is re-tuned. Cost per box over boxes 2001 to 3500 after the change, **no call limit**:

| Policy | blurred | darker | darker and blurred |
|---|---|---|---|
| Adaptive rule | 3.95 | 2.14 | 10.58 |
| Ask the cloud always | 4.02 | 2.20 | 10.65 |
| Confidence rule | 3.00 | 1.97 | 5.19 |
| Confidence rule + reserve | 3.66 | 2.13 | 6.93 |
| Q-learning (frozen) | 3.66 | 2.13 | 6.93 |
| Q-learning (online) | 3.65 | 2.13 | 6.97 |
| Q-learning (online + spot checks) | 3.11 | 2.09 | 5.10 |
| Q-learning (online + 1 box in 5 checked) | 3.19 | 2.06 | 5.32 |
| Best possible (knows the answers) | 1.58 | 0.79 | 1.47 |

* "Always ask the cloud" becomes the most expensive real policy.
* The agent trained on clean photos (frozen, or learning online without spot checks) also asks too often.
* **With spot checks it corrects itself** and ends level with the confidence rule, which was already safe
  because it only asks for unsure boxes. Checking 1 box in 5 is nearly as good as checking every box.
* With the 15-a-minute limit nobody over-asks, so there spot checks change little
  (blurred: 3.26 frozen, 3.34 with spot checks, 3.25 confidence rule).
* Limits: the darker and blurred photos are made by software, not taken on a belt. Spot checks need somebody or
  something that knows the true state of some boxes.

### 34. "Where does the public benchmark fit?"
The same kind of table-based learner on 18 public edge / hub / cloud task graphs (Zenodo,
doi:10.5281/zenodo.10654551) against a greedy scheduler. Time relative to greedy (1.00 = same): 1.39 on
small graphs, 0.93 on the size it trained on, 0.99 on large graphs it never saw. Level with greedy, not
better. It is a side result.

---

## G. What is real and what is not

### 35. "Which of your numbers are real?"
| Thing | Status |
|---|---|
| Accuracy of the three models, the confidence levels, what a cloud call is worth | real photos, held-out, public data set (not our own boxes) |
| Laptop model time | measured on this laptop |
| ESP32 time | estimate, board not connected |
| ESP32 firmware | compiled for a PC and tested there against the laptop code; not yet run on a real board |
| Cloud time | measured over the Internet to Render: 83 ms per photo (model 5.2 ms); not yet with the real belt |
| Every policy comparison | simulated belt, network and timing; real photos and real model verdicts |
| Darker / blurred photos (E8) | the public test photos changed by software; not taken on a belt |
| Wiring guide and circuit diagram | checked against the firmware pin list; not built |
| Real belt | not built |

### 36. "Is your simulator just telling you what you want?"
Four safeguards. Every simulated box is a real photo, so the models' mistakes are the real ones. The agent
and the rules are trained and tuned on the validation photos and judged on the test photos. Every policy is
judged on the same sequence of boxes. And where the agent does not win, the tables say so (the questions on
no call limit, the simple rule, the ESP32, fewer cloud calls, and which gaps are real).

### 37. "Is this how a real factory would do it?"
No. A factory puts a strong computer next to the line and rarely needs the cloud. Our belt is a test-bed. The
method matters where a cheap local check exists and a better remote one is limited: many cameras sharing one
uplink, pay-per-call vision services, battery devices.

### 38. "What runs in the cloud, and what if it goes down?"
One Render web service in Singapore: the big model, the dashboard, the box log. If it is down (or asleep: the
free plan sleeps after 15 minutes without requests), cloud answers stop arriving, the laptop's verdicts are used,
and the agent stops spending calls. Inspection never depends on the cloud. The same server also runs on AWS EC2
with the included scripts; Render was chosen because it is free and gives https without setup.

### 38b. "How does the dashboard stay live, and does the ESP32 use Wi-Fi?"
The dashboard and the gateway use long polling: each waits on the server, which answers the moment a box
finishes, a sensor changes or a button is pressed. So an update needs one trip over the Internet (about 0.1 s),
not a timer. The belt, sensors, arm and tiny model stay on the USB cable (steady, about 1 ms). The ESP32's
Wi-Fi (firmware 4, on its second core) is extra: it sends the board's own readings to the cloud, has a setup
hotspot when no Wi-Fi works, and a live page on the local network.

---

## H. Short answers to keep ready

* **Complexity per box:** one small model run, a binary search per feature, one table lookup, one update.
* **Exploration risk:** only among options within 3 cost units of the best, 2 % of the time.
* **After a fault clears:** values of options not taken drift back toward their trained value, and the test
  photo keeps measuring the cloud, so the agent starts asking again.
* **Security:** API key on every request, stored encrypted; port open only to our address; plain HTTP, so a
  production system would add TLS.
* **Energy:** the reward has an energy term (ESP32 0.5 W, laptop 8 W while computing). Assumed figures.
* **Spot checks, in one line:** when the truth about a box is known, compare both verdicts and keep a running
  average of what the cloud call saved; use that instead of the number from the training photos.
* **Why the cloud loses more on blurred photos:** it looks at a bigger picture and relies on fine detail; a
  smear removes exactly that. Training with smeared and darker photos is the fix (switched on for the next retrain;
  not measured yet).
* **Next steps:** run it on the real belt; own-belt photos; retrain with harder photos; a camera on the edge
  board; several belts sharing one budget.
