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
| Edge | ESP32 on the belt | two IR sensors, belt motor, reject arm, LEDs, safety stop, **and** a last-resort damage check | tiny CNN, 11,705 numbers, 48 x 48 grey picture |
| Gateway | laptop | USB webcam, talks to both sides, runs the Q-learning agent, **checks every box first** | MobileNetV3-Small, 128 px |
| Cloud | AWS EC2 (Mumbai) | **second opinion** on the boxes it is asked about, dashboard, backups | MobileNetV3-Large, 160 px |

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
| Gateway (laptop) | 89.9 % | 96.0 % | 18.8 % | 2.5 ms (measured on this laptop) |
| Cloud | 95.1 % | 98.0 % | 9.0 % | 8.1 ms (server on the same PC, so NO Internet in this number) |

### 7. "The ESP32 is slower AND less accurate than the laptop. That is not a trade-off."
Correct, and we say so. The only camera is a USB webcam on the laptop, so the photo has to travel to the ESP32
over a cable first, and the chip is small. No model can fix that. The real trade-off in our numbers is
**laptop against cloud**. The ESP32's value is different: it does not depend on the laptop's processor or on the
Internet. When the laptop is overloaded and the network is bad together, it is the only place that still
answers in time (question 22).

### 8. "65.3 % correct is close to guessing. Is the ESP32 model any use?"
It is weak: 48 x 48 grey pixels do not show small damage. Even on its most confident boxes it is wrong about one time
in five, so it cannot be used as a first filter. Because a shipped damaged box costs ten times a wasted good
one, its cut-off is set very low: it catches 98.0 % of damaged boxes but throws away 81.2 % of good ones.
That is still cheaper than no check at all: about 6.3 per box against 35 for a box that passes unchecked.
So it is a last resort and nothing more.

### 9. "Are the ESP32 time and the cloud time real?"
* ESP32: no. No board was connected. The time is worked out from the cable speed (460,800 baud) and 40 ns per multiply. The accuracy is exact: the PC does the same arithmetic as the firmware and a test checks they agree. `MEASURE_SPEED_VS_ACCURACY.bat` with the board plugged in replaces the estimate.
* Cloud: no. The server ran on the same laptop, so there is no Internet in that number. It must be measured again after the AWS deployment.

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
Then the simple policy "always ask the cloud too" is about as good as anything, because the laptop's verdict
is always there as a fallback. In E7 with no limit on a good network it costs 1.29 per box and Q-learning
1.42 (sim). **Learning only earns its place when calls are scarce or the timing is tricky.**

---

## D. How the nodes talk

### 14. "How does the laptop talk to the ESP32?"
One USB cable, serial at 460,800 baud, short text lines. The laptop sends: keep-alive with belt speed, push
the arm, LEDs, settings, the model file, a photo. The ESP32 sends: "box under the camera", "box at the arm",
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

---

## E. The Q-learning part

### 18. "What are the state, the actions and the reward?"
* **Actions (3):** keep the laptop's verdict, ask the cloud too, ESP32 (only when the laptop is too slow to go first).
* **State (8,640 situations):** spare time if sent to the cloud (8 levels), measured cloud answer time (3),
  ping (3), can the laptop finish in time (3), cloud queue (2), **how sure the laptop's model is about this
  box (5)**, **cloud calls left right now (4)**.
* **Reward:** minus the expected cost of the box, for the model that answered and for that confidence level.
  A shipped damaged box costs 100, a wasted good box 10, a box that passes unchecked the expected damage
  plus 5. A cloud call is worth about 2.8 on a "very unsure" box, 1.2 on an "unsure" one and about -0.3 on a "sure" one.

### 19. "Where is the sequential decision? Each box looks independent."
The cloud-call budget. A call spent on this box is missing for a harder box a few seconds later. That is why
the next state matters (discount 0.9) and why this is Q-learning and not a one-shot choice. Without a budget
it would be nearly a one-shot problem, and we say so.

### 20. "What did it learn?"
This is the trained table for a good network (rows: how sure the laptop's model is; columns: cloud calls left):

| Laptop's model is | none left | 1 left | 2 left | 3 or more left |
|---|---|---|---|---|
| very unsure | keep | **ask** | **ask** | **ask** |
| unsure | keep | keep | **ask** | **ask** |
| fairly sure | keep | keep | keep | **ask** |
| sure | keep | keep | keep | keep |

Nobody wrote that pattern. It comes out of the rewards: the scarcer the calls, the more unsure the laptop has
to be before a call is spent.

### 21. "The laptop does not know if a box was really damaged. How can it learn online?"
It does not need to. The reward only uses what is known at run time: which model answered, whether it was
in time, and how sure the laptop's model was. Each model's quality per confidence level is measured once on
the validation photos and stored in the model card.

### 22. "How was it trained?"
First in a simulator: 16,000 episodes of 150 boxes (2.4 million boxes, about two minutes). Every simulated box
is a real validation photo with the models' real verdicts; the network, belt speed, laptop load, cloud-call
limit and hidden faults are random. It visited 2,944 of the 8,640 situations (many cannot happen). Then it
keeps learning on the line, carefully: 2 % exploration, and only among options whose values are close.

### 23. "Why a table and not a neural network (DQN, PPO)?"
Seven measured features with a few levels each fit in a table. A table trains in two minutes, decides in
microseconds, and can be read: the dashboard shows what it would do in each situation.

### 24. "Why can a simple rule not do this?"
A good rule nearly can, and we tested that. The **confidence rule** ("ask the cloud when the laptop is at least
unsure and the measured cloud time fits") uses the same confidence signal as the agent.
* Over 10 situations it costs 2.83 per box, Q-learning 2.63 (sim). Q-learning is ahead by 0.21.
* Against the best hand-written policy overall (Confidence rule): clearly ahead in 5 situations (busy, fast+busy, gateway-busy, server-slow, lossy), level in 3, behind in 2 (good, gateway-overload).
* On pure budget questions (E7) the two are level.
* What Q-learning adds: nobody has to tune its confidence level, time margin and reserve (the rule's were found
  by search), and it keeps adjusting on the line.

---

## F. Results

### 25. "What are your main results?"
All simulated, 10 situations, 15 cloud calls a minute, every box a real test photo, cost = what really
happened to the box (lower is better):

| Method | Cost per box | Boxes missed | Boxes sent to the cloud | Correct verdicts |
|---|---|---|---|---|
| Q-learning frozen | 2.57 | 0.0 % | 19 % | 83.6 % |
| Q-learning learning online | 2.63 | 0.0 % | 19 % | 83.5 % |
| Confidence rule | 2.83 | 0.0 % | 8 % | 82.4 % |
| Confidence rule + reserve | 2.91 | 0.0 % | 10 % | 82.4 % |
| Ask the cloud always | 2.95 | 0.0 % | 24 % | 82.2 % |
| Adaptive rule | 3.25 | 0.8 % | 10 % | 81.2 % |
| Always-Gateway | 5.46 | 7.3 % | 0 % | 79.2 % |
| Always-Edge | 6.32 | 0.0 % | 0 % | 41.8 % |
| Threshold rule | 6.74 | 11.6 % | 9 % | 75.9 % |
| Deadline rule | 6.79 | 11.8 % | 10 % | 75.8 % |
| Always-Cloud | 8.04 | 16.3 % | 26 % | 72.6 % |

* Q-learning against the best rule that picks **one place per box** (adaptive rule): 3.25 down to 2.63, 19 % lower.
* Q-learning against the best hand-written policy of any kind (Confidence rule): ahead by 0.21, 7 % lower.
* Missed boxes: 0.8 % with the adaptive rule, 0.0 % with any second-opinion policy.

### 26. "What does the fallback buy?"
Wi-Fi that stalls now and then, a fault never seen in training: the adaptive pick-one rule missed
4.5 % of boxes, the second-opinion policies 0.0 % (sim). A late cloud answer simply does not matter any more.

### 27. "When is the ESP32 actually chosen?"
Only when the laptop is overloaded. In that test (bad network too) the agent sent 99 % of boxes to the
ESP32 and missed 0.5 %; the threshold and deadline rules, which cannot use it, missed 73 % (sim).
The adaptive rule also switches to the ESP32 (0.3 % missed). This case needs the third node, not Q-learning.

### 28. "What happens with fewer cloud calls?"
Good network, about 60 boxes a minute, cost per box (sim, E7):

| Policy | no limit | 30 a minute | 15 a minute | 8 a minute | 4 a minute |
|---|---|---|---|---|---|
| Adaptive rule | 1.23 | 1.78 | 2.14 | 2.24 | 2.33 |
| Ask the cloud always | 1.29 | 1.88 | 2.04 | 2.23 | 2.30 |
| Confidence rule | 1.45 | 1.45 | 1.57 | 1.91 | 2.08 |
| Confidence rule + reserve | 1.40 | 1.45 | 1.70 | 1.87 | 2.14 |
| Q-learning (frozen) | 1.40 | 1.40 | 1.56 | 1.98 | 2.16 |
| Q-learning (online) | 1.42 | 1.44 | 1.63 | 2.00 | 2.11 |
| Best possible (knows the answers) | 0.52 | 0.52 | 0.52 | 0.58 | 1.14 |

"Best possible" knows which boxes a cloud answer would fix. No real policy can know that; it shows the room
that is left. Between the confidence rules and Q-learning the winner changes from column to column: level.

### 29. "30 % damaged boxes is unrealistic. Other costs?"
Tested (`results/E6_SENSITIVITY.md`, sim). For each setting every model got a new cut-off, the agent was
re-trained and the rules were re-tuned.

| Setting | Q-learning online | Best rule |
|---|---|---|
| 1% damaged | 0.57 | Confidence rule: 0.61 |
| 5% damaged | 1.33 | Confidence rule: 1.31 |
| 30% damaged (base) | 2.58 | Confidence rule + reserve: 2.72 |
| base, wasted good box costs 3x | 4.85 | Confidence rule + reserve: 4.89 |
| base, escaped defect costs 3x | 4.13 | Confidence rule + reserve: 4.41 |

Q-learning was ahead in 4 of 5 settings, level in 1, behind in 0.

### 30. "What if the simulator is wrong about the real hardware?"
Tested with hardware made 2 to 3 times slower than the simulator assumed (E5, sim): adaptive rule
3.23, Q-learning online 2.74. Nothing was re-tuned. Both use live measurements, so neither breaks;
the gap is mostly the second-opinion fallback again.

### 31. "Where does the public benchmark fit?"
The same kind of table-based learner on 18 public edge / hub / cloud task graphs (Zenodo,
doi:10.5281/zenodo.10654551) against a greedy scheduler. Time relative to greedy (1.00 = same): 1.39 on
small graphs, 0.93 on the size it trained on, 0.99 on large graphs it never saw. Level with greedy, not
better. It is a side result.

---

## G. What is real and what is not

### 32. "Which of your numbers are real?"
| Thing | Status |
|---|---|
| Accuracy of the three models, the confidence levels, what a cloud call is worth | real photos, held-out, public data set (not our own boxes) |
| Laptop model time | measured on this laptop |
| ESP32 time | estimate, board not connected |
| ESP32 firmware | compiled for a PC and tested there against the laptop code; not yet run on a real board |
| Cloud time | server on the same PC; AWS not deployed yet |
| Every policy comparison | simulated belt, network and timing; real photos and real model verdicts |
| Real belt | not built |

### 33. "Is your simulator just telling you what you want?"
Four safeguards. Every simulated box is a real photo, so the models' mistakes are the real ones. The agent
and the rules are trained and tuned on the validation photos and judged on the test photos. Every policy is
judged on the same sequence of boxes. And where the agent does not win, the tables say so (questions 13, 24,
27, 28).

### 34. "Is this how a real factory would do it?"
No. A factory puts a strong computer next to the line and rarely needs the cloud. Our belt is a test-bed. The
method matters where a cheap local check exists and a better remote one is limited: many cameras sharing one
uplink, pay-per-call vision services, battery devices.

### 35. "What runs on AWS, and what if it goes down?"
One EC2 instance in Mumbai: the big model, the dashboard, backups. If it is down, cloud answers stop arriving,
the laptop's verdicts are used, and the agent stops spending calls. Inspection never depends on the cloud.

---

## H. Short answers to keep ready

* **Complexity per box:** one small model run, a binary search per feature, one table lookup, one update.
* **Exploration risk:** only among options within 3 cost units of the best, 2 % of the time.
* **After a fault clears:** values of options not taken drift back toward their trained value, and the test
  photo keeps measuring the cloud, so the agent starts asking again.
* **Security:** API key on every request, stored encrypted; port open only to our address; plain HTTP, so a
  production system would add TLS.
* **Energy:** the reward has an energy term (ESP32 0.5 W, laptop 8 W while computing). Assumed figures.
* **Next steps:** run it on the real belt; own-belt photos; a camera on the edge board; several belts sharing
  one budget.
