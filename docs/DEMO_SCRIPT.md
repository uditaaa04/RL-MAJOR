# 5-minute live demo (viva / review)

**Start**
- No hardware: double-click `START_DEMO.bat`. If the ESP32 is plugged in by USB, its tiny model is really used.
- Real belt: double-click `START_REAL_LINE.bat` (ESP32 on USB, webcam on the laptop).
- Real hardware, no belt yet: double-click `MANUAL_TEST.bat` and move the boxes by hand (section below).
- The dashboard is on the Internet: https://smart-conveyor-cloud.onrender.com. Open it 2 minutes before you start
  (the free server sleeps after 15 minutes without requests).

The dashboard opens by itself. Press **Full screen**. Pick **Light theme** for a projector, **Dark theme** for a screen.

## Fastest way: one button

Press **Run the guided demo**. The dashboard cuts the cloud calls, breaks the Internet, chokes the gateway and
overloads the cloud server in eight steps and shows a caption for each. Read the captions aloud and point at
the belt. When it ends it prints how many boxes were missed.

## By hand (5 minutes)

| Time | Do | Say |
|---|---|---|
| 0:00 | Belt running, Wi-Fi **Good**, cloud calls **15** | "Three nodes. The ESP32 runs the belt. This laptop is the gateway: webcam, small model, and the agent. The cloud has the big model. The laptop checks every box first, in about 2 ms. Then the agent decides whether the cloud should check it too." |
| 0:30 | Point at the line under the belt | "It says what happened to the last box: what the laptop's model said, how sure it was, and whether the cloud was asked. Here the cloud changed the verdict." Point at **Cloud calls** in the header. "Cloud calls are limited: 15 a minute." |
| 0:55 | Cloud calls **4** | "Now only 4 a minute. Watch which boxes still go up: only the ones the laptop is least sure about." Point at **What the agent has learned**: "sure boxes never, very unsure boxes always, the ones in between only when calls are left. Nobody wrote that." Point at the small table under the map: "And this is the agent checking itself. Left: what a cloud answer was worth on the training photos. Right: what it has measured on this line, from boxes whose true state was known afterwards." Set it back to **15**. |
| 1:40 | Press **Bad** | "The Internet gets bad. A cloud answer would come too late, so the agent stops asking and keeps the laptop's verdicts. If it does ask and the answer is late, the laptop's verdict is used. No box is missed." |
| 2:15 | Press **Overload the gateway** | "Now the laptop is choked too. Its model would answer after the box has passed. The only checker left in time is the tiny model on the ESP32." Point at the white dot on the USB line and the dashed boxes. "Weaker eyes, but no box passes unchecked." |
| 2:50 | Press **Overload the gateway** again (off), then **Good** | "The gateway times its own model every 2 seconds, sees it is fast again, and takes the work back. Then the Internet recovers and it starts asking the cloud again. Nobody told it." |
| 3:15 | Press **Overload the cloud server** | "The network is fine, but the server is overloaded. A ping cannot see that. Cloud answers come late, the laptop's verdict is used, and the agent stops spending calls." Switch it off again. |
| 3:45 | **Who decides** -> *Always the cloud*, then **Overload the cloud server** | "This is the old way: one place per box. Boxes turn red: missed." Switch back to *Q-learning agent*, overload off. |
| 4:10 | Open the **Results** page, first two tables | "Same photos at all three places: accuracy and time. And the second table: the laptop and the cloud agree on nine boxes in ten. Sending only the least-sure quarter to the cloud gives almost the same accuracy as sending everything." |
| 4:40 | Scroll down to **Which gaps are real** and **When the photos get worse** | "Below are the simulated decision experiments, replaying real photos. This table says which gaps are real and which are within chance. And this one shows what happens when the photos get darker or blurred: asking the cloud for everything becomes the worst choice, and the agent corrects itself from spot checks." |

## Live hardware demo by hand (MANUAL_TEST, 2 minutes)

| Do | Say |
|---|---|
| Point at **The ESP32 on the belt, live** | "These are the board's own readings: both IR sensors, the belt, the arm, its USB link and its own Wi-Fi." |
| Pass a hand in front of IR sensor 1, then 2 | "The dots light up on the dashboard about a tenth of a second later: sensor -> ESP32 -> USB -> laptop -> Internet -> cloud -> this screen." Point at **Gateway to this screen**. |
| Hold a box under the camera past IR sensor 1, then move it past IR sensor 2 | "IR sensor 1 took the photo. The laptop checked it, the agent decided whether the cloud should check it too, and at IR sensor 2 the arm pushed it off because it is damaged." Point at the new box in **Latest boxes**. |
| Press **Push the arm**, then **Check a box now** | "Buttons in the cloud reach the belt in one network trip: no timer. Check a box now takes a photo and decides, as if the sensor had fired." |
| Unplug nothing, just say it | "The belt never waits for the Wi-Fi: sensors, arm and the tiny model are on the USB cable. The ESP32's Wi-Fi only reports, and it has its own setup hotspot if the Wi-Fi is missing." |

## The honest sentences to have ready

- "The ESP32 is slower and weaker than the laptop, because the camera is on the laptop and the photo has to be
  sent to it over USB. Its job is the case where the laptop and the Internet both fail."
- "The ESP32 model is right about two times out of three. A 48 pixel grey photo does not show more."
- "If a row in the first table says emulated or 'on this PC', that number is not final. We say so on the page."
- "The decision experiments are simulated, but every simulated box is a real photo with the real models'
  verdicts. Against a hand-written confidence rule Q-learning is level: we repeated the comparison on 200
  re-drawn test sets and the gap is within chance. Against every rule that picks one place per box the gap is real."
- "The cloud-call limit is ours to set. With no limit and clean photos, 'always ask the cloud too' is as good as
  anything. On darker or blurred photos it is the worst choice, because there the cloud's model loses more
  than the laptop's."
- "The darker and blurred photos are made by software from the public photos. They are not from our belt."
- "Spot checks need somebody or something that knows the true state of some boxes. In the demo that is the
  pretend belt; on the real belt it is the box-order file; in a factory it would be a person checking a few boxes."

More questions: `docs/VIVA_QA.md` and the **How it works** page.
