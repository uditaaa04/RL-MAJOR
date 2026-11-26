# Start here (no typing needed)

Three nodes:

| Node | What it is | What it does |
|---|---|---|
| **Edge** | ESP32 on the belt | Runs the belt: motor, two IR sensors, reject arm. Also has a tiny model for a quick look at a box. |
| **Gateway** | This laptop | Webcam, talks to the ESP32 (USB) and to the cloud (Internet). Its Q-learning agent picks, for every box, where the photo is checked. Has the small model. |
| **Cloud** | Render server (Singapore), https://smart-conveyor-cloud.onrender.com | The big, most accurate model. Also serves the dashboard. |

Double-click these files in this folder.

| File | What it does |
|---|---|
| `START_DEMO.bat` | The whole product with **no hardware**: pretend belt, this PC is also the cloud. Opens the dashboard. If an ESP32 is plugged in, its tiny model is really used. First run builds the Python environment (2-5 min, needs Internet). |
| `FIRST_HARDWARE_TEST.bat` | **The whole first hardware test in one window.** Installs what is missing, finds the ESP32, tests the board and the two sensors, checks the camera, runs the real line, then measures speed against accuracy on the real ESP32. It says what to do at each point. The ESP32 must be flashed first. |
| `CHECK_ESP32.bat` | Tests the ESP32 alone: finds it, loads the tiny model, times it, swings the arm, shows the two sensors. Run this after wiring. |
| `CHECK_CAMERA.bat` | Takes one picture from every camera on this laptop and opens a page with them. Shows which camera the real line uses and its crop square. Type the right camera number and square in the black window; they are saved. The pictures stay on this PC. |
| `FLASH_ESP32.bat` | Puts **firmware 4** on the ESP32 with the Arduino IDE that is already installed (board ESP32 Dev Module, partition Huge APP). Plug the board in first; close the Serial Monitor. Then it shows the board's Wi-Fi messages. |
| `MANUAL_TEST.bat` | **Hand test with live data**: real ESP32, sensors, arm and webcam, belt motor off. Move each box by hand past IR sensor 1, then IR sensor 2. Every box, and both sensors, show up live on the dashboard. The agent's learned table is not changed. |
| `START_REAL_LINE.bat` | **The real belt**: ESP32 on USB + webcam. Cloud is the Render server (set up already), otherwise this PC. |
| `MEASURE_SPEED_VS_ACCURACY.bat` | The experiment of this project: the same photos checked at ESP32, gateway and cloud. Writes `results/TRADEOFF.md` and the first table on the dashboard's Results page. |
| `START_DEMO_ON_AWS.bat` | Pretend belt against the real cloud server (Render now; the name is from the AWS version). The address and key are already saved on this laptop; the real line and the measurement use them too. |
| `TRAIN_ON_DOWNLOADED_PHOTOS.bat` | Public photo set (BoxCondition ZIP in Downloads): trains all three models, the agent, every result. |
| `RETRAIN_ON_REAL_PHOTOS.bat` | The same with your own photos in `data/real`. Old models are kept. |
| `RUN_PUBLIC_BENCHMARK.bat` | Second, public test (task graphs from Zenodo). |
| `CHECK_EVERYTHING.bat` | Runs the automatic tests. |

Stop anything: press Enter in its black window.

## Order for the real hardware

1. Wire the belt with `docs/hardware/SmartConveyor_Hardware_Setup_Guide.pdf` (29 numbered wires, build order,
   checks before power) and its circuit diagram. It was checked pin by pin against the ESP32 program.
   Then flash the ESP32: double-click `FLASH_ESP32.bat` (or by hand: `docs/ESP32_WIRING.md`, which also has the
   2-minute test in the Serial Monitor and the Wi-Fi setup).
2. Close the Arduino Serial Monitor. Double-click `CHECK_ESP32.bat`. Fix anything it reports.
3. Mount the webcam above the belt, plug it into this laptop. Double-click `CHECK_CAMERA.bat`: pick the camera
   that shows the belt and set the crop square so that a box fills it. Run it again after moving the camera or the lamp.
4. Double-click `START_REAL_LINE.bat`. Put boxes on the belt. (A demo still open in another window must be
   stopped first; the launcher says so.)
5. Stop it (Enter: the belt stops within two seconds and what the agent learned is saved), then double-click
   `MEASURE_SPEED_VS_ACCURACY.bat` for the real ESP32 timing.

No belt yet? Flash the ESP32, then double-click `MANUAL_TEST.bat` and move boxes by hand: live on the dashboard.
`FIRST_HARDWARE_TEST.bat` does steps 2 to 5 in one window with only the ESP32, the two sensors and a camera
(`docs/ESP32_WIRING.md`, section 4, describes the same test by hand).

## The ESP32's own Wi-Fi (firmware 4)

The USB cable stays the fast path for the belt, the sensors, the arm and the tiny model. On top of that the board:
1. joins the Wi-Fi written in `esp32_node/secrets.h` (this file stays on this PC; GitHub only has `secrets.example.h`);
2. cannot join -> opens its own hotspot (name and password in `secrets.h`). Join it with a phone: the setup page opens
   by itself, or go to **http://192.168.4.1**. Pick the Wi-Fi, type its password and the hotspot password, **Save**.
   Back online, the hotspot closes. If the Wi-Fi drops for 20 s, the hotspot comes back;
3. online: its own live page at `http://<HOSTNAME>.local` (or its address), and it sends its sensors, belt and arm to
   the cloud dashboard by itself. The cloud address and key are copied to it over USB the first time the gateway runs.
Only 2.4 GHz Wi-Fi works (iPhone hotspot: turn on "Maximise Compatibility").

Settings you may need (file `config/system.yaml`):
- `esp32.servo_rest_deg`, `esp32.servo_push_deg`: arm angles.
- `esp32.ir_active_low`: set `false` if a sensor fires when the box leaves instead of when it arrives.
- `camera.usb_device` and `camera.roi`: which camera, and the square of its picture where the box is. `CHECK_CAMERA.bat` sets both.
- `camera.capture_delay_ms`: wait this long after IR sensor 1 fires before the photo is taken, so the box is in the middle of the square (0 = at once).
- `conveyor.min_box_gap_s`: one box counted twice by a sensor? Raise it (keep it below the time between two boxes).
- `conveyor.cam_to_arm_m` and `conveyor.speed_mps_at_full`: measure them, they set the deadline.
- `conveyor.arm_sensor`: set `false` if you have only ONE IR sensor (under the camera). The arm is then fired by
  time, so the two numbers above must be measured well.

For accuracy numbers on your own boxes, write the name of a box-order file into `conveyor.box_order_file`
(`experiments/sequences/seq_A.csv` shows the format: one line per box, `good` or `damaged`, in the order you
put them on the belt; the list repeats when it ends). The dashboard
then shows correct / wrong, and the agent uses the same file for its **spot checks**: it measures on your belt
what a cloud answer is worth instead of trusting the number from the training photos.

## In the dashboard

- Press **Run the guided demo**. It cuts the cloud calls, breaks the Internet, chokes the gateway, overloads the cloud server, step by step, and explains each step.
- Or press the Wi-Fi buttons, **Cloud calls a minute**, **Overload the gateway**, **Overload the cloud server** yourself.
- The line under the belt says in plain words what happened to the last box: what the laptop's model said, how sure it was, and whether the cloud was asked for a second opinion.

Pages:
- **Live line**: the belt, the three nodes, the decisions, the fault buttons.
- **Results**: speed against accuracy at the three places, what a cloud second opinion is worth on real photos, then the simulated decision experiments: which gaps are real and which are within chance, and what happens when the photos get darker or blurred.
- **How it works**: the whole idea in plain words (good before the viva).

## The cloud node: Render (set up)

The server is live at https://smart-conveyor-cloud.onrender.com and redeploys itself from GitHub. This laptop already
has its address and key (`cloud/.secrets/aws_server.json`). Free plan: it sleeps after 15 minutes without requests and
needs about a minute to wake up; the launchers wait for it. Open the dashboard 2 minutes before a demo, and download
the box log after a run (a restart wipes it). Details: `docs/CLOUD_RENDER.md`.

## AWS (alternative cloud node, not in use)

1. AWS website, region Mumbai, open CloudShell (the `>_` icon).
2. Actions, Upload file: `dist/server_bundle.zip`.
3. Paste: `unzip -o -q server_bundle.zip 'cloud/*' -d sc && bash sc/cloud/deploy_cloudshell.sh server_bundle.zip`
4. It prints the server address and how to show the API key. Double-click `START_DEMO_ON_AWS.bat` and paste both.
5. Then `START_REAL_LINE.bat` and `MEASURE_SPEED_VS_ACCURACY.bat` use AWS instead of Render as the cloud.
6. When finished, stop paying: `bash sc/cloud/deploy_cloudshell.sh --teardown`

## If something fails

- Send the file `logs/go_<name>.log` (for example `go_demo.log`, `go_line.log`, `go_gateway.log`, `go_esp32.log`). It says exactly what went wrong.
- "another run is still open on this PC": a demo or the real line is running in another window. Press Enter there.
- The real line shows the wrong picture (your face, not the belt): double-click `CHECK_CAMERA.bat` and type the other camera's number.
- "Python is not installed": install Python 3.12 from python.org, tick "Add python.exe to PATH".
- "No ESP32 answered": firmware not flashed, Serial Monitor still open, charge-only USB cable, or missing CP210x / CH340 driver.
- The ESP32 does not join the Wi-Fi: 5 GHz-only network, wrong password in `secrets.h`, or out of range. Join its setup
  hotspot and pick the network there. The board's messages start with `#` (Serial Monitor at 460800 baud, or `FLASH_ESP32.bat`).
- The dashboard says the board's cloud link is not ok: "wrong API key" -> run the gateway once with the board on USB (it
  copies the key); "no answer" -> the Render server is asleep or the Wi-Fi has no Internet.
- Files from the older Raspberry Pi version are moved to `_old_2node_version/` automatically. Nothing is deleted.

## Honest limits

- The ESP32 cannot run the gateway's model (it has about 0.3 MB of free memory), so it has its own much smaller one. It is the least accurate of the three.
- The webcam is on the laptop, so the photo is sent to the ESP32 over USB first. That makes the ESP32 slower than the gateway in normal conditions. Its job is the case where the gateway is overloaded and the Internet is slow: then it is the only checker that still answers in time.
- Until `MEASURE_SPEED_VS_ACCURACY.bat` has run with the board plugged in, the ESP32's time on the dashboard is an estimate, and it says so.
- The decision experiments (which policy is best) are simulated. The real belt has not been measured.
- Q-learning and the confidence rule are level: over 200 re-drawn test sets the gap between them is within chance. The real gain is the second-opinion design itself, against every rule that picks one place per box.
- The models were trained on clean public photos. On darker or blurred photos the cloud's model loses more than the gateway's (`results/shift_scores.json`). Photos from your own webcam will differ again: retrain on them (`RETRAIN_ON_REAL_PHOTOS.bat`) before trusting any accuracy number on your boxes. The next retrain also mixes darker and smeared photos into the training on purpose; that has not been measured yet.
