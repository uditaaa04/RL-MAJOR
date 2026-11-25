# Start here (no typing needed)

Three nodes:

| Node | What it is | What it does |
|---|---|---|
| **Edge** | ESP32 on the belt | Runs the belt: motor, two IR sensors, reject arm, LEDs. Also has a tiny model for a quick look at a box. |
| **Gateway** | This laptop | Webcam, talks to the ESP32 (USB) and to the cloud (Internet). Its Q-learning agent picks, for every box, where the photo is checked. Has the small model. |
| **Cloud** | AWS server | The big, most accurate model. Also serves the dashboard. |

Double-click these files in this folder.

| File | What it does |
|---|---|
| `START_DEMO.bat` | The whole product with **no hardware**: pretend belt, this PC is also the cloud. Opens the dashboard. If an ESP32 is plugged in, its tiny model is really used. First run builds the Python environment (2-5 min, needs Internet). |
| `CHECK_ESP32.bat` | Tests the ESP32 alone: finds it, loads the tiny model, times it, swings the arm, lights the LEDs, shows the two sensors. Run this after wiring. |
| `START_REAL_LINE.bat` | **The real belt**: ESP32 on USB + webcam. Cloud is the AWS server if you have set it up, otherwise this PC. |
| `MEASURE_SPEED_VS_ACCURACY.bat` | The experiment of this project: the same photos checked at ESP32, gateway and cloud. Writes `results/TRADEOFF.md` and the first table on the dashboard's Results page. |
| `START_DEMO_ON_AWS.bat` | Pretend belt against the real AWS server. Asks for the server address and key once; the real line and the measurement use them from then on. |
| `TRAIN_ON_DOWNLOADED_PHOTOS.bat` | Public photo set (BoxCondition ZIP in Downloads): trains all three models, the agent, every result. |
| `RETRAIN_ON_REAL_PHOTOS.bat` | The same with your own photos in `data/real`. Old models are kept. |
| `RUN_PUBLIC_BENCHMARK.bat` | Second, public test (task graphs from Zenodo). |
| `CHECK_EVERYTHING.bat` | Runs the automatic tests. |

Stop anything: press Enter in its black window.

## Order for the real hardware

1. Wire and flash the ESP32: `docs/ESP32_WIRING.md` (pin table, flashing, 2-minute test in the Serial Monitor).
2. Close the Arduino Serial Monitor. Double-click `CHECK_ESP32.bat`. Fix anything it reports.
3. Mount the webcam above the belt, plug it into this laptop.
4. Double-click `START_REAL_LINE.bat`. Put boxes on the belt.
5. Stop it, then double-click `MEASURE_SPEED_VS_ACCURACY.bat` for the real ESP32 timing.

Settings you may need (file `config/system.yaml`):
- `esp32.servo_rest_deg`, `esp32.servo_push_deg`: arm angles.
- `esp32.ir_active_low`: set `false` if a sensor fires when the box leaves instead of when it arrives.
- `camera.roi`: the square of the webcam picture where the box is.
- `conveyor.cam_to_arm_m` and `conveyor.speed_mps_at_full`: measure them, they set the deadline.

## In the dashboard

- Press **Run the guided demo**. It cuts the cloud calls, breaks the Internet, chokes the gateway, overloads the cloud server, step by step, and explains each step.
- Or press the Wi-Fi buttons, **Cloud calls a minute**, **Overload the gateway**, **Overload the cloud server** yourself.
- The line under the belt says in plain words what happened to the last box: what the laptop's model said, how sure it was, and whether the cloud was asked for a second opinion.

Pages:
- **Live line**: the belt, the three nodes, the decisions, the fault buttons.
- **Results**: speed against accuracy at the three places, what a cloud second opinion is worth on real photos, then the simulated decision experiments, including where Q-learning only ties.
- **How it works**: the whole idea in plain words (good before the viva).

## AWS (the cloud node)

1. AWS website, region Mumbai, open CloudShell (the `>_` icon).
2. Actions, Upload file: `dist/server_bundle.zip`.
3. Paste: `unzip -o -q server_bundle.zip 'cloud/*' -d sc && bash sc/cloud/deploy_cloudshell.sh server_bundle.zip`
4. It prints the server address and how to show the API key. Double-click `START_DEMO_ON_AWS.bat` and paste both.
5. From now on `START_REAL_LINE.bat` and `MEASURE_SPEED_VS_ACCURACY.bat` use AWS as the cloud.
6. When finished, stop paying: `bash sc/cloud/deploy_cloudshell.sh --teardown`

## If something fails

- Send the file `logs/go_<name>.log` (for example `go_demo.log`, `go_line.log`, `go_esp32.log`). It says exactly what went wrong.
- "Python is not installed": install Python 3.12 from python.org, tick "Add python.exe to PATH".
- "No ESP32 answered": firmware not flashed, Serial Monitor still open, charge-only USB cable, or missing CP210x / CH340 driver.
- Files from the older Raspberry Pi version are moved to `_old_2node_version/` automatically. Nothing is deleted.

## Honest limits

- The ESP32 cannot run the gateway's model (it has about 0.3 MB of free memory), so it has its own much smaller one. It is the least accurate of the three.
- The webcam is on the laptop, so the photo is sent to the ESP32 over USB first. That makes the ESP32 slower than the gateway in normal conditions. Its job is the case where the gateway is overloaded and the Internet is slow: then it is the only checker that still answers in time.
- Until `MEASURE_SPEED_VS_ACCURACY.bat` has run with the board plugged in, the ESP32's time on the dashboard is an estimate, and it says so.
- The decision experiments (which policy is best) are simulated. The real belt has not been measured.
