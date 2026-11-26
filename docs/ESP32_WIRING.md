# ESP32 edge node: wiring and flashing

The ESP32 runs the belt (motor, two IR sensors, reject arm) and the tiny edge model.
It talks to the gateway laptop over the **USB cable** for everything that is time-critical. Firmware 4 also joins
Wi-Fi (section 6) to send its own readings to the cloud dashboard. No extra Arduino libraries are needed.

## 1. Wiring

**The full build guide is in `docs/hardware/`**: `SmartConveyor_Hardware_Setup_Guide.pdf` (parts, 29 numbered
wires, build order, checks before power) and `SmartConveyor_Circuit_Diagram.png` (light) / `_dark.png`.
It was checked pin by pin against the firmware. This table is the short version.

| Part | Part pin | Goes to | Note |
|---|---|---|---|
| IR sensor 1 (under the camera) | OUT | **GPIO 32** (D32) | |
| | VCC | **3V3** | **Not 5 V.** 5 V on OUT can damage the ESP32. |
| | GND | GND | |
| IR sensor 2 (at the reject arm) | OUT | **GPIO 33** (D33) | Optional, see below. |
| | VCC | **3V3** | Not 5 V. |
| | GND | GND | |
| Reject arm servo | signal (orange/yellow) | **GPIO 18** (D18) | |
| | + (red) | **5.0 V from the buck converter** | Never from the ESP32. Set the buck to 5.0 V with a multimeter BEFORE the servo is connected. |
| | - (brown/black) | GND | |
| Motor driver HW-039 (BTS7960) | RPWM | **GPIO 25** (D25) | Belt speed. The only motor wire on the ESP32. |
| | VCC, R_EN, L_EN | **3V3** | 3V3, not 5 V: on 5 V the driver would not see the ESP32's 3.3 V as HIGH. |
| | LPWM, GND | GND | Reverse input held off. |
| | R_IS, L_IS | leave empty | |
| | B+, B- | 12 V adapter +, - | Reversed + and - destroys the driver. |
| | M+, M- | belt motor | Swap the two wires if the belt runs backwards. |
| ESP32 | USB | gateway laptop | Power and data. Use a data cable, not a charge-only cable. |

**Join all grounds together**: ESP32 GND, 12 V adapter -, buck OUT-, servo, IR sensors, HW-039 GND.

**No good / damaged LEDs.** The build guide PDF and the diagram still show a green and a red LED (wires 26
and 27, GPIO 16 and 17) and a test step `L 1 / L 2 / L 0`. Skip all three: this build has no LEDs, the firmware
(version 2 and later) never touches GPIO 16 or 17 and has no `L` command. The dashboard shows every verdict.

GPIO 26 and 27 stay empty. The firmware still switches GPIO 26 on while the belt runs (that was the direction
pin of the older L298N driver); with nothing wired there it does no harm. An L298N also still works:
ENA to GPIO 25 (pull its jumper off), IN1 to GPIO 26, IN2 to GPIO 27.

If the belt jerks for a moment when 12 V is switched on while the ESP32 is starting, add a 10 kohm resistor
from GPIO 25 to GND. Most HW-039 boards already hold that input low by themselves.

The blue LED on the board: steady = gateway connected, blinking = waiting for the gateway.

Different pins? Change the `#define PIN_...` lines at the top of `esp32_node/esp32_node.ino` and upload again.

### Only one IR sensor?

Fit it under the camera (GPIO 32) and set `conveyor.arm_sensor: false` in `config/system.yaml`.
The gateway then fires the reject arm **by time**: camera-to-arm distance divided by belt speed after the box
was seen. Two numbers must then be right, because nothing else tells the gateway where the box is:

- `conveyor.cam_to_arm_m`: measure it with a ruler (camera centre to the arm).
- `conveyor.speed_mps_at_full`: time a box over a known distance at full speed.

With two sensors the arm fires exactly when the box arrives, so two sensors are still the better build.

## 2. Flash the board (once)

**One double-click: `FLASH_ESP32.bat`.** It uses the arduino-cli inside Arduino IDE 2 and the ESP32 boards already
installed there, picks the board and partition by itself, uploads, then shows what the board says (its Wi-Fi
messages start with `#`). Close the Serial Monitor and any gateway window first.

Before the first flash, put your Wi-Fi names and passwords in `esp32_node/secrets.h` (copy `secrets.example.h`).
That file stays on this PC: it is never uploaded to GitHub.

By hand in Arduino IDE 2:
1. **File > Open** and pick `esp32_node/esp32_node.ino`.
2. **Tools > Board > esp32 > ESP32 Dev Module**.
3. **Tools > Partition Scheme > Huge APP (3MB No OTA/1MB SPIFFS)**. Wi-Fi + https do not fit the default 1.2 MB.
4. Plug the ESP32 in. **Tools > Port**: pick the COM port that appeared.
   No new port? Install the USB driver for the chip next to the USB socket: CP2102 ("CP210x driver") or CH340 ("CH340 driver").
5. Press **Upload** (arrow button). If it stays on "Connecting....", hold the **BOOT** button on the board until it starts writing.

The model is not inside the firmware. The gateway sends it to the board every time it starts, so
re-training never needs a re-flash.

## 3. Test without the gateway (2 minutes)

Open **Tools > Serial Monitor**, set **460800 baud** and **Newline**. Type a letter and press Enter:

| Type | What should happen |
|---|---|
| `H` | `H SCQC-ESP32 4 0 ...` comes back (4 = firmware version). The board is alive. |
| `N` | `N ONLINE 192.168.x.x -55 0 ok Hotspot`: Wi-Fi state, address, signal, setup hotspot on/off, cloud reachable, Wi-Fi name. |
| `P` | The reject arm swings out and back. |
| `B 150` | The belt runs, then stops by itself after 3 seconds. That stop is the safety rule: no gateway, no belt. |
| (hand in front of IR sensor 1) | `E C ...` appears. |
| (hand in front of IR sensor 2) | `E A ...` appears. |
| `S` | `S 0 0 ...`; the first two numbers turn 1 while a sensor sees something. |

If a sensor reports backwards (event when you take the hand away), type `C ir_low 0`, or set
`esp32.ir_active_low: false` in `config/system.yaml` (the gateway sends it at start).

Arm angle wrong? `C servo_push 60` then `P` to try values; put the good ones in `config/system.yaml` under `esp32:`.

**Close the Serial Monitor before starting the gateway.** Only one program can use the port.

## 4. Bench test before the belt is built (ESP32 on USB, two IR sensors, webcam)

No 12 V adapter is needed for this. Without 12 V the belt motor and the servo do not move (the servo's 5 V comes
from the buck converter, which is fed by the 12 V adapter). **Do not power the servo from an ESP32 pin instead**:
the board resets.

**One double-click does all four steps below in order: `FIRST_HARDWARE_TEST.bat`.** By hand they are:

1. `CHECK_ESP32.bat`: the board answers, the tiny model loads, 30 photos are timed (the first number measured
   on the real board), both sensors show an event when a hand passes.
2. `CHECK_CAMERA.bat`: one picture from every camera on the laptop. Type the number of the camera that looks at
   the table, and the crop square.
3. `MANUAL_TEST.bat` (belt motor off): hold a box under the webcam so that it also passes IR sensor 1. The photo
   is taken at that moment. Move it past IR sensor 2 **within about 3 seconds** (`conveyor.manual_window_s`).
   (`START_REAL_LINE.bat` does the same with the belt running and less time.) The dashboard shows the photo, what the
   laptop's model said, whether the cloud was asked, and the verdict. Too slow? The box is still counted, with
   the note `arm-sensor-missed` in the log, and the arm is not asked to push. Press Enter in the black window to stop.
4. `MEASURE_SPEED_VS_ACCURACY.bat` (line stopped, board still plugged in): the real time the ESP32 needs per
   photo goes into the results table.

What this test can and cannot show: sensors, photo, decision, verdict and the USB link are real. The timing is
not: a hand is not a belt. And the models were trained on public photos, so their good / damaged verdicts on
your own boxes can be wrong until they are retrained on your photos.

## 5. Why USB for the belt, and Wi-Fi only on top

- The board is plugged into the laptop for power anyway.
- The belt keeps working in a room with bad Wi-Fi.
- The link time is steady (a 32 x 32 photo takes about 22 ms to send), so the edge timing is repeatable.
- Wi-Fi runs on the ESP32's other processor core. A slow Wi-Fi or a slow cloud can never delay a sensor event,
  the arm or the tiny model.

## 6. Wi-Fi (firmware 4)

Settings in `esp32_node/secrets.h`: main Wi-Fi, backup Wi-Fi (optional), setup hotspot name and password (8 or more
characters), setup page address (default 192.168.4.1) and the live page name (`HOSTNAME` -> `http://<HOSTNAME>.local`).

| What happens | What you see |
|---|---|
| Power on | It joins the Wi-Fi saved from the setup page, otherwise the main Wi-Fi in `secrets.h`, then the backup. |
| No Wi-Fi works within 15 s | Its own hotspot opens. Join it: the setup page opens by itself (or http://192.168.4.1). Pick the Wi-Fi (2.4 GHz only), its password, the hotspot password, **Save**. The page shows when it is online and its address. |
| Online | The hotspot closes (after 20 s, or 2 minutes if a phone is still on it). Live page: `http://<HOSTNAME>.local` or its address. Its readings go to the cloud dashboard every 2 s and at once when a sensor changes. |
| Wi-Fi lost | It keeps trying; after 20 s the setup hotspot opens again until the Wi-Fi is back. |
| "Forget the saved Wi-Fi" (setup page) | Back to the networks in `secrets.h`, hotspot opens. |

The cloud address and API key reach the board over USB: the gateway hands them over every time it connects (the
board writes its memory only when they change). The board checks the cloud's https certificate against the root
certificates in `esp32_node/root_ca.h` (Let's Encrypt and Google Trust Services, valid to 2035-2040). If that check
ever fails, the setup page has a "skip the certificate check (testbed only)" box.

Serial commands for Wi-Fi: `N` status, `V name<TAB>password` join a network, `X` forget it, `U <url>` cloud address,
`Y <key>` cloud key.
