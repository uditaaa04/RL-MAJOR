# ESP32 edge node: wiring and flashing

The ESP32 runs the belt (motor, two IR sensors, reject arm, LEDs) and the tiny edge model.
It talks to the gateway laptop over the **USB cable**. No Wi-Fi setup, no extra Arduino libraries.

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
| Green LED (optional) | + through 330 ohm | GPIO 16 (RX2) | Lights for a good box. 220 ohm to 1 kohm is fine. |
| Red LED (optional) | + through 330 ohm | GPIO 17 (TX2) | Lights for a damaged box. |
| ESP32 | USB | gateway laptop | Power and data. Use a data cable, not a charge-only cable. |

**Join all grounds together**: ESP32 GND, 12 V adapter -, buck OUT-, servo, IR sensors, HW-039 GND.

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

1. Arduino IDE 2: **File > Open** and pick `esp32_node/esp32_node.ino`.
2. **Tools > Board > esp32 > ESP32 Dev Module**.
3. Plug the ESP32 in. **Tools > Port**: pick the COM port that appeared.
   No new port? Install the USB driver for the chip next to the USB socket: CP2102 ("CP210x driver") or CH340 ("CH340 driver").
4. Press **Upload** (arrow button). If it stays on "Connecting....", hold the **BOOT** button on the board until it starts writing.

The model is not inside the firmware. The gateway sends it to the board every time it starts, so
re-training never needs a re-flash.

## 3. Test without the gateway (2 minutes)

Open **Tools > Serial Monitor**, set **460800 baud** and **Newline**. Type a letter and press Enter:

| Type | What should happen |
|---|---|
| `H` | `H SCQC-ESP32 1 0 ...` comes back. The board is alive. |
| `P` | The reject arm swings out and back. |
| `B 150` | The belt runs, then stops by itself after 3 seconds. That stop is the safety rule: no gateway, no belt. |
| `L 1` / `L 2` / `L 0` | Green LED / red LED / both off. |
| (hand in front of IR sensor 1) | `E C ...` appears. |
| (hand in front of IR sensor 2) | `E A ...` appears. |
| `S` | `S 0 0 ...`; the first two numbers turn 1 while a sensor sees something. |

If a sensor reports backwards (event when you take the hand away), type `C ir_low 0`, or set
`esp32.ir_active_low: false` in `config/system.yaml` (the gateway sends it at start).

Arm angle wrong? `C servo_push 60` then `P` to try values; put the good ones in `config/system.yaml` under `esp32:`.

**Close the Serial Monitor before starting the gateway.** Only one program can use the port.

## 4. Why USB and not Wi-Fi

- The board is plugged into the laptop for power anyway.
- Nothing to configure, and it works in a room with bad Wi-Fi.
- The link time is steady (a 32 x 32 photo takes about 22 ms to send), so the edge timing is repeatable.
