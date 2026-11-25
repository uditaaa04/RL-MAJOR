# ESP32 edge node: wiring and flashing

The ESP32 runs the belt (motor, two IR sensors, reject arm, LEDs) and the tiny edge model.
It talks to the gateway laptop over the **USB cable**. No Wi-Fi setup, no extra Arduino libraries.

## 1. Wiring

| Part | Part pin | ESP32 pin | Note |
|---|---|---|---|
| IR sensor 1 (under the camera) | OUT | **GPIO 32** | |
| | VCC | **3V3** | **Not 5V.** 5V on the OUT pin can damage the ESP32. |
| | GND | GND | |
| IR sensor 2 (at the reject arm) | OUT | **GPIO 33** | |
| | VCC | **3V3** | Not 5V. |
| | GND | GND | |
| Reject arm servo (SG90) | signal (orange/yellow) | **GPIO 18** | |
| | + (red) | 5V from a separate supply | Do not feed the servo from the ESP32 3V3 pin. |
| | - (brown/black) | GND | |
| Motor driver L298N | ENA | **GPIO 25** | Pull the small ENA jumper off first. |
| | IN1 | **GPIO 26** | |
| | IN2 | **GPIO 27** | |
| | OUT1, OUT2 | belt motor | Swap the two wires if the belt runs backwards. |
| | +12V, GND | motor supply | 6 to 12 V, whatever the motor needs. |
| Green LED (optional) | + through 220 ohm | GPIO 16 | Lights for a good box. |
| Red LED (optional) | + through 220 ohm | GPIO 17 | Lights for a damaged box. |
| ESP32 | USB | gateway laptop | Power and data. Use a data cable, not a charge-only cable. |

**Join all grounds together**: ESP32 GND, L298N GND, servo supply GND, IR sensor GND. Without a common
ground nothing works reliably.

The blue LED on the board: steady = gateway connected, blinking = waiting for the gateway.

Different pins? Change the `#define PIN_...` lines at the top of `esp32_node/esp32_node.ino` and upload again.

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
