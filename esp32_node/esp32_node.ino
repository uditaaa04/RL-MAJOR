/*
  Smart Conveyor QC  -  EDGE NODE (ESP32)
  =====================================================================
  What this board does
    1. Runs the conveyor: belt motor, two IR sensors, reject arm (servo), two LEDs.
    2. Runs the TINY edge model: the gateway laptop sends a very small grey photo,
       the ESP32 answers "chance this box is damaged" a few milliseconds later.
    3. Safety: if the gateway laptop goes silent for 3 seconds, the belt stops.

  How it talks: USB cable to the gateway laptop (serial, 460800 baud).
  No Wi-Fi name or password is needed. No extra Arduino libraries are needed.

  Flash it
    Arduino IDE 2  ->  Tools > Board > "ESP32 Dev Module"  ->  pick the COM port  ->  Upload.
    Close the Serial Monitor before starting the gateway program (only one program can use the port).

  Text commands (one per line; you can type them in the Serial Monitor at 460800 baud to test)
    H                hello                 -> H SCQC-ESP32 <version> <model loaded 0/1> <free memory>
    K <duty>         keep-alive + belt speed 0..255 (sent by the gateway every second)
    B <duty>         belt speed 0..255
    P [hold_ms]      push the reject arm out and back
    L <0|1|2>        LEDs: 0 off, 1 good (green), 2 damaged (red)
    S                status                -> S <ir_camera> <ir_arm> <duty> <free memory> <uptime ms>
    C <name> <value> setting: servo_rest, servo_push (degrees), ir_low (1 = sensor gives LOW on a box),
                     debounce_ms
    W <bytes> <sum>  + raw bytes: load the tiny model           -> W OK <layers> <input size>
    I <id> <bytes> <sum> + raw bytes: one grey photo            -> R <id> <chance x10000> <compute us> <receive ms>
  Events the board sends by itself
    BOOT SCQC-ESP32 <version>     after power-up / reset
    E C <ms>                      IR sensor 1: a box is under the camera
    E A <ms>                      IR sensor 2: a box is at the reject arm
*/
#ifdef HOST_TEST
#include "host_stub.h"            // lets the same file be compiled and tested on a PC
#else
#include <Arduino.h>
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------- pins: change here if you wire it differently ----------------
#define PIN_IR_CAMERA   32      // IR sensor 1 OUT  (box under the camera)
#define PIN_IR_ARM      33      // IR sensor 2 OUT  (box at the reject arm)
#define PIN_SERVO       18      // reject arm servo signal (orange / yellow wire)
#define PIN_MOTOR_ENA   25      // L298N ENA  (belt speed, PWM)  - remove the ENA jumper on the L298N
#define PIN_MOTOR_IN1   26      // L298N IN1
#define PIN_MOTOR_IN2   27      // L298N IN2
#define PIN_LED_GOOD    16      // optional green LED (+ 220 ohm resistor)
#define PIN_LED_BAD     17      // optional red LED   (+ 220 ohm resistor)
#define PIN_LED_LINK     2      // blue LED on the board: ON = gateway connected

#define BAUD            460800
#define FW_VERSION      1
#define LINK_TIMEOUT_MS 3000    // no message from the gateway for this long -> stop the belt

// ---------------- settings the gateway can change (command C) ----------------
static int  g_servo_rest = 0;       // degrees
static int  g_servo_push = 80;      // degrees
static bool g_ir_low = true;        // most IR modules give LOW when they see a box
static uint32_t g_debounce_ms = 20;

// ---------------- data types (kept above every function on purpose) ----------------
struct IrSensor {
  uint8_t pin;
  char tag;
  bool busy;                // a box is in front of the sensor right now
  uint32_t clear_since;     // when the sensor last went clear (0 = not clear)
};
struct ConvLayer {
  uint8_t cin, cout, stride;
  const float *w;                          // [cout][cin][3][3]
  const float *b;                          // [cout]
};

// ---------------- PWM helpers (work with ESP32 Arduino core 2.x and 3.x) ----------------
#define SERVO_HZ   50
#define SERVO_BITS 16
#define MOTOR_HZ   5000
#define MOTOR_BITS 8
#define CH_SERVO   0
#define CH_MOTOR   2

static void pwmBegin() {
#if defined(HOST_TEST)
#elif defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttachChannel(PIN_SERVO, SERVO_HZ, SERVO_BITS, CH_SERVO);      // fixed channels: the servo and
  ledcAttachChannel(PIN_MOTOR_ENA, MOTOR_HZ, MOTOR_BITS, CH_MOTOR);  // the motor must not share a timer
#else
  ledcSetup(CH_SERVO, SERVO_HZ, SERVO_BITS);
  ledcAttachPin(PIN_SERVO, CH_SERVO);
  ledcSetup(CH_MOTOR, MOTOR_HZ, MOTOR_BITS);
  ledcAttachPin(PIN_MOTOR_ENA, CH_MOTOR);
#endif
}

static void pwmWrite(int pin, int channel, uint32_t duty) {
#if defined(HOST_TEST)
  hostPwm(pin, duty);
  (void)channel;
#elif defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(pin, duty);
  (void)channel;
#else
  ledcWrite(channel, duty);
  (void)pin;
#endif
}

static void servoWrite(int deg) {
  if (deg < 0) deg = 0;
  if (deg > 180) deg = 180;
  uint32_t us = 500 + (uint32_t)deg * 2000 / 180;              // 0.5 ms .. 2.5 ms pulse
  pwmWrite(PIN_SERVO, CH_SERVO, us * 65535UL / 20000UL);       // 20 ms period
}

// ---------------- belt ----------------
static int g_duty = 0;             // what the gateway asked for (0..255)
static bool g_link_ok = false;
static uint32_t g_last_rx = 0;

static void motorApply(int duty) {
  if (duty < 0) duty = 0;
  if (duty > 255) duty = 255;
  digitalWrite(PIN_MOTOR_IN1, duty > 0 ? HIGH : LOW);
  digitalWrite(PIN_MOTOR_IN2, LOW);
  pwmWrite(PIN_MOTOR_ENA, CH_MOTOR, (uint32_t)duty);
}

// ---------------- reject arm (non-blocking) ----------------
static bool g_arm_out = false;
static uint32_t g_arm_back_at = 0;

static void armPush(uint32_t hold_ms) {
  servoWrite(g_servo_push);
  g_arm_out = true;
  g_arm_back_at = millis() + hold_ms;      // a second push while it is out simply keeps it out longer
}

static void armService() {
  if (g_arm_out && (int32_t)(millis() - g_arm_back_at) >= 0) {
    servoWrite(g_servo_rest);
    g_arm_out = false;
  }
}

// ---------------- IR sensors ----------------
static IrSensor g_ir[2] = { { PIN_IR_CAMERA, 'C', false, 0 }, { PIN_IR_ARM, 'A', false, 0 } };

static inline bool irSees(const IrSensor &s) {
  return (digitalRead(s.pin) == LOW) == g_ir_low;
}

// Called very often, also while a photo is being received or the model is running,
// so a box is reported within a few milliseconds whatever the board is doing.
static void pollIR() {
  uint32_t now = millis();
  for (int i = 0; i < 2; i++) {
    IrSensor &s = g_ir[i];
    bool sees = irSees(s);
    if (!s.busy) {
      if (sees) {                           // leading edge: report at once (no waiting)
        s.busy = true;
        s.clear_since = 0;
        Serial.print("E ");
        Serial.print(s.tag);
        Serial.print(' ');
        Serial.println(now);
      }
    } else if (sees) {
      s.clear_since = 0;
    } else {
      if (s.clear_since == 0) s.clear_since = now ? now : 1;
      else if (now - s.clear_since >= g_debounce_ms) s.busy = false;   // really gone
    }
  }
  armService();
}

// ---------------- the tiny model: a few 3x3 convolutions, average, one output ----------------
#define MAX_CONV   4
#define ACT_MAX    9216                    // most numbers one layer may produce
#define IMG_MAX    (48 * 48 * 3)
#define MODEL_MAX  65536

static uint8_t *g_blob = NULL;
static bool g_model_ok = false;
static uint8_t g_in_size = 0, g_in_ch = 1, g_nconv = 0;
static float g_mean = 0.5f, g_std = 0.25f;
static ConvLayer g_conv[MAX_CONV];
static const float *g_dense_w = NULL;
static float g_dense_b = 0;
static float *g_act_a = NULL, *g_act_b = NULL;
static uint8_t g_img[IMG_MAX];

// Model file layout (all little-endian, everything on 4-byte boundaries):
//   "TNY1" | in_size, in_channels, n_conv, 0 | mean f32 | std f32
//   per conv: cin, cout, stride, 0 | weights f32[cout*cin*9] | bias f32[cout]
//   dense: weights f32[last cout] | bias f32
static const char *modelParse(uint8_t *blob, size_t n) {
  if (n < 20 || memcmp(blob, "TNY1", 4) != 0) return "bad header";
  uint8_t size = blob[4], ch = blob[5], nconv = blob[6];
  if (size < 8 || size > 48 || (ch != 1 && ch != 3) || nconv < 1 || nconv > MAX_CONV) return "bad shape";
  float mean, sd;
  memcpy(&mean, blob + 8, 4);
  memcpy(&sd, blob + 12, 4);
  if (!(sd > 1e-6f)) return "bad std";
  size_t off = 16;
  int h = size, c = ch;
  ConvLayer conv[MAX_CONV];
  for (int l = 0; l < nconv; l++) {
    if (off + 4 > n) return "short file";
    uint8_t cin = blob[off], cout = blob[off + 1], stride = blob[off + 2];
    off += 4;
    if (cin != c || cout < 1 || cout > 64 || stride < 1 || stride > 2) return "bad layer";
    size_t nw = (size_t)cout * cin * 9;
    if (off + 4 * (nw + cout) > n) return "short file";
    conv[l].cin = cin; conv[l].cout = cout; conv[l].stride = stride;
    conv[l].w = (const float *)(blob + off); off += 4 * nw;
    conv[l].b = (const float *)(blob + off); off += 4 * (size_t)cout;
    h = (h + 2 - 3) / stride + 1;
    c = cout;
    if ((size_t)h * h * c > ACT_MAX) return "layer too big";
  }
  if (off + 4 * ((size_t)c + 1) != n) return "wrong length";
  g_dense_w = (const float *)(blob + off);
  memcpy(&g_dense_b, blob + off + 4 * (size_t)c, 4);
  for (int l = 0; l < nconv; l++) g_conv[l] = conv[l];
  g_in_size = size; g_in_ch = ch; g_nconv = nconv; g_mean = mean; g_std = sd;
  return NULL;
}

static float modelRun(const uint8_t *px) {
  int h = g_in_size, c = g_in_ch;
  const float k = 1.0f / (255.0f * g_std), m = g_mean / g_std;
  int n = h * h * c;
  float *in = g_act_a, *out = g_act_b;
  for (int i = 0; i < n; i++) in[i] = (float)px[i] * k - m;
  for (int l = 0; l < g_nconv; l++) {
    const ConvLayer &L = g_conv[l];
    const int s = L.stride, ho = (h + 2 - 3) / s + 1, hw = h * h, howo = ho * ho;
    for (int co = 0; co < L.cout; co++) {
      const float *wco = L.w + (size_t)co * L.cin * 9;
      const float bias = L.b[co];
      float *o = out + (size_t)co * howo;
      for (int oy = 0; oy < ho; oy++) {
        const int y0 = oy * s - 1;
        for (int ox = 0; ox < ho; ox++) {
          const int x0 = ox * s - 1;
          float acc = bias;
          for (int ci = 0; ci < L.cin; ci++) {
            const float *ip = in + (size_t)ci * hw;
            const float *wp = wco + ci * 9;
            for (int ky = 0; ky < 3; ky++) {
              const int iy = y0 + ky;
              if (iy < 0 || iy >= h) continue;
              const float *row = ip + iy * h;
              for (int kx = 0; kx < 3; kx++) {
                const int ix = x0 + kx;
                if (ix < 0 || ix >= h) continue;
                acc += row[ix] * wp[ky * 3 + kx];
              }
            }
          }
          o[oy * ho + ox] = acc > 0 ? acc : 0;
        }
      }
      pollIR();                             // stay responsive to the belt while thinking
    }
    float *t = in; in = out; out = t;
    h = ho; c = L.cout;
  }
  float z = g_dense_b;
  const int hw = h * h;
  for (int ci = 0; ci < c; ci++) {
    float sum = 0;
    const float *ip = in + (size_t)ci * hw;
    for (int i = 0; i < hw; i++) sum += ip[i];
    z += g_dense_w[ci] * (sum / hw);
  }
  return 1.0f / (1.0f + expf(-z));
}

// ---------------- serial helpers ----------------
static uint32_t sum16(const uint8_t *p, size_t n) {
  uint32_t s = 0;
  for (size_t i = 0; i < n; i++) s = (s + p[i]) & 0xFFFF;
  return s;
}

// Read exactly n raw bytes (the photo or the model). false = they did not all arrive in time.
static bool readBytesExact(uint8_t *dst, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  uint32_t last = millis();
  while (got < n) {
    int avail = Serial.available();
    if (avail > 0) {
      size_t want = n - got;
      if ((size_t)avail < want) want = (size_t)avail;
      got += Serial.readBytes((char *)(dst + got), want);
      last = millis();
    } else {
      if (millis() - last > timeout_ms) return false;
      pollIR();
#ifdef HOST_TEST
      hostIdle();
#endif
    }
  }
  g_last_rx = millis();
  return true;
}

static void cmdModel(long n, long want_sum) {
  if (n < 20 || n > MODEL_MAX) { Serial.println("W ERR size"); return; }
  uint8_t *blob = (uint8_t *)malloc((size_t)n);
  if (!blob) { Serial.println("W ERR memory"); return; }
  if (!readBytesExact(blob, (size_t)n, 1500)) { free(blob); Serial.println("W ERR timeout"); return; }
  if ((long)sum16(blob, (size_t)n) != want_sum) { free(blob); Serial.println("W ERR checksum"); return; }
  const char *err = modelParse(blob, (size_t)n);   // changes nothing if the file is refused
  if (err) { free(blob); Serial.print("W ERR "); Serial.println(err); return; }
  if (g_blob) free(g_blob);
  g_blob = blob;                            // the layer pointers now point into this block
  g_model_ok = true;
  Serial.print("W OK ");
  Serial.print(g_nconv);
  Serial.print(' ');
  Serial.println(g_in_size);
}

static void cmdInfer(long id, long n, long want_sum) {
  uint32_t t0 = millis();
  if (n < 1 || n > IMG_MAX) { Serial.print("R "); Serial.print(id); Serial.println(" ERR size"); return; }
  if (!readBytesExact(g_img, (size_t)n, 300)) { Serial.print("R "); Serial.print(id); Serial.println(" ERR timeout"); return; }
  uint32_t rx_ms = millis() - t0;
  if ((long)sum16(g_img, (size_t)n) != want_sum) { Serial.print("R "); Serial.print(id); Serial.println(" ERR checksum"); return; }
  if (!g_model_ok) { Serial.print("R "); Serial.print(id); Serial.println(" ERR nomodel"); return; }
  if (n != (long)g_in_size * g_in_size * g_in_ch) { Serial.print("R "); Serial.print(id); Serial.println(" ERR shape"); return; }
  uint32_t t1 = micros();
  float p = modelRun(g_img);
  uint32_t us = micros() - t1;
  Serial.print("R ");
  Serial.print(id);
  Serial.print(' ');
  Serial.print((long)(p * 10000.0f + 0.5f));
  Serial.print(' ');
  Serial.print(us);
  Serial.print(' ');
  Serial.println(rx_ms);
}

static void handleLine(char *line) {
  char cmd = line[0];
  char *args = line + 1;
  long a = 0, b = 0, c = 0;
  int n = sscanf(args, "%ld %ld %ld", &a, &b, &c);
  switch (cmd) {
    case 'H':
      Serial.print("H SCQC-ESP32 ");
      Serial.print(FW_VERSION);
      Serial.print(' ');
      Serial.print(g_model_ok ? 1 : 0);
      Serial.print(' ');
#ifdef HOST_TEST
      Serial.println(200000);
#else
      Serial.println(ESP.getFreeHeap());
#endif
      break;
    case 'K':                                // keep-alive, carries the belt speed
    case 'B':
      if (n >= 1) { g_duty = (int)a; motorApply(g_duty); }
      if (cmd == 'B') Serial.println("OK");
      break;
    case 'P':
      armPush(n >= 1 && a > 0 ? (uint32_t)a : 350);
      break;
    case 'L':
      digitalWrite(PIN_LED_GOOD, a == 1 ? HIGH : LOW);
      digitalWrite(PIN_LED_BAD, a == 2 ? HIGH : LOW);
      break;
    case 'S':
      Serial.print("S ");
      Serial.print(irSees(g_ir[0]) ? 1 : 0); Serial.print(' ');
      Serial.print(irSees(g_ir[1]) ? 1 : 0); Serial.print(' ');
      Serial.print(g_duty); Serial.print(' ');
#ifdef HOST_TEST
      Serial.print(200000);
#else
      Serial.print(ESP.getFreeHeap());
#endif
      Serial.print(' ');
      Serial.println(millis());
      break;
    case 'C': {
      char name[24];
      long v = 0;
      if (sscanf(args, "%23s %ld", name, &v) == 2) {
        if (!strcmp(name, "servo_rest")) { g_servo_rest = (int)v; if (!g_arm_out) servoWrite(g_servo_rest); }
        else if (!strcmp(name, "servo_push")) g_servo_push = (int)v;
        else if (!strcmp(name, "ir_low")) g_ir_low = v != 0;
        else if (!strcmp(name, "debounce_ms")) g_debounce_ms = (uint32_t)v;
        else { Serial.println("C ERR name"); break; }
        Serial.println("OK");
      } else Serial.println("C ERR");
      break;
    }
    case 'W':
      if (n >= 2) cmdModel(a, b); else Serial.println("W ERR args");
      break;
    case 'I':
      if (n >= 3) cmdInfer(a, b, c); else Serial.println("R 0 ERR args");
      break;
    default:
      break;                                 // unknown line: ignore (noise on the cable)
  }
}

static char g_line[80];
static size_t g_len = 0;

void setup() {
  Serial.setRxBufferSize(8192);              // a whole photo fits, nothing is lost while the model runs
  Serial.begin(BAUD);
  pinMode(PIN_IR_CAMERA, INPUT_PULLUP);
  pinMode(PIN_IR_ARM, INPUT_PULLUP);
  pinMode(PIN_MOTOR_IN1, OUTPUT);
  pinMode(PIN_MOTOR_IN2, OUTPUT);
  pinMode(PIN_LED_GOOD, OUTPUT);
  pinMode(PIN_LED_BAD, OUTPUT);
  pinMode(PIN_LED_LINK, OUTPUT);
  pwmBegin();
  motorApply(0);
  servoWrite(g_servo_rest);
  g_act_a = (float *)malloc(sizeof(float) * ACT_MAX);
  g_act_b = (float *)malloc(sizeof(float) * ACT_MAX);
  // a box already in front of a sensor at power-up is not a new box
  for (int i = 0; i < 2; i++) g_ir[i].busy = irSees(g_ir[i]);
  delay(50);
  Serial.println();
  Serial.print("BOOT SCQC-ESP32 ");
  Serial.println(FW_VERSION);
}

void loop() {
  while (Serial.available() > 0) {
    int ch = Serial.read();
    if (ch < 0) break;
    g_last_rx = millis();
    if (!g_link_ok) { g_link_ok = true; digitalWrite(PIN_LED_LINK, HIGH); }
    if (ch == '\n' || ch == '\r') {
      if (g_len > 0) {
        g_line[g_len] = 0;
        g_len = 0;
        handleLine(g_line);                  // W and I read their raw bytes inside this call
      }
    } else if (g_len < sizeof(g_line) - 1) {
      g_line[g_len++] = (char)ch;
    } else {
      g_len = 0;                             // line too long: throw it away
    }
  }
  pollIR();
  // gateway silent -> stop the belt, blink the blue LED
  if (g_link_ok && millis() - g_last_rx > LINK_TIMEOUT_MS) {
    g_link_ok = false;
    g_duty = 0;
    motorApply(0);
  }
  if (!g_link_ok) digitalWrite(PIN_LED_LINK, (millis() / 400) % 2 ? HIGH : LOW);
#ifdef HOST_TEST
  hostIdle();
#endif
}
