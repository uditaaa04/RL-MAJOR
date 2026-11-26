/*
  Smart Conveyor QC  -  EDGE NODE (ESP32)
  =====================================================================
  What this board does
    1. Runs the conveyor: belt motor, two IR sensors, reject arm (servo).
    2. Runs the TINY edge model: the gateway laptop sends a very small grey photo,
       the ESP32 answers "chance this box is damaged" a few milliseconds later.
    3. Safety: if the gateway laptop goes silent for 3 seconds, the belt stops.
    4. Wi-Fi (version 4, on the board's second processor core, so it can never slow down 1-3):
       - joins the Wi-Fi saved in its memory (the first time: DEFAULT_WIFI_SSID below);
       - cannot join -> opens its own hotspot "SmartConveyor-XXXX" (password AP_PASS below). Join it with a
         phone or laptop: the setup page opens by itself (or go to http://192.168.4.1). Pick the Wi-Fi, type
         its password, Save. Back online, the hotspot closes. If the Wi-Fi drops for 20 s, the hotspot returns;
       - online: its own live page at http://smartconveyor.local (or its IP address), and it sends its sensors,
         belt and arm straight to the cloud dashboard (https, checked with the root certificates below).
         The cloud address and API key come over the USB cable from the gateway (commands U and Y).

  How it talks: USB cable to the gateway laptop (serial, 460800 baud) for everything that is time-critical.
  No extra Arduino libraries are needed (WiFi, WebServer, DNSServer, Preferences, HTTPClient come with the board).

  Flash it
    Arduino IDE 2  ->  Tools > Board > "ESP32 Dev Module"
                   ->  Tools > Partition Scheme > "Huge APP (3MB No OTA/1MB SPIFFS)"   (Wi-Fi + https need room)
                   ->  pick the COM port  ->  Upload.
    Close the Serial Monitor before starting the gateway program (only one program can use the port).

  Text commands (one per line; you can type them in the Serial Monitor at 460800 baud to test)
    H                hello                 -> H SCQC-ESP32 <version> <model loaded 0/1> <free memory>
    K <duty>         keep-alive + belt speed 0..255 (sent by the gateway every second)
    B <duty>         belt speed 0..255
    P [hold_ms]      push the reject arm out and back
    S                status                -> S <ir_camera> <ir_arm> <duty> <free memory> <uptime ms>
    C <name> <value> setting: servo_rest, servo_push (degrees), ir_low (1 = sensor gives LOW on a box),
                     debounce_ms
    W <bytes> <sum>  + raw bytes: load the tiny model           -> W OK <layers> <input size>
    I <id> <bytes> <sum> + raw bytes: one grey photo            -> R <id> <chance x10000> <compute us> <receive ms>
    N                Wi-Fi status   -> N <OFF|CONNECTING|ONLINE|LOST> <ip|-> <rssi> <hotspot 0/1> <cloud> <wifi name|->
    V <name><TAB><password>   join this Wi-Fi (saved in the board's memory)              -> OK
    X                forget the saved Wi-Fi (the hotspot opens)                          -> OK
    U <url>          cloud server address, e.g. https://name.onrender.com ("U -" = none) -> OK
    Y <key>          cloud API key ("Y -" = none)                                        -> OK
  Events the board sends by itself
    BOOT SCQC-ESP32 <version>     after power-up / reset
    E C <ms>                      IR sensor 1: a box is under the camera
    E A <ms>                      IR sensor 2: a box is at the reject arm
    # <text>                      Wi-Fi news for people reading the Serial Monitor (the gateway ignores it)
*/
#ifdef HOST_TEST
#include "host_stub.h"            // lets the same file be compiled and tested on a PC
#define WIFI_ON 0
#else
#include <Arduino.h>
#define WIFI_ON 1                 // 0 = build without Wi-Fi (USB only, like version 3)
#endif
#if WIFI_ON
#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <Preferences.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <time.h>
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------- pins: change here if you wire it differently ----------------
#define PIN_IR_CAMERA   32      // IR sensor 1 OUT  (box under the camera)
#define PIN_IR_ARM      33      // IR sensor 2 OUT  (box at the reject arm)
#define PIN_SERVO       18      // reject arm servo signal (orange / yellow wire)
#define PIN_MOTOR_ENA   25      // belt speed, PWM: HW-039 (BTS7960) RPWM, or L298N ENA (pull its jumper off)
#define PIN_MOTOR_IN1   26      // L298N IN1 only. With the HW-039 leave this pin empty (it still switches, harmless)
#define PIN_MOTOR_IN2   27      // L298N IN2 only. With the HW-039 leave this pin empty
#define PIN_LED_LINK     2      // the blue LED that is already ON the board: steady = gateway connected,
                                // blinking = waiting for the gateway. Nothing to wire.
// (There are no good / damaged lamps: the dashboard shows every verdict. GPIO 16 and 17 are free.)

#define BAUD            460800
#define FW_VERSION      4       // 2: the two verdict LEDs (GPIO 16, 17) and the L command were removed
                                // 3: a refused model or photo is read off the cable and thrown away (its bytes
                                //    can no longer be mistaken for commands); a new model replaces the old one
                                //    even when memory is short
                                // 4: Wi-Fi: saved network or setup hotspot, live page, data straight to the cloud
#define LINK_TIMEOUT_MS 3000    // no message from the gateway for this long -> stop the belt

// ---------------- Wi-Fi names and passwords ----------------
// Put yours in esp32_node/secrets.h (copy secrets.example.h). That file is never uploaded to GitHub.
#if !defined(HOST_TEST) && defined(__has_include)
#if __has_include("secrets.h")
#include "secrets.h"
#endif
#endif
#ifndef DEFAULT_WIFI_SSID
#define DEFAULT_WIFI_SSID ""              // main Wi-Fi, tried first (until another one is saved from the setup page)
#endif
#ifndef DEFAULT_WIFI_PASS
#define DEFAULT_WIFI_PASS ""
#endif
#ifndef BACKUP_WIFI_SSID
#define BACKUP_WIFI_SSID ""               // backup Wi-Fi (e.g. a phone hotspot), tried second
#endif
#ifndef BACKUP_WIFI_PASS
#define BACKUP_WIFI_PASS ""
#endif
#ifndef AP_NAME
#define AP_NAME "SmartConveyor-Setup"     // the board's own setup hotspot
#endif
#ifndef AP_PASS
#define AP_PASS "conveyor123"             // its password, also asked for by the Save button (8+ characters)
#endif
#ifndef AP_IP
#define AP_IP 192, 168, 4, 1              // the setup page address: http://192.168.4.1
#endif
#ifndef HOSTNAME
#define HOSTNAME "smartconveyor"          // the live page: http://smartconveyor.local
#endif
#define HEARTBEAT_MS 2000                 // readings to the cloud this often, and at once when something changes

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

// ---------------- what the Wi-Fi side reads (written only by the belt code; 32-bit values, read whole) --------
static volatile uint32_t g_count_cam = 0, g_count_arm = 0, g_pushes = 0;
static volatile bool g_net_kick = false;   // something changed: send the readings to the cloud now

// ---------------- reject arm (non-blocking) ----------------
static bool g_arm_out = false;
static uint32_t g_arm_back_at = 0;

static void armPush(uint32_t hold_ms) {
  servoWrite(g_servo_push);
  g_arm_out = true;
  g_arm_back_at = millis() + hold_ms;      // a second push while it is out simply keeps it out longer
  g_pushes = g_pushes + 1;
  g_net_kick = true;
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
        if (i == 0) g_count_cam = g_count_cam + 1; else g_count_arm = g_count_arm + 1;
        g_net_kick = true;
      }
    } else if (sees) {
      s.clear_since = 0;
    } else {
      if (s.clear_since == 0) s.clear_since = now ? now : 1;
      else if (now - s.clear_since >= g_debounce_ms) { s.busy = false; g_net_kick = true; }   // really gone
    }
  }
  armService();
}

// ---------------- the tiny model: a few 3x3 convolutions, average, one output ----------------
#define MAX_CONV   4
#define ACT_MAX    4608                    // most numbers one layer may produce: 24x24x8 for the 48x48 model
                                           // (2 buffers of this; 9216 left too little memory for https)
#define IMG_MAX    (48 * 48 * 3)
#define MODEL_MAX  65536

static uint8_t *g_blob = NULL;
static uint8_t *g_res = NULL;                 // model memory set aside at start-up (see cmdModel)
static size_t g_res_cap = 0;
#define MODEL_RESERVE 49152                    // 48 kB: the 48x48 model is 46,852 bytes
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

// The raw bytes of a refused model or photo are still on their way: read them and throw them away.
// Otherwise loop() would read them as text and a stray "P" in them would swing the arm.
// Stops after n bytes, or when the cable has been quiet for quiet_ms.
static void skipBytes(long n, uint32_t quiet_ms) {
  if (n > 1048576L) n = 1048576L;          // a header damaged on the cable can claim any size
  uint32_t last = millis();
  while (n > 0) {
    int avail = Serial.available();
    if (avail > 0) {
      while (avail-- > 0 && n > 0) { Serial.read(); n--; }
      last = millis();
    } else {
      if (millis() - last > quiet_ms) break;
      pollIR();
#ifdef HOST_TEST
      hostIdle();
#endif
    }
  }
  g_last_rx = millis();
}

#ifdef HOST_TEST
#define fwAlloc(n) hostMalloc(n)          // the PC test can make this fail on purpose
#else
#define fwAlloc(n) malloc(n)
#endif

// The model's memory is set aside at start-up (g_res), before Wi-Fi and https break the memory into small
// pieces: later, one free block of 47 kB is often not there any more ("W ERR memory" on the real board).
static void dropBlob(uint8_t *b) {
  if (b && b != g_res) free(b);
}

static void cmdModel(long n, long want_sum) {
  if (n < 20 || n > MODEL_MAX) { skipBytes(n, 200); Serial.println("W ERR size"); return; }
  if (!g_act_a || !g_act_b) { skipBytes(n, 200); Serial.println("W ERR memory"); return; }   // no work memory at start-up
  bool fits_res = g_res && (size_t)n <= g_res_cap;
  uint8_t *blob = (fits_res && g_blob != g_res) ? g_res : (uint8_t *)fwAlloc((size_t)n);
  if (!blob && g_blob && (fits_res || g_blob != g_res)) {   // no room for the old and the new together: the old goes
    dropBlob(g_blob);
    g_blob = NULL;
    g_model_ok = false;
    blob = fits_res ? g_res : (uint8_t *)fwAlloc((size_t)n);
  }
  if (!blob) { skipBytes(n, 200); Serial.println("W ERR memory"); return; }
  if (!readBytesExact(blob, (size_t)n, 1500)) { dropBlob(blob); Serial.println("W ERR timeout"); return; }
  if ((long)sum16(blob, (size_t)n) != want_sum) { dropBlob(blob); Serial.println("W ERR checksum"); return; }
  const char *err = modelParse(blob, (size_t)n);   // changes nothing if the file is refused
  if (err) { dropBlob(blob); Serial.print("W ERR "); Serial.println(err); return; }
  if (g_blob && g_blob != blob) dropBlob(g_blob);
  g_blob = blob;                            // the layer pointers now point into this block
  g_model_ok = true;
  Serial.print("W OK ");
  Serial.print(g_nconv);
  Serial.print(' ');
  Serial.println(g_in_size);
}

static void cmdInfer(long id, long n, long want_sum) {
  uint32_t t0 = millis();
  if (n < 1 || n > IMG_MAX) { skipBytes(n, 200); Serial.print("R "); Serial.print(id); Serial.println(" ERR size"); return; }
  if (!readBytesExact(g_img, (size_t)n, 300)) { Serial.print("R "); Serial.print(id); Serial.println(" ERR timeout"); return; }
  uint32_t rx_ms = millis() - t0;
  if ((long)sum16(g_img, (size_t)n) != want_sum) { Serial.print("R "); Serial.print(id); Serial.println(" ERR checksum"); return; }
  if (!g_model_ok || !g_act_a || !g_act_b) { Serial.print("R "); Serial.print(id); Serial.println(" ERR nomodel"); return; }
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

// =====================================================================================================
// Wi-Fi (version 4). Runs as its own task on processor core 0; the belt code above runs in loop() on core 1.
// The belt code only WRITES the readings below and the "please change" requests; this task only READS the
// readings and owns everything Wi-Fi (radio, web pages, saved settings, the cloud connection).
// Nothing here prints to the USB cable directly: news goes through g_netmsg and loop() prints it, so a
// Wi-Fi line can never land in the middle of a belt message.
// =====================================================================================================
struct NetRequest {                       // USB commands V / X / U / Y, handed to the Wi-Fi task
  bool wifi, forget, url, key;
  char ssid[33], pass[65], url_v[128], key_v[80];
};
struct NetStatus {                        // what N and the pages report
  char state[12], ip[16], ssid[33], cloud[40];
  int rssi;
  bool hotspot;
};
static NetRequest g_req;
static NetStatus g_net = { "OFF", "-", "-", "not set", 0, false };
static char g_netmsg[120];
static volatile bool g_netmsg_ready = false;

#if WIFI_ON
static portMUX_TYPE g_mux = portMUX_INITIALIZER_UNLOCKED;
#define NET_LOCK() portENTER_CRITICAL(&g_mux)
#define NET_UNLOCK() portEXIT_CRITICAL(&g_mux)
#else
#define NET_LOCK()
#define NET_UNLOCK()
#endif

static void copyStr(char *dst, size_t n, const char *src) {
  size_t i = 0;
  for (; src && src[i] && i + 1 < n; i++) dst[i] = src[i];
  dst[i] = 0;
}

#if WIFI_ON
// printed by loop() (core 1) as "# <text>"
static void netSay(const char *text) {
  if (g_netmsg_ready) return;             // the last one is not printed yet: drop this one
  copyStr(g_netmsg, sizeof g_netmsg, text);
  g_netmsg_ready = true;
}

enum NetState { NET_OFF, NET_CONNECTING, NET_ONLINE, NET_LOST };
static const char *NET_NAMES[] = { "OFF", "CONNECTING", "ONLINE", "LOST" };

static WebServer g_web(80);
static DNSServer g_dns;
static Preferences g_prefs;
static WiFiClientSecure g_tls;           // ONE object for the board's whole life: HTTPClient keeps a pointer to it,
                                         // so it is only ever stopped, never deleted (deleting it crashed the board)
static bool g_tls_cfg = false;           // certificate check set on g_tls
static HTTPClient g_http;
static bool g_http_ready = false;        // begin() done and headers added: the next POSTs reuse the connection
static String g_cloud_note = "";

static void cloudReset() {               // new address / key / certificate setting, or repeated failures
  g_http.end();
  g_tls.stop();                          // closes the https session and frees its buffers
  g_http_ready = false;
  g_tls_cfg = false;                     // set the certificate check again before the next connection
}

static NetState g_state = NET_OFF;
static uint32_t g_state_t = 0, g_online_t = 0, g_try_t = 0, g_hb_last = 0, g_hb_wait_until = 0;
static bool g_ap = false, g_web_on = false, g_mdns = false, g_insecure = false;
static int g_try_idx = 0;                 // 0 = saved / main Wi-Fi, 1 = backup Wi-Fi
static int g_hb_fail = 0;
static uint32_t g_hb_ms = 0;              // time of the last cloud request
static String g_ssid, g_pass, g_url, g_key, g_scan_json = "[]";
static bool g_scanning = false;
static char g_apname[33];

#include "root_ca.h"

static uint32_t g_pub_t = 0;
static void publish() {
  if (millis() - g_pub_t < 250) return;   // 4 times a second is plenty (and spares the memory)
  g_pub_t = millis();
  // Ask the radio FIRST, outside the lock: NET_LOCK is a critical section (interrupts off), and a Wi-Fi call
  // or a String inside it waits on another lock or allocates memory -> the ESP32 aborts and restarts.
  char ip[16] = "-";
  int rssi = 0;
  if (g_state == NET_ONLINE) {
    copyStr(ip, sizeof ip, WiFi.localIP().toString().c_str());
    rssi = WiFi.RSSI();
  }
  NET_LOCK();                             // only plain copies in here
  copyStr(g_net.state, sizeof g_net.state, NET_NAMES[g_state]);
  g_net.hotspot = g_ap;
  copyStr(g_net.ip, sizeof g_net.ip, ip);
  g_net.rssi = rssi;
  NET_UNLOCK();
}

static void setCloudNote(const char *t) {
  if (g_cloud_note == t) return;
  g_cloud_note = t;
  NET_LOCK();
  copyStr(g_net.cloud, sizeof g_net.cloud, t);
  NET_UNLOCK();
}

static void hotspotOn() {
  if (g_ap) return;
  WiFi.mode(WIFI_AP_STA);
  if (!WiFi.softAP(g_apname, AP_PASS)) {
    netSay("hotspot could not start (is AP_PASS 8 or more characters?)");
    return;
  }
  IPAddress want(AP_IP);
  if (!(WiFi.softAPIP() == want)) {
    WiFi.softAPConfig(want, want, IPAddress(255, 255, 255, 0));
  }
  g_dns.start(53, "*", WiFi.softAPIP());  // every name -> this board: phones open the setup page by themselves
  g_ap = true;
  char t[96];
  snprintf(t, sizeof t, "hotspot %s on, setup page http://%s", g_apname, WiFi.softAPIP().toString().c_str());
  netSay(t);
}

static void hotspotOff() {
  if (!g_ap) return;
  g_dns.stop();
  WiFi.softAPdisconnect(true);
  WiFi.mode(WIFI_STA);
  g_ap = false;
  netSay("hotspot off");
}

// which network to try now: saved / main first, then the backup
static bool pickNetwork(String &ssid, String &pass) {
  if (g_try_idx == 0 && g_ssid.length()) { ssid = g_ssid; pass = g_pass; return true; }
  if (g_try_idx == 1 && strlen(BACKUP_WIFI_SSID)) { ssid = BACKUP_WIFI_SSID; pass = BACKUP_WIFI_PASS; return true; }
  return false;
}

static void startConnect() {
  String ssid, pass;
  if (!pickNetwork(ssid, pass)) {
    g_try_idx = 0;
    if (!pickNetwork(ssid, pass)) {       // nothing saved at all: only the hotspot
      g_state = NET_OFF;
      hotspotOn();
      return;
    }
  }
  WiFi.disconnect(false, false);
  WiFi.begin(ssid.c_str(), pass.c_str());
  g_state = NET_CONNECTING;
  g_state_t = g_try_t = millis();
  NET_LOCK();
  copyStr(g_net.ssid, sizeof g_net.ssid, ssid.c_str());
  NET_UNLOCK();
  char t[80];
  snprintf(t, sizeof t, "joining Wi-Fi %s", ssid.c_str());
  netSay(t);
}

static void goOnline() {
  g_state = NET_ONLINE;
  g_online_t = millis();
  g_try_idx = 0;
  if (g_mdns) MDNS.end();
  g_mdns = MDNS.begin(HOSTNAME);
  if (g_mdns) MDNS.addService("http", "tcp", 80);
  configTime(0, 0, "pool.ntp.org", "time.google.com");    // a correct clock for checking certificates
  char t[96];
  snprintf(t, sizeof t, "online: http://%s  (http://%s.local)", WiFi.localIP().toString().c_str(), HOSTNAME);
  netSay(t);
}

// one step of the Wi-Fi plan, called about every 10 ms
static void netStep() {
  uint32_t now = millis();
  bool up = WiFi.status() == WL_CONNECTED;
  bool someone_on_hotspot = g_ap && WiFi.softAPgetStationNum() > 0;
  switch (g_state) {
    case NET_OFF:
      if (!g_ap) hotspotOn();
      break;
    case NET_CONNECTING:
      if (up) { goOnline(); break; }
      if (now - g_state_t > 15000) {               // 15 s without success
        if (g_try_idx == 0 && strlen(BACKUP_WIFI_SSID)) { g_try_idx = 1; startConnect(); break; }
        if (!g_ap) hotspotOn();
        g_state = NET_LOST;
        g_state_t = now;
      }
      break;
    case NET_ONLINE:
      if (!up) { g_state = NET_LOST; g_state_t = now; netSay("Wi-Fi lost, trying again"); break; }
      // back online: close the hotspot (later if a phone is still on it, so it can read the new address)
      if (g_ap && (now - g_online_t > 120000 || (now - g_online_t > 20000 && !someone_on_hotspot))) hotspotOff();
      break;
    case NET_LOST:
      if (up) { goOnline(); break; }
      if (now - g_state_t > 20000 && !g_ap) hotspotOn();
      // try again every 30 s, but not while someone is on the setup page (trying moves the radio channel)
      if (now - g_try_t > 30000 && !someone_on_hotspot) {
        g_try_idx = (g_try_idx == 0 && strlen(BACKUP_WIFI_SSID)) ? 1 : 0;
        startConnect();
      }
      break;
  }
  if (g_scanning) {
    int n = WiFi.scanComplete();
    if (n >= 0) {
      String j = "[";
      for (int i = 0; i < n && i < 20; i++) {
        String name = WiFi.SSID(i);
        if (!name.length()) continue;
        String e;
        for (size_t k = 0; k < name.length(); k++) {
          char c = name[k];
          if (c == '"' || c == '\\') e += '\\';
          if ((unsigned char)c >= 32) e += c;
        }
        if (j.length() > 1) j += ",";
        j += "{\"ssid\":\"" + e + "\",\"rssi\":" + String(WiFi.RSSI(i)) +
             ",\"open\":" + (WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "true" : "false") + "}";
      }
      j += "]";
      g_scan_json = j;
      WiFi.scanDelete();
      g_scanning = false;
    } else if (n == WIFI_SCAN_FAILED) {
      g_scanning = false;
    }
  }
  publish();
}

static void saveWifi(const String &ssid, const String &pass) {
  g_ssid = ssid;
  g_pass = pass;
  g_prefs.putString("ssid", ssid);
  g_prefs.putString("pass", pass);
  g_try_idx = 0;
  startConnect();
}

static void forgetWifi() {
  g_prefs.remove("ssid");
  g_prefs.remove("pass");
  g_ssid = DEFAULT_WIFI_SSID;              // back to the defaults compiled in
  g_pass = DEFAULT_WIFI_PASS;
  WiFi.disconnect(false, false);
  g_state = NET_LOST;
  g_state_t = millis() - 20000;            // open the hotspot now
  netSay("saved Wi-Fi forgotten");
}

// ---------------- the board's readings as JSON (pages and cloud) ----------------
static String jsonEsc(const char *s) {
  String e;
  for (; *s; s++) {
    if (*s == '"' || *s == '\\') e += '\\';
    if ((unsigned char)*s >= 32) e += *s;
  }
  return e;
}

static String liveJson() {
  NetStatus n;
  NET_LOCK();
  n = g_net;
  NET_UNLOCK();
  char b[900];
  snprintf(b, sizeof b,
           "{\"fw\":%d,\"ir_camera\":%d,\"ir_arm\":%d,\"duty\":%d,\"belt_on\":%d,\"count_camera\":%lu,"
           "\"count_arm\":%lu,\"pushes\":%lu,\"arm_out\":%d,\"usb_link\":%d,\"model\":%d,\"uptime_ms\":%lu,"
           "\"free_heap\":%lu,\"min_heap\":%lu,\"wifi\":\"%s\",\"ip\":\"%s\",\"rssi\":%d,\"hotspot\":%d,"
           "\"hotspot_name\":\"%s\",\"cloud\":\"%s\",\"cloud_ms\":%lu,\"host\":\"%s\",\"ssid\":\"%s\",\"cloud_url\":\"%s\"}",
           FW_VERSION, g_ir[0].busy ? 1 : 0, g_ir[1].busy ? 1 : 0, g_duty, g_duty > 0 ? 1 : 0,
           (unsigned long)g_count_cam, (unsigned long)g_count_arm, (unsigned long)g_pushes, g_arm_out ? 1 : 0,
           g_link_ok ? 1 : 0, g_model_ok ? 1 : 0, (unsigned long)millis(), (unsigned long)ESP.getFreeHeap(),
           (unsigned long)ESP.getMinFreeHeap(), n.state, n.ip, n.rssi, n.hotspot ? 1 : 0,
           jsonEsc(g_apname).c_str(), jsonEsc(n.cloud).c_str(), (unsigned long)g_hb_ms, HOSTNAME,
           jsonEsc(n.ssid).c_str(), jsonEsc(g_url.c_str()).c_str());
  return String(b);
}

// ---------------- readings to the cloud dashboard (https, keep-alive) ----------------
static void heartbeat() {
  if (g_state != NET_ONLINE) return;
  if (!g_url.length() || !g_key.length()) { setCloudNote("not set (plug into the gateway once)"); return; }
  uint32_t now = millis();
  bool due = now - g_hb_last >= HEARTBEAT_MS || (g_net_kick && now - g_hb_last >= 250);
  if (!due || (int32_t)(now - g_hb_wait_until) < 0) return;
  g_net_kick = false;
  g_hb_last = now;
  if (!g_http_ready) {                    // one https connection, kept open: no new handshake every time
    // a new https connection needs about 40 kB; with less, skip this time instead of risking the board
    if (ESP.getFreeHeap() < 45000 || ESP.getMaxAllocHeap() < 20000) {
      char m[40];
      snprintf(m, sizeof m, "waiting: only %lu kB free", (unsigned long)(ESP.getFreeHeap() / 1024));
      setCloudNote(m);
      g_hb_wait_until = now + 10000;
      return;
    }
    if (!g_tls_cfg) {
      if (g_insecure) g_tls.setInsecure();
      else g_tls.setCACert(ROOT_CA);
      g_tls_cfg = true;
    }
    g_http.setReuse(true);
    g_http.setConnectTimeout(6000);
    g_http.setTimeout(6000);
    if (!g_http.begin(g_tls, g_url + "/api/edge")) { setCloudNote("bad cloud address"); return; }
    g_http.addHeader("Content-Type", "application/json");
    g_http.addHeader("X-API-Key", g_key);
    g_http.addHeader("X-Device-Id", "esp32-1");
    g_http_ready = true;
  }
  uint32_t t0 = millis();
  int code = g_http.POST(liveJson());
  if (code > 0) g_http.getString();       // read the whole answer, so the connection can carry the next one
  g_hb_ms = millis() - t0;
  char t[40];
  if (code == 200) {
    g_hb_fail = 0;
    snprintf(t, sizeof t, "ok, %lu ms", (unsigned long)g_hb_ms);
  } else {
    g_hb_fail++;
    if (code == 401) snprintf(t, sizeof t, "wrong API key");
    else if (code < 0) snprintf(t, sizeof t, "no answer (%d)", code);
    else snprintf(t, sizeof t, "error %d", code);
    uint32_t back = 2000UL * (uint32_t)(g_hb_fail < 15 ? g_hb_fail : 15);   // wait longer after each failure
    g_hb_wait_until = now + back;
    g_http.end();                         // start a fresh connection next time
    g_http_ready = false;
    if (g_hb_fail >= 3) cloudReset();     // ... and a fresh https client after repeated failures
  }
  setCloudNote(t);
}

// ---------------- web pages ----------------
#include "web_pages.h"                    // PAGE_HEAD, PAGE_LIVE, PAGE_WIFI (kept out of the .ino on purpose)

static void sendPage(const char *body) {
  String html = PAGE_HEAD;
  html += body;
  g_web.sendHeader("Cache-Control", "no-store");
  g_web.send(200, "text/html; charset=utf-8", html);
}

static bool pinOk() {
  return g_web.arg("pin") == AP_PASS;
}

static void onSave() {
  if (!pinOk()) { sendPage("<p class=\"bad\">Wrong setup password. Nothing was changed.</p><p><a href=\"/wifi\">Back</a></p></main></body></html>"); return; }
  String ssid = g_web.arg("ssid"), pass = g_web.arg("pass"), url = g_web.arg("url"), key = g_web.arg("key");
  ssid.trim();
  url.trim();
  key.trim();
  if (!ssid.length() || ssid.length() > 32 || pass.length() > 63 || (pass.length() > 0 && pass.length() < 8)) {
    sendPage("<p class=\"bad\">The Wi-Fi name is empty or too long, or the password is shorter than 8 characters.</p><p><a href=\"/wifi\">Back</a></p></main></body></html>");
    return;
  }
  if (url.length()) {
    while (url.endsWith("/")) url.remove(url.length() - 1);
    if (url.startsWith("http://") || url.startsWith("https://")) { g_url = url; g_prefs.putString("url", url); cloudReset(); }
  }
  if (key.length()) { g_key = key; g_prefs.putString("key", key); cloudReset(); }
  bool ins = g_web.hasArg("insecure");
  if (ins != g_insecure) {
    g_insecure = ins;
    g_prefs.putBool("insecure", ins);
    cloudReset();
  }
  sendPage("<p>Saved. The board is joining the Wi-Fi now (up to 15 s).</p>"
           "<div class=\"row\"><span>Now</span><b id=\"now\">joining ...</b></div>"
           "<p>When it says <b>online</b>, the hotspot closes by itself and the board is reachable at the address shown. "
           "Your phone goes back to its usual Wi-Fi.</p><p><a href=\"/wifi\">Back</a></p>"
           "<script>function st(){fetch('/api/live').then(function(r){return r.json()}).then(function(d){"
           "document.getElementById('now').textContent=d.wifi==='ONLINE'?'online: http://'+d.ip+'  or  http://'+d.host+'.local'"
           ":d.wifi.toLowerCase()})}setInterval(st,1000);</script></main></body></html>");
  saveWifi(ssid, pass);
}

static void onForget() {
  if (!pinOk()) { sendPage("<p class=\"bad\">Wrong setup password. Nothing was changed.</p><p><a href=\"/wifi\">Back</a></p></main></body></html>"); return; }
  sendPage("<p>The saved Wi-Fi is forgotten. The board goes back to the default Wi-Fi in its program, "
           "and the setup hotspot opens.</p><p><a href=\"/wifi\">Back</a></p></main></body></html>");
  forgetWifi();
}

static void onScan() {
  if (!g_scanning && g_scan_json == "[]") {
    WiFi.scanNetworks(true);              // in the background; the page asks again
    g_scanning = true;
  }
  if (g_scanning) { g_web.send(200, "application/json", "{\"scanning\":true}"); return; }
  String body = "{\"networks\":" + g_scan_json + "}";
  g_scan_json = "[]";                     // the next visit searches again
  g_web.send(200, "application/json", body);
}

static void onNotFound() {
  if (g_ap) {                             // the hotspot: every address leads to the setup page
    g_web.sendHeader("Location", String("http://") + WiFi.softAPIP().toString() + "/wifi", true);
    g_web.send(302, "text/plain", "");
    return;
  }
  g_web.send(404, "text/plain", "not found");
}

static void webBegin() {
  g_web.on("/", HTTP_GET, []() { sendPage(PAGE_LIVE); });
  g_web.on("/wifi", HTTP_GET, []() { sendPage(PAGE_WIFI); });
  g_web.on("/api/live", HTTP_GET, []() {
    g_web.sendHeader("Cache-Control", "no-store");
    g_web.send(200, "application/json", liveJson());
  });
  g_web.on("/api/scan", HTTP_GET, onScan);
  g_web.on("/save", HTTP_POST, onSave);
  g_web.on("/forget", HTTP_POST, onForget);
  g_web.onNotFound(onNotFound);
  g_web.begin();
  g_web_on = true;
}

// requests from the USB cable (commands V X U Y), applied here so only this task touches the settings
static void applyRequests() {
  NetRequest r;
  NET_LOCK();
  r = g_req;
  g_req.wifi = g_req.forget = g_req.url = g_req.key = false;
  NET_UNLOCK();
  if (r.url && g_url != r.url_v) {        // written to flash only when it really changed
    g_url = r.url_v;
    g_prefs.putString("url", g_url);
    cloudReset();
  }
  if (r.key && g_key != r.key_v) {
    g_key = r.key_v;
    g_prefs.putString("key", g_key);
    cloudReset();
  }
  if (r.forget) forgetWifi();
  if (r.wifi) saveWifi(String(r.ssid), String(r.pass));
}

static void netTask(void *) {
  g_prefs.begin("scqc", false);
  g_ssid = g_prefs.getString("ssid", DEFAULT_WIFI_SSID);
  g_pass = g_prefs.getString("pass", DEFAULT_WIFI_PASS);
  g_url = g_prefs.getString("url", "");
  g_key = g_prefs.getString("key", "");
  g_insecure = g_prefs.getBool("insecure", false);
  copyStr(g_apname, sizeof g_apname, AP_NAME);
  WiFi.persistent(false);                 // the board keeps the Wi-Fi itself (Preferences), not twice
  WiFi.setHostname(HOSTNAME);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  webBegin();
  startConnect();
  for (;;) {
    applyRequests();
    netStep();
    if (g_ap) g_dns.processNextRequest();
    g_web.handleClient();
    heartbeat();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

static void netBegin() {
  xTaskCreatePinnedToCore(netTask, "wifi", 16384, NULL, 1, NULL, 0);
}
#else
static void netBegin() {}
#endif

// USB commands V X U Y N (handled in loop(), on core 1)
static const char *skipSpaces(const char *a) {
  while (*a == ' ') a++;
  return a;
}

static void cmdNet(char cmd, const char *args) {
  const char *a = skipSpaces(args);
  if (cmd == 'N') {
    NetStatus n;
    NET_LOCK();
    n = g_net;
    NET_UNLOCK();
    Serial.print("N ");
    Serial.print(n.state); Serial.print(' ');
    Serial.print(n.ip); Serial.print(' ');
    Serial.print(n.rssi); Serial.print(' ');
    Serial.print(n.hotspot ? 1 : 0); Serial.print(' ');
    Serial.print(n.cloud[0] == 'o' && n.cloud[1] == 'k' ? "ok" : "no"); Serial.print(' ');
    Serial.println(n.ssid);
    return;
  }
  NET_LOCK();
  if (cmd == 'V') {
    const char *tab = strchr(a, '\t');
    size_t ln = tab ? (size_t)(tab - a) : strlen(a);
    if (ln > 32) ln = 32;
    memcpy(g_req.ssid, a, ln);
    g_req.ssid[ln] = 0;
    copyStr(g_req.pass, sizeof g_req.pass, tab ? tab + 1 : "");
    g_req.wifi = ln > 0;
  } else if (cmd == 'X') {
    g_req.forget = true;
  } else if (cmd == 'U') {
    copyStr(g_req.url_v, sizeof g_req.url_v, strcmp(a, "-") ? a : "");
    size_t n = strlen(g_req.url_v);
    while (n > 0 && g_req.url_v[n - 1] == '/') g_req.url_v[--n] = 0;
    g_req.url = true;
  } else if (cmd == 'Y') {
    copyStr(g_req.key_v, sizeof g_req.key_v, strcmp(a, "-") ? a : "");
    g_req.key = true;
  }
  NET_UNLOCK();
  Serial.println("OK");
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
    case 'N':
    case 'V':
    case 'X':
    case 'U':
    case 'Y':
      cmdNet(cmd, args);
      break;
    default:
      break;                                 // unknown line: ignore (noise on the cable)
  }
}

static char g_line[200];
static size_t g_len = 0;

void setup() {
  Serial.setRxBufferSize(8192);              // a whole photo fits, nothing is lost while the model runs
  Serial.begin(BAUD);
  pinMode(PIN_IR_CAMERA, INPUT_PULLUP);
  pinMode(PIN_IR_ARM, INPUT_PULLUP);
  pinMode(PIN_MOTOR_IN1, OUTPUT);
  pinMode(PIN_MOTOR_IN2, OUTPUT);
  pinMode(PIN_LED_LINK, OUTPUT);
  pwmBegin();
  motorApply(0);
  servoWrite(g_servo_rest);
  g_act_a = (float *)malloc(sizeof(float) * ACT_MAX);
  g_act_b = (float *)malloc(sizeof(float) * ACT_MAX);
  g_res = (uint8_t *)malloc(MODEL_RESERVE);  // before Wi-Fi starts, while one big free block still exists
  g_res_cap = g_res ? MODEL_RESERVE : 0;
  // a box already in front of a sensor at power-up is not a new box
  for (int i = 0; i < 2; i++) g_ir[i].busy = irSees(g_ir[i]);
  delay(50);
  Serial.println();
  Serial.print("BOOT SCQC-ESP32 ");
  Serial.println(FW_VERSION);
  netBegin();                                // Wi-Fi on the other core (nothing to do in the PC test build)
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
  if (g_netmsg_ready) {                      // Wi-Fi news (only this task prints, so lines never mix)
    Serial.print("# ");
    Serial.println(g_netmsg);
    g_netmsg_ready = false;
  }
#ifdef HOST_TEST
  hostIdle();
#endif
}
