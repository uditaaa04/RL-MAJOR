// Minimal stand-in for the Arduino functions the ESP32 sketch uses, so the SAME sketch file can be
// compiled with g++ and tested on a PC (tests/test_esp32_firmware.py). Serial = stdin/stdout,
// pin levels are set by lines "pin <n> <0|1>" written to file descriptor 3.
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>

#define HIGH 1
#define LOW 0
#define INPUT_PULLUP 2
#define OUTPUT 1

static int host_pins[64];
static uint32_t host_pwm[64];
static bool host_init_done = false;
static std::string host_ctl;

static inline uint64_t host_now_us() {
  static auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
}
static inline uint32_t millis() { return (uint32_t)(host_now_us() / 1000); }
static inline uint32_t micros() { return (uint32_t)host_now_us(); }
static inline void delay(uint32_t ms) { usleep(ms * 1000); }

static inline void host_init() {
  if (host_init_done) return;
  host_init_done = true;
  for (int i = 0; i < 64; i++) host_pins[i] = HIGH;          // pull-ups: idle = HIGH
  fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
  fcntl(3, F_SETFL, fcntl(3, F_GETFL) | O_NONBLOCK);
}
static inline void host_poll_ctl() {
  char buf[256];
  ssize_t n;
  while ((n = read(3, buf, sizeof buf)) > 0) host_ctl.append(buf, (size_t)n);
  size_t nl;
  while ((nl = host_ctl.find('\n')) != std::string::npos) {
    int pin = 0, v = 0;
    if (sscanf(host_ctl.substr(0, nl).c_str(), "pin %d %d", &pin, &v) == 2 && pin >= 0 && pin < 64) host_pins[pin] = v;
    host_ctl.erase(0, nl + 1);
  }
}
static inline void pinMode(int, int) { host_init(); }
static inline int digitalRead(int pin) { host_poll_ctl(); return host_pins[pin]; }
static inline void digitalWrite(int pin, int v) {
  if (host_pins[pin] != v && (pin == 16 || pin == 17)) fprintf(stderr, "led %d %d\n", pin, v);
  host_pins[pin] = v;
}
static inline void hostPwm(int pin, uint32_t duty) {
  if (host_pwm[pin] != duty) fprintf(stderr, "pwm %d %u\n", pin, (unsigned)duty);
  host_pwm[pin] = duty;
}
static inline void hostIdle() { usleep(200); }

struct HostSerial {
  std::string rx;
  void setRxBufferSize(int) {}
  void begin(long) { host_init(); }
  void fill() {
    char buf[4096];
    ssize_t n;
    while ((n = ::read(0, buf, sizeof buf)) > 0) rx.append(buf, (size_t)n);
    if (n == 0) _exit(0);                                    // the test closed the port
  }
  int available() { fill(); return (int)rx.size(); }
  int read() { fill(); if (rx.empty()) return -1; int c = (unsigned char)rx[0]; rx.erase(0, 1); return c; }
  size_t readBytes(char *dst, size_t n) { fill(); if (n > rx.size()) n = rx.size(); memcpy(dst, rx.data(), n); rx.erase(0, n); return n; }
  void out(const std::string &s) { size_t o = 0; while (o < s.size()) { ssize_t n = ::write(1, s.data() + o, s.size() - o); if (n > 0) o += (size_t)n; else usleep(100); } }
  void print(const char *s) { out(s); }
  void print(char c) { out(std::string(1, c)); }
  void print(int v) { out(std::to_string(v)); }
  void print(long v) { out(std::to_string(v)); }
  void print(unsigned int v) { out(std::to_string(v)); }
  void print(unsigned long v) { out(std::to_string(v)); }
  template <class T> void println(T v) { print(v); out("\r\n"); }
  void println() { out("\r\n"); }
};
static HostSerial Serial;

void setup();
void loop();
int main() {
  setup();
  for (;;) loop();
}
