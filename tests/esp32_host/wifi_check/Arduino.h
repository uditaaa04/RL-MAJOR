// Syntax-check stand-in for the ESP32 Arduino core (signatures as in arduino-esp32 2.x / 3.x).
// Only used by tests/test_esp32.py::test_wifi_firmware_compiles_against_stubs (g++ -fsyntax-only).
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <functional>
#define ESP_ARDUINO_VERSION_MAJOR 3
#define HIGH 1
#define LOW 0
#define INPUT_PULLUP 0x05
#define OUTPUT 0x03
class String {
 public:
  std::string s;
  String() {}
  String(const char *c) : s(c ? c : "") {}
  String(const String &o) = default;
  String &operator=(const String &o) = default;
  String &operator=(const char *c) { s = c ? c : ""; return *this; }
  explicit String(char c) : s(1, c) {}
  explicit String(int v) : s(std::to_string(v)) {}
  explicit String(long v) : s(std::to_string(v)) {}
  explicit String(unsigned int v) : s(std::to_string(v)) {}
  explicit String(unsigned long v) : s(std::to_string(v)) {}
  unsigned int length() const { return (unsigned int)s.size(); }
  const char *c_str() const { return s.c_str(); }
  char operator[](unsigned int i) const { return s[i]; }
  String &operator+=(const String &o) { s += o.s; return *this; }
  String &operator+=(const char *c) { s += c; return *this; }
  String &operator+=(char c) { s += c; return *this; }
  bool operator==(const String &o) const { return s == o.s; }
  bool operator==(const char *c) const { return s == c; }
  bool operator!=(const String &o) const { return s != o.s; }
  bool operator!=(const char *c) const { return s != c; }
  bool startsWith(const String &p) const { return s.rfind(p.s, 0) == 0; }
  bool endsWith(const String &p) const { return s.size() >= p.s.size() && s.compare(s.size() - p.s.size(), p.s.size(), p.s) == 0; }
  void remove(unsigned int i) { s.erase(i); }
  void trim() {}
};
inline String operator+(const String &a, const String &b) { String r(a); r += b; return r; }
inline String operator+(const String &a, const char *b) { String r(a); r += b; return r; }
inline String operator+(const char *a, const String &b) { String r(a); r += b; return r; }
class IPAddress {
 public:
  IPAddress() {}
  IPAddress(uint8_t, uint8_t, uint8_t, uint8_t) {}
  String toString() const { return String("0.0.0.0"); }
  bool operator==(const IPAddress &) const { return true; }
};
struct HardwareSerial {
  void setRxBufferSize(size_t) {}
  void begin(unsigned long) {}
  int available() { return 0; }
  int read() { return -1; }
  size_t readBytes(char *, size_t) { return 0; }
  size_t print(const char *) { return 0; }
  size_t print(char) { return 0; }
  size_t print(int) { return 0; }
  size_t print(long) { return 0; }
  size_t print(unsigned int) { return 0; }
  size_t print(unsigned long) { return 0; }
  size_t println(const char *) { return 0; }
  size_t println(int) { return 0; }
  size_t println(long) { return 0; }
  size_t println(unsigned int) { return 0; }
  size_t println(unsigned long) { return 0; }
  size_t println() { return 0; }
};
extern HardwareSerial Serial;
struct EspClass { uint32_t getFreeHeap() { return 0; } uint32_t getMinFreeHeap() { return 0; } };
extern EspClass ESP;
unsigned long millis();
unsigned long micros();
void delay(uint32_t);
void pinMode(uint8_t, uint8_t);
int digitalRead(uint8_t);
void digitalWrite(uint8_t, uint8_t);
bool ledcAttachChannel(uint8_t pin, uint32_t freq, uint8_t resolution, int8_t channel);
bool ledcWrite(uint8_t pin, uint32_t duty);
void configTime(long gmtOffset_sec, int daylightOffset_sec, const char *server1, const char *server2 = nullptr, const char *server3 = nullptr);
typedef struct { int owner; int count; } portMUX_TYPE;
#define portMUX_INITIALIZER_UNLOCKED {0, 0}
void portENTER_CRITICAL(portMUX_TYPE *);
void portEXIT_CRITICAL(portMUX_TYPE *);
typedef void (*TaskFunction_t)(void *);
typedef void *TaskHandle_t;
typedef uint32_t TickType_t;
#define pdMS_TO_TICKS(x) ((TickType_t)(x))
int xTaskCreatePinnedToCore(TaskFunction_t, const char *, uint32_t, void *, unsigned, TaskHandle_t *, int);
void vTaskDelay(TickType_t);
