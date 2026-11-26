#pragma once
#include "Arduino.h"
#include "WiFiClientSecure.h"
class HTTPClient {
 public:
  bool begin(NetworkClient &client, String url) { (void)client; (void)url; return true; }
  void end() {}
  void setReuse(bool) {}
  void setConnectTimeout(int32_t) {}
  void setTimeout(uint16_t) {}
  void addHeader(const String &name, const String &value, bool first = false, bool replace = true) { (void)name; (void)value; (void)first; (void)replace; }
  int POST(String payload) { (void)payload; return 200; }
  String getString() { return String(); }
};
