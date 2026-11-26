#pragma once
#include "Arduino.h"
class DNSServer {
 public:
  bool start(const uint16_t &port, const String &domainName, const IPAddress &resolvedIP) { (void)port; (void)domainName; (void)resolvedIP; return true; }
  void processNextRequest() {}
  void stop() {}
};
