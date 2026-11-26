#pragma once
#include "Arduino.h"
typedef enum { HTTP_ANY, HTTP_GET, HTTP_HEAD, HTTP_POST } HTTPMethod;
class Uri { public: Uri(const char *) {} Uri(const String &) {} };
class WebServer {
 public:
  typedef std::function<void(void)> THandlerFunction;
  explicit WebServer(int port = 80) { (void)port; }
  void begin() {}
  void handleClient() {}
  WebServer &on(const Uri &uri, HTTPMethod method, THandlerFunction fn) { (void)uri; (void)method; (void)fn; return *this; }
  void onNotFound(THandlerFunction) {}
  String arg(const String &) { return String(); }
  bool hasArg(const String &) const { return false; }
  void send(int code, const char *content_type = NULL, const String &content = String("")) { (void)code; (void)content_type; (void)content; }
  void sendHeader(const String &name, const String &value, bool first = false) { (void)name; (void)value; (void)first; }
};
