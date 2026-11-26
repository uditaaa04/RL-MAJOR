#pragma once
#include "Arduino.h"
class Preferences {
 public:
  bool begin(const char *name, bool readOnly = false, const char *partition_label = NULL) { (void)name; (void)readOnly; (void)partition_label; return true; }
  String getString(const char *key, String defaultValue = String()) { (void)key; return defaultValue; }
  size_t putString(const char *key, String value) { (void)key; (void)value; return 0; }
  bool getBool(const char *key, bool defaultValue = false) { (void)key; return defaultValue; }
  size_t putBool(const char *key, bool value) { (void)key; (void)value; return 0; }
  bool remove(const char *key) { (void)key; return true; }
};
