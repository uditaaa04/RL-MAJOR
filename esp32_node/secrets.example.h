// Copy this file to secrets.h (same folder) and fill it in. secrets.h is never uploaded to GitHub.
// The ESP32 sees 2.4 GHz Wi-Fi only (a phone hotspot works; on an iPhone turn on "Maximise Compatibility").
#pragma once
#define DEFAULT_WIFI_SSID "your-wifi-name"    // main Wi-Fi, tried first
#define DEFAULT_WIFI_PASS "your-wifi-password"
#define BACKUP_WIFI_SSID  ""                  // backup Wi-Fi (optional), tried second
#define BACKUP_WIFI_PASS  ""
#define AP_NAME           "SmartConveyor-Setup"   // the board's own setup hotspot when no Wi-Fi works
#define AP_PASS           "change-me-123"         // 8 or more characters; also asked for by the Save button
#define AP_IP             192, 168, 4, 1          // setup page: http://192.168.4.1
#define HOSTNAME          "smartconveyor"         // live page: http://smartconveyor.local
