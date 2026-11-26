#pragma once
#include "Arduino.h"
typedef enum { WIFI_OFF = 0, WIFI_STA, WIFI_AP, WIFI_AP_STA } wifi_mode_t;
typedef enum { WL_IDLE_STATUS = 0, WL_NO_SSID_AVAIL, WL_SCAN_COMPLETED, WL_CONNECTED, WL_CONNECT_FAILED, WL_CONNECTION_LOST, WL_DISCONNECTED } wl_status_t;
typedef enum { WIFI_AUTH_OPEN = 0, WIFI_AUTH_WEP, WIFI_AUTH_WPA_PSK, WIFI_AUTH_WPA2_PSK } wifi_auth_mode_t;
#define WIFI_SCAN_RUNNING (-1)
#define WIFI_SCAN_FAILED (-2)
struct WiFiClass {
  bool mode(wifi_mode_t) { return true; }
  wl_status_t begin(const char *ssid, const char *passphrase = NULL) { (void)ssid; (void)passphrase; return WL_IDLE_STATUS; }
  bool disconnect(bool wifioff = false, bool eraseap = false, unsigned long timeoutLength = 100) { (void)wifioff; (void)eraseap; (void)timeoutLength; return true; }
  wl_status_t status() { return WL_IDLE_STATUS; }
  IPAddress localIP() { return IPAddress(); }
  int8_t RSSI() { return 0; }
  int32_t RSSI(uint8_t) { return 0; }
  String SSID(uint8_t) const { return String(); }
  wifi_auth_mode_t encryptionType(uint8_t) { return WIFI_AUTH_OPEN; }
  int16_t scanNetworks(bool async = false, bool show_hidden = false) { (void)async; (void)show_hidden; return 0; }
  int16_t scanComplete() { return 0; }
  void scanDelete() {}
  bool softAP(const char *ssid, const char *passphrase = NULL, int channel = 1, int ssid_hidden = 0, int max_connection = 4, bool ftm_responder = false) { (void)ssid; (void)passphrase; (void)channel; (void)ssid_hidden; (void)max_connection; (void)ftm_responder; return true; }
  bool softAPConfig(IPAddress, IPAddress, IPAddress) { return true; }
  IPAddress softAPIP() { return IPAddress(); }
  bool softAPdisconnect(bool wifioff = false) { (void)wifioff; return true; }
  uint8_t softAPgetStationNum() { return 0; }
  void persistent(bool) {}
  bool setHostname(const char *) { return true; }
  bool setAutoReconnect(bool) { return true; }
};
extern WiFiClass WiFi;
