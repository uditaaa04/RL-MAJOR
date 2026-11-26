#pragma once
#include "Arduino.h"
class NetworkClient { public: virtual ~NetworkClient() {} };
class NetworkClientSecure : public NetworkClient { public: void setInsecure() {} void setCACert(const char *) {} };
typedef NetworkClientSecure WiFiClientSecure;
