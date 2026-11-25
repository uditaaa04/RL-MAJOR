"""Named test conditions. The network names are the same as the dashboard's Wi-Fi buttons."""
from dataclasses import replace

from .env import Condition

GOOD = Condition(name="good", belt_mps=0.20, gap_s=1.0, rtt_s=0.030, jitter_s=0.004, loss=0.0, bw_mbps=10)
BUSY = Condition(name="busy", belt_mps=0.20, gap_s=1.0, rtt_s=0.100, jitter_s=0.030, loss=0.01, bw_mbps=4)
BAD = Condition(name="bad", belt_mps=0.20, gap_s=1.0, rtt_s=0.280, jitter_s=0.080, loss=0.03, bw_mbps=1.5)
FAST_BUSY = replace(BUSY, name="fast+busy", belt_mps=0.30)
SLOW_BUSY = replace(BUSY, name="slow+busy", belt_mps=0.13)
GW_BUSY = replace(BUSY, name="gateway-busy", gw_factor=5.0, cpu=0.95)
GW_OVERLOAD = replace(BAD, name="gateway-overload", gw_factor=60.0, cpu=1.0)   # bad network AND a choked gateway:
                                                                             # only the ESP32 can still answer in time
BW_CAP = replace(GOOD, name="bw-cap", bw_mbps=0.3)               # pings look fine, uploads crawl
SERVER_SLOW = replace(GOOD, name="server-slow", server_factor=25.0)  # cloud server overloaded
LOSSY = replace(BUSY, name="lossy", loss=0.06)
STALLS = replace(GOOD, name="wifi-stalls", stall_prob=0.20, stall_s=(0.4, 0.8))  # NEVER seen in pre-training

STEADY = [GOOD, BUSY, BAD, FAST_BUSY, GW_BUSY, GW_OVERLOAD, BW_CAP, SERVER_SLOW]   # (+ LOSSY, STALLS in E1)

# change-in-the-middle experiments: (title, before/after condition, changed condition)
CHANGES = [
    ("Network gets bad (visible in ping)", GOOD, BAD),
    ("Belt speeds up (less time per box)", SLOW_BUSY, FAST_BUSY),
    ("Cloud server overloaded (invisible to the gateway)", GOOD, SERVER_SLOW),
    ("Gateway overloaded while the network is bad", BAD, GW_OVERLOAD),
    ("Wi-Fi bandwidth capped (invisible in ping)", GOOD, BW_CAP),
    ("Wi-Fi stalls appear (NEVER seen in training)", GOOD, STALLS),
]
