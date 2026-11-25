#!/usr/bin/env bash
# =============================================================================
# Make the Pi's Wi-Fi behave like a good / busy / bad network (Linux tc + netem)
#
#   sudo bash tools/net_shape.sh <scenario> [interface]
#
#   scenario : good | busy | bad | bwcap | lossy | stalls | down | show
#   interface: default wlan0  (use eth0 if the Pi is on a cable)
#
# It shapes traffic LEAVING the Pi (uploads to the cloud server + pings), which is
# exactly the direction that carries the box photos. Same names as the
# simulator (sim/scenarios.py) and the dashboard buttons.
#
# To let the dashboard buttons run this without a password, add ONE line with
# `sudo visudo` (replace pi and the path with yours):
#   pi ALL=(root) NOPASSWD: /usr/bin/bash /home/pi/SmartConveyor_QL/tools/net_shape.sh *
# and set  pi.allow_shaping: true  in config/system.yaml
# =============================================================================
set -e
SC="${1:-show}"
IF="${2:-wlan0}"

clear_rules() { tc qdisc del dev "$IF" root 2>/dev/null || true; }

# The real path Pi 5 -> Internet -> AWS Mumbai already has ~30 ms round trip, so the
# delays below are ADDED on top to reach the simulator's conditions (sim/scenarios.py):
#   good ~30 ms | busy ~100 ms, 1% loss, 4 Mb/s | bad ~280 ms, 3% loss, 1.5 Mb/s
case "$SC" in
  good)   clear_rules ;;                                                         # normal Wi-Fi + Internet
  busy)   clear_rules; tc qdisc add dev "$IF" root netem delay 70ms 25ms distribution normal loss 1% rate 4mbit ;;
  bad)    clear_rules; tc qdisc add dev "$IF" root netem delay 250ms 70ms distribution normal loss 3% rate 1500kbit ;;
  bwcap)  clear_rules; tc qdisc add dev "$IF" root netem rate 300kbit ;;          # pings look fine, photos crawl
  lossy)  clear_rules; tc qdisc add dev "$IF" root netem delay 70ms 25ms distribution normal loss 6% rate 4mbit ;;
  stalls) clear_rules; tc qdisc add dev "$IF" root netem delay 5ms loss gemodel 1% 15% 100% 0% ;;  # burst outages (interference)
  down)   clear_rules; tc qdisc add dev "$IF" root netem loss 100% ;;
  show)   ;;
  *) echo "unknown scenario '$SC' (good busy bad bwcap lossy stalls down show)"; exit 1 ;;
esac
echo "[net_shape] $IF -> $SC"
tc qdisc show dev "$IF"
