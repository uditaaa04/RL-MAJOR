# The cloud part on Render (free) — deployment guide

Same cloud server as the AWS version (`server/app.py`: big model, live dashboard, API key), hosted on a
**Render free web service** instead of an EC2 instance. No credit card, HTTPS included.

## What runs where

| Node | Where | Talks to the next one by |
|---|---|---|
| Edge | ESP32 on the belt | USB serial, 460,800 baud |
| Gateway | laptop next to the belt (webcam, small model, Q-agent) | HTTPS + `X-API-Key` |
| Cloud | Render web service, Singapore region (big model, dashboard) | — |

## Deploy (once)

1. Sign in to <https://render.com> with GitHub.
2. **New → Blueprint** → repository `uditaaa04/RL-MAJOR` → **Apply**. Render reads `render.yaml`:
   free plan, Singapore, `pip install -r cloud/requirements-render.txt`, `python -m server.app --cpu-only`,
   health check `/health`, and a **random API key** in `CONVEYOR_API_KEY`.
3. Wait for **Live** (first build about 3–5 minutes). The address is `https://smart-conveyor-cloud(-xxxx).onrender.com`.
4. API key: service → **Environment** → `CONVEYOR_API_KEY` → reveal.

Every push to the branch `claude/smart-conveyor-project` redeploys automatically.

## Connect the gateway

`cloud/.secrets/aws_server.json` on the laptop (the file name is historical; any cloud address works):

```json
{"url": "https://<name>.onrender.com", "key": "<CONVEYOR_API_KEY>"}
```

or double-click `START_DEMO_ON_AWS.bat` once and paste the two values. Then `START_REAL_LINE.bat` and
`MEASURE_SPEED_VS_ACCURACY.bat` use the Render server as the cloud node. Check it from the laptop:

```powershell
$env:CONVEYOR_API_KEY = "<key>"
python -m tools.check_server --server https://<name>.onrender.com
```

## Limits of the free plan (say them in the viva)

* **Sleeps after 15 minutes without requests**; the next request wakes it in about a minute. Open the
  address 2 minutes before a demo. While the gateway runs it pings every 0.5 s, so it stays awake.
* **0.1 CPU, 512 MB RAM**: the big model answers much slower than on the laptop or on EC2. Report the time
  measured with `tools.check_server`, not the laptop number.
* **No disk that survives a restart**: the box log (SQLite) is lost when the service restarts, redeploys or
  sleeps. Download the CSV from the dashboard after a run.
* One shared API key for the gateway and the dashboard (same as the AWS version).
