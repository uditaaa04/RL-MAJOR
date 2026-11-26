# The cloud part on Render (free) - deployment guide

The cloud server (`server/app.py`: big model, live dashboard, API key) runs on a **Render free web service** in
Singapore: **https://smart-conveyor-cloud.onrender.com**. No credit card, https included. (The AWS EC2 version is
still in `cloud/` and `docs/CLOUD_AWS.md` as an alternative.)

## What runs where

| Node | Where | Talks to the next one by |
|---|---|---|
| Edge | ESP32 on the belt | USB serial, 460,800 baud (belt, sensors, arm, tiny model); its own Wi-Fi for its readings (firmware 4) |
| Gateway | laptop next to the belt (webcam, small model, Q-agent) | https + `X-API-Key` |
| Cloud | Render web service, Singapore (big model, dashboard) | - |

## Live data path (lowest delay)

```
IR sensor 1 -> ESP32 -> USB (~1 ms) -> gateway: photo, decision, check -> https POST /api/events -> Render
                                                                                   |  (wakes every waiting request)
dashboard (browser) <- GET /api/state?after=<n>&wait=20 (long polling) <----------+
gateway            <- GET /api/control?after=<n>&wait=20  <- dashboard buttons (belt on/off, push arm, check now)
ESP32 (Wi-Fi)       -> https POST /api/edge every 2 s and at once on a sensor change
```

* Nothing waits for a timer: the server answers a waiting request the moment something changes, so an update
  takes one trip over the Internet. Measured with the local test (same PC): a dashboard button -> gateway ->
  checked box -> dashboard in 18 ms; over the Internet add about 2 x 40-70 ms.
* An older browser page or gateway without long polling still works (it asks every 0.8 s / 1 s).
* Up to 6 requests may wait at once (`CONVEYOR_LONGPOLL_MAX`); more dashboards simply poll.

## Measured (from the gateway laptop in India, 2026-11-26)

| | median | 95th percentile |
|---|---|---|
| ping (`/ping`) | 71 ms | 148 ms |
| box photo checked in the cloud (`/infer`, round trip) | 83 ms | 97 ms |
| of that, the model on the server | 5.2 ms | 19 ms |

First live test of the instant path (same day, from the laptop, one try each): an ESP32-style reading posted to
`/api/edge` lit the dashboard's ESP32 panel in 84 ms, and a dashboard button reached the waiting gateway in 104 ms.
Not yet with the real board or belt.

The same test photo gave the same answer as on the laptop (good, 0.0026): the same model. Requests without the
key, or with a wrong one, get 401.

## Deploy (already done; to do it again)

1. Sign in to https://render.com with GitHub.
2. **New -> Blueprint** -> repository `uditaaa04/RL-MAJOR`, branch `claude/smart-conveyor-project` -> **Apply**.
   `render.yaml`: free plan, Singapore, `pip install -r cloud/requirements-render.txt`,
   `python -m server.app --cpu-only`, health check `/health`, a random API key in `CONVEYOR_API_KEY`,
   one model thread (`CONVEYOR_ONNX_THREADS=1`) and 16 web threads (`CONVEYOR_HTTP_THREADS`).
3. API key: service -> **Environment** -> `CONVEYOR_API_KEY`.

Every push to the branch redeploys automatically (about 1-2 minutes).

## Connect the gateway (already done on this laptop)

`cloud/.secrets/aws_server.json` (the file name is from the AWS version; never uploaded):
```json
{"url": "https://smart-conveyor-cloud.onrender.com", "key": "<CONVEYOR_API_KEY>"}
```
`START_REAL_LINE.bat`, `MANUAL_TEST.bat`, `START_DEMO_ON_AWS.bat` and `MEASURE_SPEED_VS_ACCURACY.bat` use it. They
wait up to 2 minutes for a sleeping server. When the gateway connects to an ESP32 with firmware 4, it copies the
address and key to the board, so the board can send its own readings over Wi-Fi.

Check it from the laptop:
```powershell
$env:CONVEYOR_API_KEY = "<key>"
python -m tools.check_server --server https://smart-conveyor-cloud.onrender.com
```

## Limits of the free plan (say them in the viva)

* **Sleeps after 15 minutes without requests**; the next request wakes it in about a minute. Open the dashboard
  2 minutes before a demo. While the gateway or the ESP32's Wi-Fi is sending, it stays awake.
* 750 free hours a month per workspace: one service awake all month uses about 720-744, so it just fits.
* **0.1 CPU, 512 MB RAM.** The model took 5.2 ms per photo in the test; under load it will be slower.
* **No disk that survives a restart**: the box log (SQLite) is lost when the service restarts, redeploys or
  sleeps. Download the CSV from the dashboard after a run.
* One shared API key for the gateway, the dashboard and the ESP32.
