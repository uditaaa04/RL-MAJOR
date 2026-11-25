# The cloud part — AWS deployment guide

The ESP32 on the belt is the **edge**, the laptop next to it is the **gateway**. The **cloud** is one AWS EC2 server in Mumbai (`ap-south-1`) that runs
the big defect model, the live dashboard and a small model registry, with S3 for backups and CloudWatch for
monitoring. Everything is created by one script and removed by another.

---

## 1. Architecture

```
                         Internet (Wi-Fi -> ISP -> AWS Mumbai, ~30 ms)
 ┌──────── gateway laptop (+ ESP32) ──────┐        ┌──────────────── AWS ap-south-1 ────────────────────────────────┐
 │ IR-1 -> camera -> STATE -> Q-agent      │        │  EC2 instance (Elastic IP)  systemd: smart-conveyor            │
 │   EDGE/GATEWAY: tiny + small CNN here   │  HTTP  │   POST /infer      big CNN (MobileNetV3-L, ONNX)               │
 │   CLOUD: JPEG ─────────────────────────────────────>  GET  /ping       round-trip probe                            │
 │ ping /0.5 s, test image /2 s ─────────────────────>  /api/events     box log -> SQLite (/var/lib/smart-conveyor)  │
 │ box events, status /1 s ──────────────────────────>  /api/status     dashboard  <── browser (API key)            │
 │ Q-table backup /60 s ─────────────────────────────>  /api/qtable     backups                                      │
 │ tools/sync_models  <──────────────────────────────── /api/models     model registry (sha256)                      │
 │ every request: X-API-Key                │        │        │ instance role (no keys on disk)                       │
 └─────────────────────────────────────────┘        │        ├──> S3   smart-conveyor/data/  (box CSVs, Q-tables)   │
                                                    │        ├──> CloudWatch metrics + dashboard + alarms (e-mail)   │
                                                    │        └──> CloudWatch Logs  /smart-conveyor/<stack>            │
                                                    │  SSM Parameter Store: API key (SecureString)                  │
                                                    │  SSM Run Command: remote update, Session Manager: shell       │
                                                    └───────────────────────────────────────────────────────────────┘
```

**Why this split (the research point):** the cloud model is more accurate but its answer has to cross Wi-Fi +
the Internet and can arrive after the box reaches the reject arm. The Q-agent on the gateway learns, box by box,
when the cloud answer will be in time. The cloud never decides anything for the gateway — if AWS is down, the gateway
keeps inspecting every box locally.

### What gets created (`cloud/aws/cloudformation.yaml`)

| Resource | Purpose |
|---|---|
| EC2 instance (Amazon Linux 2023, encrypted gp3 disk, IMDSv2 only) | runs `python -m server.app` as a systemd service with auto-restart |
| Elastic IP | fixed address, so the gateway's config never changes |
| Security group | opens only the server port (8000) to `AllowedCidr`; SSH closed by default |
| IAM role (least privilege) | read the code bundle, write backups, read the API key, put metrics/logs, SSM |
| CloudWatch dashboard `<stack>-line` | missed %, cloud share %, answer time, server inference time, boxes/min, CPU |
| CloudWatch alarms + SNS e-mail (optional) | missed boxes > 5 % for 5 min; server status check failed |
| CloudWatch log group `/smart-conveyor/<stack>` | server + bootstrap logs (14-day retention) |
| (created by the deploy script) S3 bucket `smart-conveyor-<account>-<region>` | code bundle + backups, private, encrypted |
| (created by the deploy script) SSM SecureString `/smart-conveyor/<stack>-api-key` | the API key — never in the code or the bundle |

---

## 2. Before you start (once)

1. An AWS account (the free tier covers a `t3.micro`).
2. **AWS CLI v2** on the laptop: install <https://awscli.amazonaws.com/AWSCLIV2.msi>, then
   ```powershell
   aws configure          # access key of an IAM user with admin rights on YOUR account; region: ap-south-1
   ```
3. The project's Python environment (see README): `py -m venv .venv; .venv\Scripts\activate; pip install -r requirements-laptop.txt`

## 3. Deploy (one command, ~5–8 minutes the first time)

```powershell
.\cloud\deploy_aws.ps1
```

What it does, in order:
1. creates the private, encrypted S3 bucket (once);
2. creates a random 256-bit API key in SSM Parameter Store (once) and saves a copy in `cloud\.secrets\` (git-ignored);
3. builds `dist\server_bundle.zip` (server, shared code, simulator, ONNX models — no secrets, no PyTorch files) and uploads it;
4. opens port 8000 to **your current public IP only** (`/32`);
5. creates the CloudFormation stack and **waits until the server answers `/health`** on the instance;
6. prints the server URL, the dashboard URL, the CloudWatch dashboard and the next commands.

Useful options:

| Option | Meaning |
|---|---|
| `-AllowedCidr 0.0.0.0/0` | reachable from any network (the API key still protects it). Use when the gateway is on a different network (campus Wi-Fi) than the laptop. Better: the gateway network's public IP with `/32` |
| `-AlarmEmail you@example.com` | e-mail alarms (confirm the subscription mail from AWS) |
| `-InstanceType c6i.large` | faster, non-burstable CPU for measurement runs (`t3` instances can run out of CPU credits mid-experiment) |
| `-RotateApiKey` | make a new key (update the gateway afterwards) |
| `-Region`, `-StackName` | other region / a second deployment |

Running the script again updates the stack and pushes the newest code to the running instance.

## 4. Check it

```powershell
$env:CONVEYOR_API_KEY = Get-Content cloud\.secrets\api_key_smart-conveyor.txt
python -m tools.check_server --server http://<elastic-ip>:8000
```
It checks `/health`, that a request **without** the key is refused (401), that `/infer` works, and measures
ping, full answer time for one box photo and server inference time (median / 95th percentile) — the
"measured" column of the paper's Table IV. Results go to `results/server_check.json`.

Open the dashboard at `http://<elastic-ip>:8000/`, press **Key** and paste the API key (it is stored only in
that browser).

**No hardware yet?** Run the gateway with the pretend belt against the real cloud — the Internet path is real, nothing
is emulated:
```powershell
python -m gateway_node.main --mock --server http://<elastic-ip>:8000
```

## 5. Connect the gateway

Double-click `START_DEMO_ON_AWS.bat` once and paste the server address and the API key. They are kept in
`cloud/.secrets/aws_server.json` on this laptop. From then on `START_REAL_LINE.bat` (ESP32 + webcam) and
`MEASURE_SPEED_VS_ACCURACY.bat` use the AWS server as the cloud node.

By hand:
```powershell
$env:CONVEYOR_API_KEY = "<api-key>"
python -m tools.check_server --server http://<elastic-ip>:8000     # the latency that matters: from the gateway
python -m gateway_node.main --server http://<elastic-ip>:8000      # real belt
```

## 6. Day-to-day operation

| Task | How |
|---|---|
| Push new code/models to the server | `.\cloud\update_server.ps1` (SSM Run Command, no SSH; rolls back if the new version does not start) |
| Re-train the agent **in the cloud** | Session Manager shell (EC2 console → Connect) → `cd /opt/smart-conveyor/app && sudo /opt/smart-conveyor/venv/bin/python -m sim.train_q` → on the gateway `python -m tools.sync_models` |
| Get the box log | dashboard **Download CSV**, or `GET /api/export.csv?session=all`, or S3 `smart-conveyor/data/events/` |
| Restore the gateway's learned Q-table (new laptop) | `python -m tools.sync_models --restore-qtable`, then start with `--resume` |
| Server logs | CloudWatch → Log groups → `/smart-conveyor/<stack>`, or `journalctl -u smart-conveyor` / `/var/log/smart-conveyor/server.log` on the instance |
| Metrics | CloudWatch dashboard `<stack>-line` (link printed by the deploy script) |
| Pause to save money | EC2 console → Stop instance (the Elastic IP and the data stay; start it again before the next session) |

### API summary

| Endpoint | Key | What |
|---|---|---|
| `GET /ping`, `GET /health`, `GET /` | no | round-trip probe, health, dashboard page |
| `POST /infer` | yes | JPEG → `{p_damaged, label, server_ms, heatmap}` |
| `POST /api/events`, `POST /api/status`, `GET /api/state` | yes | gateway → dashboard data |
| `GET/POST /api/control`, `POST /api/reset` | yes | dashboard buttons (validated: unknown keys/values are ignored) |
| `GET /api/export.csv?session=current|all|<n>` | yes | box log |
| `GET /api/models`, `GET /api/models/<file>` | yes | model registry (whitelisted files only; the big model never leaves the server) |
| `POST /api/qtable`, `GET /api/qtable/latest` | yes | Q-table backups (validated `.npz`, last 20 kept per device) |

## 7. Security — what is done, and the honest limits

Done:
* every API call needs `X-API-Key` (compared in constant time); the key lives in SSM Parameter Store
  (encrypted) and in `/etc/smart-conveyor.env` (mode 600) — never in the code, the bundle or git;
* port 8000 open only to `AllowedCidr`; SSH closed (use Session Manager);
* IAM role with least privilege (only its own S3 prefixes, its own log group, its own metric namespace);
* encrypted disk and S3 objects, S3 public access blocked, IMDSv2 only;
* the service runs as an unprivileged user and can write only its data and log folders;
* uploads limited to 4 MB; dashboard control values validated; model downloads whitelisted.

Limits (say them in the viva):
* The server speaks **plain HTTP**, so the key could be read by someone sniffing the same Wi-Fi. For a testbed
  this is acceptable with a tight `AllowedCidr`. Production would add TLS — e.g. a domain name + Caddy as an
  HTTPS reverse proxy, or an Application Load Balancer with an ACM certificate (adds one TLS handshake per new
  connection; the gateway's keep-alive sessions avoid repeating it per box).
* One shared key for the gateway and the dashboard; per-device keys or AWS IoT Core certificates would be next.

## 8. Cost (approximate — check the AWS pricing page for current numbers)

| Item | Rough cost in Mumbai |
|---|---|
| `t3.micro` (default, as in the paper) | about US$0.01 per hour; free tier for eligible accounts |
| `t3.small` | about US$0.02–0.03 per hour (≈ US$17/month if left on 24×7) |
| `c6i.large` (long measurement runs) | about US$0.09 per hour — use it only while measuring |
| Elastic IP / public IPv4 | about US$0.005 per hour |
| 16 GB gp3 disk, S3, CloudWatch | well under US$2/month at this scale |

Stop the instance between sessions, and run the teardown when the project is over.

## 9. Remove everything

```powershell
.\cloud\teardown_aws.ps1                              # stack only; keeps the S3 backups and the key
.\cloud\teardown_aws.ps1 -DeleteBucket -DeleteApiKey  # everything (backups cannot be recovered)
```

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `AWS CLI v2 not found` | install the MSI, open a new terminal, `aws configure` |
| Stack fails at `ServerInstance` (timeout) | CloudWatch log stream `<id>/bootstrap`, or Session Manager → `sudo cat /var/log/smart-conveyor-bootstrap.log` |
| `check_server` cannot reach the server | your public IP changed / gateway on another network → re-run deploy with the right `-AllowedCidr` |
| 401 everywhere | wrong key: `Get-Content cloud\.secrets\api_key_<stack>.txt`; on the gateway check `cloud/.secrets/aws_server.json` |
| Dashboard "API key needed" | press **Key** and paste it |
| Answer times jump during long runs on `t3.*` | CPU credits used up → `-InstanceType c6i.large` for experiments |
| No default VPC in the account | create one: `aws ec2 create-default-vpc` |
| Metrics missing in CloudWatch | they appear after the first boxes (one point per minute); check `/health` → `aws.error` |
