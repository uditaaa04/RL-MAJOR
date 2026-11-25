#!/bin/bash
# =============================================================================
# Installs the Smart Conveyor cloud server on an Amazon Linux 2023 EC2 instance.
# Called once by the CloudFormation UserData (as root) after the code bundle is
# unpacked to /opt/smart-conveyor/app. Safe to run again (idempotent).
#
# Environment (set by the UserData):
#   AWS_REGION ARTIFACT_BUCKET BUNDLE_KEY API_KEY_PARAM SERVER_PORT LOG_GROUP LINE_NAME
#
# Result:
#   systemd service  smart-conveyor   (python -m server.app, user "conveyor", auto-restart)
#   secrets          /etc/smart-conveyor.env  (mode 600: API key + AWS settings)
#   data             /var/lib/smart-conveyor  (SQLite box log, Q-table backups)
#   logs             /var/log/smart-conveyor/server.log -> CloudWatch Logs group $LOG_GROUP
#   update script    /opt/smart-conveyor/update.sh  (pull a new bundle from S3 + restart)
# =============================================================================
set -euo pipefail

: "${AWS_REGION:?}" "${ARTIFACT_BUCKET:?}" "${BUNDLE_KEY:?}" "${API_KEY_PARAM:?}"
SERVER_PORT="${SERVER_PORT:-8000}"
LOG_GROUP="${LOG_GROUP:-/smart-conveyor/server}"
LINE_NAME="${LINE_NAME:-line-1}"
BASE=/opt/smart-conveyor
APP="$BASE/app"
VENV="$BASE/venv"

echo "[bootstrap] packages"
dnf install -y python3.11 python3.11-pip unzip

echo "[bootstrap] service user + folders"
id -u conveyor >/dev/null 2>&1 || useradd --system --home-dir "$BASE" --shell /sbin/nologin conveyor
mkdir -p /var/lib/smart-conveyor /var/log/smart-conveyor
chown -R conveyor:conveyor /var/lib/smart-conveyor /var/log/smart-conveyor

echo "[bootstrap] python environment"
[ -x "$VENV/bin/python" ] || python3.11 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$APP/cloud/requirements-cloud.txt"

echo "[bootstrap] secrets (API key from SSM Parameter Store, never stored in the code bundle)"
KEY=$(aws ssm get-parameter --name "$API_KEY_PARAM" --with-decryption \
      --query Parameter.Value --output text --region "$AWS_REGION")
umask 077
cat > /etc/smart-conveyor.env <<EOF
CONVEYOR_API_KEY=$KEY
CONVEYOR_S3_BUCKET=$ARTIFACT_BUCKET
CONVEYOR_S3_PREFIX=smart-conveyor/data
CONVEYOR_METRICS=1
CONVEYOR_DATA_DIR=/var/lib/smart-conveyor
CONVEYOR_WHERE=AWS $AWS_REGION
CONVEYOR_PORT=$SERVER_PORT
CONVEYOR_LINE=$LINE_NAME
AWS_REGION=$AWS_REGION
AWS_DEFAULT_REGION=$AWS_REGION
PYTHONDONTWRITEBYTECODE=1
EOF
umask 022
chmod 600 /etc/smart-conveyor.env

echo "[bootstrap] systemd service"
install -m 644 "$APP/cloud/aws/smart-conveyor.service" /etc/systemd/system/smart-conveyor.service
systemctl daemon-reload
systemctl enable smart-conveyor
systemctl restart smart-conveyor

echo "[bootstrap] update script"
cat > "$BASE/update.sh" <<EOF
#!/bin/bash
# Pull the newest code bundle from S3 and restart (run by cloud/update_server.ps1 through SSM).
set -euo pipefail
aws s3 cp "s3://$ARTIFACT_BUCKET/$BUNDLE_KEY" "$BASE/bundle.zip" --region "$AWS_REGION"
rm -rf "$BASE/app.new" && mkdir -p "$BASE/app.new"
unzip -q -o "$BASE/bundle.zip" -d "$BASE/app.new"
[ -f "$APP/config/calibration.json" ] && cp "$APP/config/calibration.json" "$BASE/app.new/config/" || true
"$VENV/bin/pip" install --quiet -r "$BASE/app.new/cloud/requirements-cloud.txt"
rm -rf "$BASE/app.old" && mv "$APP" "$BASE/app.old" && mv "$BASE/app.new" "$APP"
systemctl restart smart-conveyor
for i in \$(seq 1 45); do curl -fsS "http://127.0.0.1:$SERVER_PORT/health" && echo && exit 0; sleep 2; done
echo "server did not come back - rolling back"; rm -rf "$APP" && mv "$BASE/app.old" "$APP"; systemctl restart smart-conveyor; exit 1
EOF
chmod 750 "$BASE/update.sh"

echo "[bootstrap] CloudWatch agent for the server log (best effort)"
if dnf install -y amazon-cloudwatch-agent; then
  cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<EOF
{"logs": {"logs_collected": {"files": {"collect_list": [
  {"file_path": "/var/log/smart-conveyor/server.log", "log_group_name": "$LOG_GROUP", "log_stream_name": "{instance_id}/server"},
  {"file_path": "/var/log/smart-conveyor-bootstrap.log", "log_group_name": "$LOG_GROUP", "log_stream_name": "{instance_id}/bootstrap"}
]}}}}
EOF
  /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s \
     -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json || echo "cloudwatch agent not started"
else
  echo "cloudwatch agent not installed - logs stay on the instance"
fi

echo "[bootstrap] waiting for the server"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$SERVER_PORT/health"; then echo; echo "[bootstrap] server is up"; exit 0; fi
  sleep 2
done
echo "[bootstrap] server did not answer /health"; journalctl -u smart-conveyor --no-pager | tail -50
exit 1
