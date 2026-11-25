# Container image of the cloud server - an alternative to the EC2 bootstrap (e.g. for ECS / any Docker host).
#   docker build -f cloud/Dockerfile -t smart-conveyor-server .
#   docker run -p 8000:8000 -e CONVEYOR_API_KEY=<key> -v conveyor-data:/data smart-conveyor-server
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONVEYOR_DATA_DIR=/data CONVEYOR_WHERE=docker
WORKDIR /app

COPY cloud/requirements-cloud.txt /tmp/requirements-cloud.txt
RUN pip install --no-cache-dir -r /tmp/requirements-cloud.txt \
 && useradd --system --uid 10001 --home-dir /app conveyor \
 && mkdir -p /data && chown conveyor /data

COPY common/ common/
COPY server/ server/
COPY sim/ sim/
COPY config/system.yaml config/system.yaml
COPY tools/__init__.py tools/check_server.py tools/
COPY models/cloud_model.onnx models/edge_model.onnx models/model_card.json models/q_table.npz \
     models/rule_params.json models/probe.jpg models/

USER conveyor
EXPOSE 8000
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["python", "-m", "server.app", "--cpu-only"]
