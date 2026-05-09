# Autonomous Manufacturing Agent

Cluster 17 of the NextAura 500 SDKs / 25 Clusters project.

AI-native production floor — perception, planning, and quality at machine speed. The LangGraph agent ingests production telemetry, identifies patterns, surfaces issues before line stops, and triggers escalation workflows autonomously.

## Architecture

- YOLO + OpenCV for vision-based quality inspection
- NVIDIA DeepStream for multi-camera GPU video pipeline
- cuRobo + MoveIt for GPU-accelerated robot motion planning
- LangGraph + Ollama for autonomous operator assistant
- Kafka for production event streaming
- TimescaleDB for time-series telemetry storage
- Temporal for durable workflow orchestration
- Prometheus + Grafana for production floor observability
- Great Expectations for telemetry data quality
- NATS for low-latency machine-to-agent messaging

## SDKs Used

NVIDIA DeepStream SDK, YOLO (Ultralytics), OpenCV SDK, NVIDIA VPI, NVIDIA Isaac Perceptor, cuRobo, MoveIt SDK, ROS 2 SDK, LangGraph, Ollama, Kafka SDK, TimescaleDB, FastAPI, Grafana SDK, Prometheus Client, Redis SDK, NATS SDK, Temporal SDK, Weights & Biases, Great Expectations

## Quickstart

```bash
pip install -r requirements.txt
docker-compose up -d  # Kafka, TimescaleDB, Prometheus

# Run vision inspection on a camera feed
python main.py --mode inspect --source camera://0

# Start the autonomous operator agent
python main.py --mode agent --line LINE_A

# Simulate production telemetry
python main.py --mode simulate --duration 60

# Launch full production floor demo
python main.py --mode demo
```

## Structure

```
perception/
  inspection.py     YOLO vision quality inspection pipeline
  deepstream.py     NVIDIA DeepStream multi-camera pipeline
telemetry/
  kafka_producer.py Production event streaming to Kafka
  timescale.py      TimescaleDB time-series telemetry storage
  quality.py        Great Expectations telemetry validation
agent/
  operator.py       LangGraph autonomous operator assistant
  escalation.py     Temporal workflow orchestration
planning/
  motion.py         cuRobo GPU motion planning
  ros2_bridge.py    ROS 2 integration bridge
observability/
  metrics.py        Prometheus production metrics
  dashboard.py      Grafana dashboard provisioning
api/
  server.py         FastAPI production floor API
main.py             Entry point
```
