"""
Prometheus metrics for manufacturing floor observability.
Tracks defect rates, throughput, inference latency, and machine health.
SDKs: prometheus-client
"""
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server, CollectorRegistry,
)

# Inspection metrics
PARTS_INSPECTED = Counter(
    "manufacturing_parts_inspected_total",
    "Total parts inspected",
    ["line_id", "camera_id"],
)
PARTS_FAILED = Counter(
    "manufacturing_parts_failed_total",
    "Total parts failed inspection",
    ["line_id", "verdict"],
)
DEFECTS_DETECTED = Counter(
    "manufacturing_defects_detected_total",
    "Total defects detected",
    ["line_id", "defect_type", "severity"],
)
DEFECT_RATE = Gauge(
    "manufacturing_defect_rate",
    "Current defect rate (rolling window)",
    ["line_id"],
)
THROUGHPUT_PPH = Gauge(
    "manufacturing_throughput_parts_per_hour",
    "Current production throughput",
    ["line_id"],
)
INSPECTION_LATENCY = Histogram(
    "manufacturing_inspection_latency_ms",
    "YOLO inference latency per frame",
    ["line_id", "model"],
    buckets=[5, 10, 20, 30, 50, 75, 100, 150, 200, 500],
)

# Machine health metrics
MACHINE_TEMPERATURE = Gauge(
    "manufacturing_machine_temperature_celsius",
    "Machine temperature",
    ["machine_id", "line_id"],
)
MACHINE_VIBRATION = Gauge(
    "manufacturing_machine_vibration_mm_s",
    "Machine vibration level",
    ["machine_id", "line_id"],
)
MACHINE_UPTIME = Gauge(
    "manufacturing_machine_uptime_pct",
    "Machine uptime percentage",
    ["machine_id", "line_id"],
)
ACTIVE_FAULTS = Gauge(
    "manufacturing_active_faults",
    "Number of active machine faults",
    ["line_id"],
)

# Agent metrics
AGENT_ACTIONS = Counter(
    "manufacturing_agent_actions_total",
    "Agent actions taken",
    ["line_id", "action_type"],
)
ESCALATIONS = Counter(
    "manufacturing_escalations_total",
    "Total escalations triggered",
    ["line_id", "severity"],
)


class ProductionMetricsCollector:
    """Unified metrics interface for the production floor."""

    def __init__(self, line_id: str, port: int = 9090):
        self.line_id = line_id
        self.port = port
        self._server_started = False

    def start_server(self):
        if not self._server_started:
            start_http_server(self.port)
            self._server_started = True
            print(f"[Prometheus] Metrics server on :{self.port}/metrics")

    def record_inspection(self, result):
        PARTS_INSPECTED.labels(line_id=self.line_id, camera_id=result.camera_id).inc()
        if result.verdict.value in ("FAIL", "QUARANTINE"):
            PARTS_FAILED.labels(line_id=self.line_id, verdict=result.verdict.value).inc()
        for defect in result.defects:
            DEFECTS_DETECTED.labels(
                line_id=self.line_id,
                defect_type=defect.defect_type.value,
                severity=defect.severity,
            ).inc()
        INSPECTION_LATENCY.labels(line_id=self.line_id, model="yolo").observe(result.inference_ms)

    def update_defect_rate(self, rate: float):
        DEFECT_RATE.labels(line_id=self.line_id).set(rate)

    def update_throughput(self, pph: int):
        THROUGHPUT_PPH.labels(line_id=self.line_id).set(pph)

    def update_machine(self, machine_id: str, temp: float, vibration: float, uptime: float):
        MACHINE_TEMPERATURE.labels(machine_id=machine_id, line_id=self.line_id).set(temp)
        MACHINE_VIBRATION.labels(machine_id=machine_id, line_id=self.line_id).set(vibration)
        MACHINE_UPTIME.labels(machine_id=machine_id, line_id=self.line_id).set(uptime)

    def record_agent_action(self, action_type: str):
        AGENT_ACTIONS.labels(line_id=self.line_id, action_type=action_type).inc()

    def record_escalation(self, severity: str = "high"):
        ESCALATIONS.labels(line_id=self.line_id, severity=severity).inc()
