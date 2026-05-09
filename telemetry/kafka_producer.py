"""
Production event streaming to Kafka.
Every inspection result, machine reading, and agent action is an event.
SDKs: kafka-python, PyArrow
"""
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import pyarrow as pa
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable


TOPICS = {
    "inspection": "manufacturing.inspection.results",
    "telemetry": "manufacturing.machine.telemetry",
    "alerts": "manufacturing.alerts",
    "agent_actions": "manufacturing.agent.actions",
    "line_events": "manufacturing.line.events",
}

INSPECTION_SCHEMA = pa.schema([
    pa.field("result_id", pa.string()),
    pa.field("timestamp", pa.float64()),
    pa.field("part_id", pa.string()),
    pa.field("line_id", pa.string()),
    pa.field("verdict", pa.string()),
    pa.field("n_defects", pa.int32()),
    pa.field("has_critical", pa.bool_()),
    pa.field("inference_ms", pa.float32()),
])


class ManufacturingEventProducer:
    """
    Kafka producer for manufacturing floor events.
    Publishes inspection results, telemetry, alerts, and agent actions.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        line_id: str = "LINE_A",
    ):
        self.line_id = line_id
        self._fallback: List[dict] = []
        self.producer = None

        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode(),
                acks=1,
                linger_ms=5,
                batch_size=32768,
            )
            print(f"[Kafka] Producer connected: {bootstrap_servers}")
        except NoBrokersAvailable:
            print(f"[Kafka] No brokers at {bootstrap_servers}. Using fallback queue.")

    def emit_inspection(self, result) -> bool:
        """Publish an InspectionResult to Kafka."""
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        return self._emit(TOPICS["inspection"], payload)

    def emit_telemetry(
        self,
        machine_id: str,
        metrics: Dict[str, float],
        sensor_type: str = "general",
    ) -> bool:
        payload = {
            "machine_id": machine_id,
            "line_id": self.line_id,
            "sensor_type": sensor_type,
            "metrics": metrics,
            "timestamp": time.time(),
        }
        return self._emit(TOPICS["telemetry"], payload)

    def emit_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        machine_id: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> bool:
        payload = {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "machine_id": machine_id,
            "line_id": self.line_id,
            "details": details or {},
            "timestamp": time.time(),
        }
        return self._emit(TOPICS["alerts"], payload)

    def emit_agent_action(
        self,
        action_type: str,
        reasoning: str,
        parameters: Dict[str, Any],
    ) -> bool:
        payload = {
            "action_type": action_type,
            "reasoning": reasoning,
            "parameters": parameters,
            "line_id": self.line_id,
            "timestamp": time.time(),
        }
        return self._emit(TOPICS["agent_actions"], payload)

    def _emit(self, topic: str, payload: dict) -> bool:
        if self.producer:
            try:
                self.producer.send(topic, value=payload)
                return True
            except Exception as e:
                print(f"[Kafka] Emit error: {e}")
        self._fallback.append({"topic": topic, **payload})
        return False

    def flush(self):
        if self.producer:
            self.producer.flush()

    def close(self):
        if self.producer:
            self.producer.close()
