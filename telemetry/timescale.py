"""
TimescaleDB time-series telemetry storage for production floor data.
Hypertables for inspection results, machine metrics, and alerts.
SDKs: psycopg2, TimescaleDB, Polars
"""
import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from dataclasses import asdict

import psycopg2
import psycopg2.extras
import pandas as pd
import polars as pl


class ManufacturingTimeseriesDB:
    """
    TimescaleDB storage for manufacturing floor telemetry.
    Auto-partitioned hypertables for high-ingest rate data.
    """

    def __init__(self, conn_str: Optional[str] = None):
        self.conn_str = conn_str or os.environ.get(
            "TIMESCALE_URL",
            "postgresql://postgres:password@localhost:5432/manufacturing"
        )
        self.conn = psycopg2.connect(self.conn_str)
        self._setup_schema()
        print("[TimescaleDB] Manufacturing telemetry DB ready")

    def _setup_schema(self):
        with self.conn.cursor() as cur:
            # Inspection results hypertable
            cur.execute("""
                CREATE TABLE IF NOT EXISTS inspection_results (
                    time         TIMESTAMPTZ NOT NULL,
                    result_id    TEXT,
                    part_id      TEXT,
                    line_id      TEXT,
                    camera_id    TEXT,
                    verdict      TEXT,
                    n_defects    INTEGER,
                    has_critical BOOLEAN,
                    inference_ms FLOAT
                )
            """)
            cur.execute("""
                SELECT create_hypertable('inspection_results', 'time', if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 hour')
            """)

            # Machine telemetry hypertable
            cur.execute("""
                CREATE TABLE IF NOT EXISTS machine_telemetry (
                    time          TIMESTAMPTZ NOT NULL,
                    machine_id    TEXT,
                    line_id       TEXT,
                    metric_name   TEXT,
                    metric_value  DOUBLE PRECISION,
                    unit          TEXT
                )
            """)
            cur.execute("""
                SELECT create_hypertable('machine_telemetry', 'time', if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 hour')
            """)

            # Alerts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS production_alerts (
                    time       TIMESTAMPTZ NOT NULL,
                    alert_id   TEXT,
                    line_id    TEXT,
                    alert_type TEXT,
                    severity   TEXT,
                    message    TEXT,
                    resolved   BOOLEAN DEFAULT FALSE
                )
            """)
            cur.execute("""
                SELECT create_hypertable('production_alerts', 'time', if_not_exists => TRUE)
            """)

            # Continuous aggregate: 5-minute defect rate per line
            cur.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS defect_rate_5min
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('5 minutes', time) AS bucket,
                    line_id,
                    COUNT(*) AS total_parts,
                    SUM(CASE WHEN verdict = 'FAIL' THEN 1 ELSE 0 END) AS failed_parts,
                    AVG(inference_ms) AS avg_inference_ms
                FROM inspection_results
                GROUP BY bucket, line_id
                WITH NO DATA
            """)
            self.conn.commit()

    def insert_inspection(self, result) -> bool:
        """Insert an InspectionResult into TimescaleDB."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO inspection_results
                    (time, result_id, part_id, line_id, camera_id, verdict, n_defects, has_critical, inference_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.fromtimestamp(result.timestamp, tz=timezone.utc),
                    result.result_id, result.part_id, result.line_id,
                    result.camera_id, result.verdict.value,
                    result.n_defects, result.has_critical, result.inference_ms,
                ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[TimescaleDB] Insert inspection failed: {e}")
            self.conn.rollback()
            return False

    def insert_telemetry(
        self, machine_id: str, line_id: str,
        metrics: Dict[str, float], unit: str = ""
    ):
        """Insert machine telemetry metrics."""
        now = datetime.now(tz=timezone.utc)
        rows = [
            (now, machine_id, line_id, k, v, unit)
            for k, v in metrics.items()
        ]
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO machine_telemetry (time, machine_id, line_id, metric_name, metric_value, unit) VALUES %s",
                rows,
            )
        self.conn.commit()

    def get_defect_rate(self, line_id: str, minutes: int = 30) -> float:
        """Calculate defect rate over the last N minutes."""
        since = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN verdict = 'FAIL' THEN 1 ELSE 0 END) AS failed
                FROM inspection_results
                WHERE line_id = %s AND time > %s
            """, (line_id, since))
            row = cur.fetchone()
        if row and row[0] > 0:
            return row[1] / row[0]
        return 0.0

    def get_throughput(self, line_id: str, minutes: int = 60) -> int:
        """Get parts inspected per hour."""
        since = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM inspection_results WHERE line_id = %s AND time > %s",
                (line_id, since)
            )
            count = cur.fetchone()[0]
        return int(count * 60 / minutes)

    def get_recent_failures(self, line_id: str, limit: int = 20) -> pl.DataFrame:
        """Get most recent failed inspections."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT time, part_id, verdict, n_defects, inference_ms
                FROM inspection_results
                WHERE line_id = %s AND verdict IN ('FAIL', 'QUARANTINE')
                ORDER BY time DESC
                LIMIT %s
            """, (line_id, limit))
            rows = cur.fetchall()
        return pl.from_dicts(list(rows)) if rows else pl.DataFrame()

    def get_machine_trend(self, machine_id: str, metric: str, hours: int = 4) -> pl.DataFrame:
        """Get a machine metric trend over last N hours."""
        since = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT time, metric_value
                FROM machine_telemetry
                WHERE machine_id = %s AND metric_name = %s AND time > %s
                ORDER BY time ASC
            """, (machine_id, metric, since))
            rows = cur.fetchall()
        return pl.from_dicts(list(rows)) if rows else pl.DataFrame()

    def close(self):
        self.conn.close()
