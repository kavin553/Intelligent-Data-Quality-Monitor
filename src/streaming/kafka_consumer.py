"""
Real-time Kafka consumer for DataShield anomaly detection.

Architecture:
  raw.data.events → DataShieldKafkaConsumer → quality checks → datashield.alerts

Message format (JSON):
  {"table_name": "orders", "data": {"col": [v1, v2, ...]}, "timestamp": "2026-01-01T00:00:00Z"}
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    bootstrap_servers: str
    input_topic: str
    alert_topic: str
    group_id: str = "datashield-consumer"
    contamination: float = 0.1
    auto_offset_reset: str = "earliest"
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 30000


class DataShieldKafkaConsumer:
    """
    Event-driven anomaly detection consumer.

    Consumes data events from Kafka, runs statistical + ML quality checks,
    and publishes alerts back to a Kafka topic — no polling required.
    """

    def __init__(self, config: StreamingConfig):
        self.config = config
        self._running = False
        self._consumer = None
        self._producer = None
        self._baseline_metadata: Dict = {}

        # Metrics
        self.messages_processed: int = 0
        self.alerts_fired: int = 0
        self.errors: int = 0
        self._start_time: Optional[float] = None

        # Lazy imports so module loads without confluent-kafka installed
        self._schema_discovery = None
        self._ml_detector = None

    def _init_components(self):
        from quality_engine.schema import SchemaDiscovery
        from ml_features.ml_anomaly_detector import MLAnomalyDetector

        self._schema_discovery = SchemaDiscovery()
        self._ml_detector = MLAnomalyDetector(contamination=self.config.contamination)

    def _init_kafka(self):
        try:
            from confluent_kafka import Consumer, KafkaException

            self._consumer = Consumer(
                {
                    "bootstrap.servers": self.config.bootstrap_servers,
                    "group.id": self.config.group_id,
                    "auto.offset.reset": self.config.auto_offset_reset,
                    "enable.auto.commit": False,
                    "max.poll.interval.ms": self.config.max_poll_interval_ms,
                    "session.timeout.ms": self.config.session_timeout_ms,
                }
            )
            self._consumer.subscribe([self.config.input_topic])
            logger.info(
                "Kafka consumer subscribed to topic: %s", self.config.input_topic
            )
        except ImportError:
            raise RuntimeError(
                "confluent-kafka not installed. Run: pip install confluent-kafka"
            )

    def start(self):
        """Start consuming messages. Blocks until stop() is called."""
        self._init_components()
        self._init_kafka()

        from DataShield.src.streaming.kafka_producer import DataShieldKafkaProducer

        self._producer = DataShieldKafkaProducer(
            self.config.bootstrap_servers, self.config.alert_topic
        )

        self._running = True
        self._start_time = time.time()
        backoff = 1.0

        logger.info("DataShield Kafka consumer started")

        try:
            while self._running:
                try:
                    msg = self._consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        logger.error("Kafka error: %s", msg.error())
                        self.errors += 1
                        continue

                    self._process_message(msg)
                    self._consumer.commit(asynchronous=False)
                    backoff = 1.0  # reset backoff on success

                except Exception as e:
                    logger.error("Processing error (backoff %.1fs): %s", backoff, e)
                    self.errors += 1
                    time.sleep(min(backoff, 60.0))
                    backoff = min(backoff * 2, 60.0)

        finally:
            if self._consumer:
                self._consumer.close()
            if self._producer:
                self._producer.close()
            logger.info("Consumer stopped. Stats: %s", self.get_stats())

    def stop(self):
        """Signal graceful shutdown."""
        logger.info("Stopping DataShield Kafka consumer...")
        self._running = False

    def _process_message(self, msg):
        """Deserialize, quality-check, and alert on a single Kafka message."""
        raw = json.loads(msg.value().decode("utf-8"))
        table_name = raw.get("table_name", "unknown")
        data = raw.get("data", {})

        if not data:
            logger.warning("Empty data payload for table: %s", table_name)
            return

        df = pd.DataFrame(data)
        self.messages_processed += 1

        # Auto-discover baseline if first time seeing this table
        if table_name not in self._baseline_metadata:
            self._handle_new_table(table_name, df)
            return  # First batch is baseline — no alerts yet

        alerts = self._run_quality_checks(table_name, df)

        if alerts:
            self._publish_alerts(table_name, alerts)
            self.alerts_fired += len(alerts)
            logger.info(
                "Fired %d alerts for table %s (msg offset=%s)",
                len(alerts),
                table_name,
                msg.offset(),
            )

    def _handle_new_table(self, table_name: str, df: pd.DataFrame):
        """Register baseline metadata for a newly seen table."""
        metadata = self._schema_discovery.discover(df, table_name)
        self._baseline_metadata[table_name] = metadata
        logger.info(
            "Registered baseline for new table: %s (%d rows, %d cols)",
            table_name,
            len(df),
            len(df.columns),
        )

    def _run_quality_checks(self, table_name: str, df: pd.DataFrame) -> List[dict]:
        """Run statistical + ML anomaly detection. Returns list of alert dicts."""
        from quality_engine.anomaly_detector import AnomalyDetector

        baseline = self._baseline_metadata[table_name]
        alerts = []

        stat_alerts = AnomalyDetector(baseline).detect(df)
        alerts.extend(a.to_dict() for a in stat_alerts)

        ml_alerts = self._ml_detector.detect(df)
        alerts.extend(a.to_dict() for a in ml_alerts)

        return alerts

    def _publish_alerts(self, table_name: str, alerts: List[dict]):
        """Send detected alerts to the Kafka alert topic."""
        if self._producer:
            payload = {
                "table_name": table_name,
                "alert_count": len(alerts),
                "alerts": alerts,
                "detected_at": pd.Timestamp.utcnow().isoformat(),
            }
            self._producer.publish_alert(payload)

    def get_stats(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "messages_processed": self.messages_processed,
            "alerts_fired": self.alerts_fired,
            "errors": self.errors,
            "uptime_seconds": round(uptime, 1),
            "tables_tracked": len(self._baseline_metadata),
            "alerts_per_message": (
                round(self.alerts_fired / max(self.messages_processed, 1), 3)
            ),
        }
