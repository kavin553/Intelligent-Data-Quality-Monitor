"""
DataShield Kafka alert producer.
Publishes anomaly alerts and blast radius reports to Kafka topics.
"""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DataShieldKafkaProducer:
    """Wraps confluent-kafka Producer for structured alert publishing."""

    def __init__(self, bootstrap_servers: str, alert_topic: str):
        self.alert_topic = alert_topic
        self._producer = None
        self._init_producer(bootstrap_servers)

    def _init_producer(self, bootstrap_servers: str):
        try:
            from confluent_kafka import Producer

            self._producer = Producer(
                {
                    "bootstrap.servers": bootstrap_servers,
                    "acks": "all",
                    "retries": 3,
                    "retry.backoff.ms": 500,
                }
            )
        except ImportError:
            raise RuntimeError(
                "confluent-kafka not installed. Run: pip install confluent-kafka"
            )

    def _delivery_callback(self, err, msg):
        if err:
            logger.error(
                "Alert delivery failed: %s (topic=%s, partition=%s)",
                err,
                msg.topic(),
                msg.partition(),
            )
        else:
            logger.debug(
                "Alert delivered to %s [%s] @ offset %s",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )

    def publish_alert(self, alert_dict: dict):
        """Publish an anomaly alert. Uses table_name as partition key."""
        if not self._producer:
            return
        key = alert_dict.get("table_name", "unknown").encode("utf-8")
        value = json.dumps(alert_dict, default=str).encode("utf-8")
        self._producer.produce(
            self.alert_topic, key=key, value=value, callback=self._delivery_callback
        )
        self._producer.poll(0)

    def publish_blast_radius(self, report_dict: dict):
        """Publish a blast radius report to the alert topic."""
        if not self._producer:
            return
        enriched = {**report_dict, "event_type": "blast_radius", "published_at": datetime.now(timezone.utc).isoformat()}
        key = report_dict.get("source_table_name", "unknown").encode("utf-8")
        value = json.dumps(enriched, default=str).encode("utf-8")
        self._producer.produce(
            self.alert_topic, key=key, value=value, callback=self._delivery_callback
        )
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0):
        """Wait for all pending messages to be delivered."""
        if self._producer:
            remaining = self._producer.flush(timeout=timeout)
            if remaining > 0:
                logger.warning("%d messages were not delivered within %.1fs", remaining, timeout)

    def close(self):
        self.flush()
        logger.info("Kafka producer closed")
