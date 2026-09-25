"""
Self-Healing Pipeline Engine

Automatically remediates data quality anomalies without human intervention.

Decision tree:
  NULL_EXPLOSION       → quarantine_partition + trigger_upstream_rerun
  ROW_COUNT_SPIKE      → flag_for_review + block_downstream (if critical)
  SCHEMA_DRIFT         → schema_rollback + send_slack_alert
  PII_EXPOSURE         → pii_mask_column + send_pagerduty  (CRITICAL)
  CARDINALITY_COLLAPSE → rollback_to_snapshot + send_slack_alert
  DISTRIBUTION_SHIFT   → flag_for_review + send_slack_alert (WARNING only)
  COST_ANOMALY         → flag_for_review
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from .actions import ActionStatus, ActionType, RemediationAction, RemediationResult

logger = logging.getLogger(__name__)


class RemediationEngine:
    """
    Orchestrates automatic remediation actions when anomalies are detected.

    In dry_run=True mode, all actions are logged but not executed — safe for testing.
    """

    def __init__(self, lineage_db=None, dry_run: bool = False):
        self.lineage_db = lineage_db
        self.dry_run = dry_run
        self._history: List[RemediationResult] = []
        self._quarantined_tables: Set[str] = set()
        self._blocked_tables: Set[str] = set()

        if dry_run:
            logger.info("RemediationEngine initialized in DRY RUN mode — no actions will execute")

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def remediate(self, anomaly) -> RemediationResult:
        """Select and execute remediation actions for one anomaly."""
        start = time.time()
        anomaly_type = anomaly.anomaly_type.value
        table_name = anomaly.table_name

        logger.info("Starting remediation: table=%s anomaly=%s", table_name, anomaly_type)

        actions = self._select_actions(anomaly)
        for action in actions:
            self._execute_action(action, anomaly)

        result = RemediationResult(
            anomaly_type=anomaly_type,
            table_name=table_name,
            actions_taken=actions,
            total_remediation_time_ms=(time.time() - start) * 1000,
        )

        self._history.append(result)
        logger.info(
            "Remediation complete: table=%s status=%s actions=%d",
            table_name,
            result.overall_status,
            len(actions),
        )
        return result

    def remediate_batch(self, anomalies: list) -> List[RemediationResult]:
        """Remediate a batch of anomalies. Deduplicates by table+type."""
        seen = set()
        results = []
        for anomaly in anomalies:
            key = (anomaly.table_name, anomaly.anomaly_type.value)
            if key not in seen:
                seen.add(key)
                results.append(self.remediate(anomaly))
        return results

    def get_remediation_history(self) -> List[RemediationResult]:
        return list(self._history)

    def get_stats(self) -> Dict:
        total = len(self._history)
        if total == 0:
            return {"total_remediations": 0, "success_rate": 0.0}

        succeeded = sum(1 for r in self._history if r.overall_status == "fully_remediated")
        action_counts: Dict[str, int] = {}
        for result in self._history:
            for action in result.actions_taken:
                t = action.action_type.value
                action_counts[t] = action_counts.get(t, 0) + 1

        return {
            "total_remediations": total,
            "success_rate": round(succeeded / total, 3),
            "quarantined_tables": len(self._quarantined_tables),
            "blocked_tables": len(self._blocked_tables),
            "actions_by_type": action_counts,
        }

    # ─────────────────────────────────────────────
    # Decision Tree
    # ─────────────────────────────────────────────

    def _select_actions(self, anomaly) -> List[RemediationAction]:
        anomaly_type = anomaly.anomaly_type.value
        table = anomaly.table_name
        severity = anomaly.severity.value
        col = getattr(anomaly, "column_name", None)

        actions = []

        if anomaly_type == "null_rate_explosion":
            actions.append(RemediationAction(
                ActionType.QUARANTINE_PARTITION, table,
                f"Null rate explosion detected in {col or 'table'}",
                parameters={"column": col, "partition": "latest"},
            ))
            actions.append(RemediationAction(
                ActionType.TRIGGER_UPSTREAM_RERUN, table,
                "Upstream rerun needed to repopulate nulls",
            ))

        elif anomaly_type == "row_count_spike":
            actions.append(RemediationAction(
                ActionType.FLAG_FOR_REVIEW, table,
                f"Row count spike: deviation {anomaly.deviation_percent:.1f}%",
                parameters={"deviation_percent": anomaly.deviation_percent},
            ))
            if severity == "critical":
                actions.append(RemediationAction(
                    ActionType.BLOCK_DOWNSTREAM, table,
                    "Critical row count spike — blocking downstream to prevent propagation",
                ))

        elif anomaly_type == "schema_drift":
            actions.append(RemediationAction(
                ActionType.SCHEMA_ROLLBACK, table,
                "Schema drift detected — rolling back to last known-good version",
                parameters={"drift_details": anomaly.details},
            ))
            actions.append(RemediationAction(
                ActionType.SEND_SLACK_ALERT, table,
                "Schema changed unexpectedly",
                parameters={"channel": "#data-alerts", "severity": severity},
            ))

        elif anomaly_type == "pii_exposure":
            actions.append(RemediationAction(
                ActionType.PII_MASK_COLUMN, table,
                f"PII detected in column: {col}",
                parameters={"column": col, "pattern": anomaly.details.get("pattern_matched")},
            ))
            actions.append(RemediationAction(
                ActionType.SEND_PAGERDUTY, table,
                "CRITICAL: PII exposure detected — immediate masking required",
                parameters={"severity": "critical", "column": col},
            ))

        elif anomaly_type == "cardinality_collapse":
            actions.append(RemediationAction(
                ActionType.ROLLBACK_TO_SNAPSHOT, table,
                f"Cardinality in {col} collapsed — rolling back to T-1 snapshot",
                parameters={"column": col, "snapshot": "T-1"},
            ))
            actions.append(RemediationAction(
                ActionType.SEND_SLACK_ALERT, table,
                f"Cardinality collapse in {col}",
                parameters={"channel": "#data-alerts"},
            ))

        elif anomaly_type == "distribution_shift":
            actions.append(RemediationAction(
                ActionType.FLAG_FOR_REVIEW, table,
                f"Distribution shift in {col} — flagged for manual review",
                parameters={"column": col, "z_score": anomaly.details.get("z_score")},
            ))
            if severity in ("critical", "warning"):
                actions.append(RemediationAction(
                    ActionType.SEND_SLACK_ALERT, table,
                    f"Distribution shift detected in column {col}",
                    parameters={"channel": "#data-alerts"},
                ))

        else:
            actions.append(RemediationAction(
                ActionType.FLAG_FOR_REVIEW, table,
                f"Anomaly type {anomaly_type} — flagged for review",
            ))

        return actions

    # ─────────────────────────────────────────────
    # Action Executors
    # ─────────────────────────────────────────────

    def _execute_action(self, action: RemediationAction, anomaly=None):
        if self.dry_run:
            action.status = ActionStatus.SKIPPED
            action.result_message = "DRY RUN — action not executed"
            logger.info("[DRY RUN] Would execute: %s on %s", action.action_type.value, action.table_name)
            return

        start = time.time()
        try:
            handler = {
                ActionType.QUARANTINE_PARTITION: self._quarantine_partition,
                ActionType.TRIGGER_UPSTREAM_RERUN: self._trigger_upstream_rerun,
                ActionType.ROLLBACK_TO_SNAPSHOT: self._rollback_to_snapshot,
                ActionType.BLOCK_DOWNSTREAM: self._block_downstream,
                ActionType.SCHEMA_ROLLBACK: self._schema_rollback,
                ActionType.PII_MASK_COLUMN: self._pii_mask_column,
                ActionType.SEND_SLACK_ALERT: self._send_slack_alert,
                ActionType.SEND_PAGERDUTY: self._send_pagerduty,
                ActionType.FLAG_FOR_REVIEW: self._flag_for_review,
            }.get(action.action_type)

            if handler:
                msg = handler(action)
                action.status = ActionStatus.EXECUTED
                action.result_message = msg
            else:
                action.status = ActionStatus.SKIPPED
                action.result_message = "No handler registered"

        except Exception as e:
            action.status = ActionStatus.FAILED
            action.result_message = str(e)
            logger.error("Action failed: %s — %s", action.action_type.value, e)

        action.executed_at = datetime.now(timezone.utc).isoformat()
        action.duration_ms = (time.time() - start) * 1000

    def _quarantine_partition(self, action: RemediationAction) -> str:
        self._quarantined_tables.add(action.table_name)
        partition = action.parameters.get("partition", "latest")
        logger.warning("QUARANTINE: table=%s partition=%s", action.table_name, partition)
        return f"Partition '{partition}' quarantined for table {action.table_name}"

    def _trigger_upstream_rerun(self, action: RemediationAction) -> str:
        logger.info("RERUN: Triggering Airflow DAG rerun for upstream of %s", action.table_name)
        # In production: call Airflow REST API POST /dags/{dag_id}/dagRuns
        return f"Airflow rerun triggered for upstream of {action.table_name} (simulated)"

    def _rollback_to_snapshot(self, action: RemediationAction) -> str:
        snapshot = action.parameters.get("snapshot", "T-1")
        logger.warning("ROLLBACK: table=%s to snapshot=%s", action.table_name, snapshot)
        # In production: call storage layer API to swap partition pointer
        return f"Table {action.table_name} rolled back to {snapshot} snapshot (simulated)"

    def _block_downstream(self, action: RemediationAction) -> str:
        self._blocked_tables.add(action.table_name)
        logger.warning("BLOCK: Marking downstream of %s as blocked", action.table_name)
        return f"Downstream jobs for {action.table_name} blocked"

    def _schema_rollback(self, action: RemediationAction) -> str:
        logger.warning("SCHEMA ROLLBACK: table=%s", action.table_name)
        # In production: call Schema Registry to revert to previous version
        return f"Schema for {action.table_name} rolled back to previous version (simulated)"

    def _pii_mask_column(self, action: RemediationAction) -> str:
        col = action.parameters.get("column", "unknown")
        pattern = action.parameters.get("pattern", "unknown")
        logger.critical("PII MASK: table=%s column=%s pattern=%s", action.table_name, col, pattern)
        # In production: apply column-level masking policy via data catalog API
        return f"Column {col} masked (pattern={pattern}) in table {action.table_name} (simulated)"

    def _send_slack_alert(self, action: RemediationAction) -> str:
        channel = action.parameters.get("channel", "#data-alerts")
        logger.info("SLACK: Sending alert to %s for table %s", channel, action.table_name)
        # In production: POST to Slack webhook URL
        return f"Slack alert sent to {channel}: {action.reason}"

    def _send_pagerduty(self, action: RemediationAction) -> str:
        severity = action.parameters.get("severity", "critical")
        logger.critical("PAGERDUTY: severity=%s table=%s", severity, action.table_name)
        # In production: POST to PagerDuty Events API v2
        return f"PagerDuty incident created: severity={severity} table={action.table_name} (simulated)"

    def _flag_for_review(self, action: RemediationAction) -> str:
        logger.info("FLAG: %s — %s", action.table_name, action.reason)
        return f"Table {action.table_name} flagged for manual review: {action.reason}"
