"""
Remediation action types and result models.
Each action is idempotent, logged, and auditable.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List


class ActionType(Enum):
    QUARANTINE_PARTITION = "quarantine_partition"
    TRIGGER_UPSTREAM_RERUN = "trigger_upstream_rerun"
    ROLLBACK_TO_SNAPSHOT = "rollback_to_snapshot"
    BLOCK_DOWNSTREAM = "block_downstream"
    SCHEMA_ROLLBACK = "schema_rollback"
    PII_MASK_COLUMN = "pii_mask_column"
    SEND_SLACK_ALERT = "send_slack_alert"
    SEND_PAGERDUTY = "send_pagerduty"
    FLAG_FOR_REVIEW = "flag_for_review"


class ActionStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RemediationAction:
    action_type: ActionType
    table_name: str
    reason: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    executed_at: str = ""
    duration_ms: float = 0.0
    result_message: str = ""

    def to_dict(self) -> Dict:
        return {
            "action_type": self.action_type.value,
            "table_name": self.table_name,
            "reason": self.reason,
            "parameters": self.parameters,
            "status": self.status.value,
            "executed_at": self.executed_at,
            "duration_ms": self.duration_ms,
            "result_message": self.result_message,
        }


@dataclass
class RemediationResult:
    anomaly_type: str
    table_name: str
    actions_taken: List[RemediationAction]
    total_remediation_time_ms: float
    remediated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def actions_succeeded(self) -> int:
        return sum(1 for a in self.actions_taken if a.status == ActionStatus.EXECUTED)

    @property
    def actions_failed(self) -> int:
        return sum(1 for a in self.actions_taken if a.status == ActionStatus.FAILED)

    @property
    def overall_status(self) -> str:
        if self.actions_failed == 0:
            return "fully_remediated"
        if self.actions_succeeded > 0:
            return "partially_remediated"
        return "failed"

    def to_dict(self) -> Dict:
        return {
            "anomaly_type": self.anomaly_type,
            "table_name": self.table_name,
            "overall_status": self.overall_status,
            "actions_taken": [a.to_dict() for a in self.actions_taken],
            "actions_succeeded": self.actions_succeeded,
            "actions_failed": self.actions_failed,
            "total_remediation_time_ms": self.total_remediation_time_ms,
            "remediated_at": self.remediated_at,
        }
