"""
Data Contract Registry

Stores and manages data contracts between producers and consumers.
A contract defines what schema a producer MUST provide and consumers EXPECT.

Breaking change detection prevents schema regressions from reaching production.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class FieldContract:
    name: str
    type: str  # "string", "integer", "float", "boolean", "timestamp"
    nullable: bool = False
    description: str = ""
    constraints: Dict = field(default_factory=dict)  # min, max, allowed_values, pattern

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
            "description": self.description,
            "constraints": self.constraints,
        }


@dataclass
class DataContract:
    table_name: str
    version: str
    producer: str
    consumers: List[str]
    fields: List[FieldContract]
    description: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "active"  # "active", "deprecated", "draft"

    def to_dict(self) -> Dict:
        return {
            "table_name": self.table_name,
            "version": self.version,
            "producer": self.producer,
            "consumers": self.consumers,
            "fields": [f.to_dict() for f in self.fields],
            "description": self.description,
            "created_at": self.created_at,
            "status": self.status,
        }

    def get_field(self, name: str) -> Optional[FieldContract]:
        return next((f for f in self.fields if f.name == name), None)

    @property
    def required_fields(self) -> List[str]:
        return [f.name for f in self.fields if not f.nullable]


@dataclass
class CompatibilityResult:
    compatible: bool
    compatibility_type: str  # "FULL", "BACKWARD", "FORWARD", "BREAKING"
    breaking_changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "compatible": self.compatible,
            "compatibility_type": self.compatibility_type,
            "breaking_changes": self.breaking_changes,
            "warnings": self.warnings,
        }


class ContractRegistry:
    """
    In-memory contract registry with versioning and compatibility checking.

    In production, back this with PostgreSQL or Confluent Schema Registry.
    """

    def __init__(self):
        # table_name → list of DataContract (ordered oldest → newest)
        self._contracts: Dict[str, List[DataContract]] = {}

    def register(self, contract: DataContract) -> str:
        """Register a new contract version. Returns contract_id."""
        table = contract.table_name
        if table not in self._contracts:
            self._contracts[table] = []
        self._contracts[table].append(contract)
        contract_id = f"{table}:v{contract.version}"
        return contract_id

    def get(self, table_name: str, version: str = "latest") -> Optional[DataContract]:
        versions = self._contracts.get(table_name, [])
        if not versions:
            return None
        if version == "latest":
            return versions[-1]
        return next((c for c in versions if c.version == version), None)

    def list_subjects(self) -> List[str]:
        return list(self._contracts.keys())

    def get_versions(self, table_name: str) -> List[str]:
        return [c.version for c in self._contracts.get(table_name, [])]

    def deprecate(self, table_name: str, version: str) -> bool:
        contract = self.get(table_name, version)
        if contract:
            contract.status = "deprecated"
            return True
        return False

    def check_compatibility(
        self, table_name: str, new_contract: DataContract
    ) -> CompatibilityResult:
        """
        Check if new_contract is compatible with the existing latest contract.

        FULL       — backward + forward compatible (additive optional fields only)
        BACKWARD   — new consumers can read old data (fields added as nullable)
        FORWARD    — old consumers can read new data (fields removed were optional)
        BREAKING   — field removed, type changed, required field added
        """
        existing = self.get(table_name)
        if not existing:
            return CompatibilityResult(True, "FULL")

        breaking = []
        warnings = []

        existing_fields = {f.name: f for f in existing.fields}
        new_fields = {f.name: f for f in new_contract.fields}

        # Check for removed fields
        for fname, fdef in existing_fields.items():
            if fname not in new_fields:
                if not fdef.nullable:
                    breaking.append(f"Required field removed: '{fname}'")
                else:
                    warnings.append(f"Optional field removed: '{fname}' (forward-incompatible)")

        # Check for type changes
        for fname, new_fdef in new_fields.items():
            if fname in existing_fields:
                old_type = existing_fields[fname].type
                if old_type != new_fdef.type:
                    breaking.append(
                        f"Type changed for '{fname}': {old_type} → {new_fdef.type}"
                    )

        # Check for new required fields
        for fname, new_fdef in new_fields.items():
            if fname not in existing_fields and not new_fdef.nullable:
                breaking.append(
                    f"New required field added: '{fname}' (existing data won't have it)"
                )
            elif fname not in existing_fields and new_fdef.nullable:
                warnings.append(f"New optional field added: '{fname}' (backward compatible)")

        if breaking:
            return CompatibilityResult(False, "BREAKING", breaking, warnings)
        if warnings:
            return CompatibilityResult(True, "BACKWARD", [], warnings)
        return CompatibilityResult(True, "FULL")
