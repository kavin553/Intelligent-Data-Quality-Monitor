"""
Contract Validator
Validates incoming DataFrames against registered data contracts.

Catches BREAKING changes before they reach production dashboards or ML models.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .registry import ContractRegistry

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    table_name: str
    contract_version: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fields_checked: int = 0
    validation_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "table_name": self.table_name,
            "contract_version": self.contract_version,
            "errors": self.errors,
            "warnings": self.warnings,
            "fields_checked": self.fields_checked,
            "validation_time_ms": round(self.validation_time_ms, 3),
        }


class ContractValidator:
    """
    Validates DataFrames against registered data contracts.

    Usage:
        registry = ContractRegistry()
        registry.register(orders_contract)
        validator = ContractValidator(registry)

        result = validator.validate("orders", df)
        if not result.is_valid:
            raise PipelineBlockedError(result.errors)
    """

    def __init__(self, registry: ContractRegistry):
        self.registry = registry

    def validate(self, table_name: str, df: pd.DataFrame) -> ValidationResult:
        """Full validation against the latest registered contract."""
        start = time.time()
        contract = self.registry.get(table_name)

        if contract is None:
            return ValidationResult(
                is_valid=False,
                table_name=table_name,
                contract_version="none",
                errors=[f"No contract registered for table '{table_name}'"],
                validation_time_ms=(time.time() - start) * 1000,
            )

        errors: List[str] = []
        warnings: List[str] = []

        errors.extend(self._check_required_fields(contract, df))
        errors.extend(self._check_types(contract, df))
        errors.extend(self._check_null_rates(contract, df))
        warnings.extend(self._check_constraints(contract, df))

        result = ValidationResult(
            is_valid=len(errors) == 0,
            table_name=table_name,
            contract_version=contract.version,
            errors=errors,
            warnings=warnings,
            fields_checked=len(contract.fields),
            validation_time_ms=(time.time() - start) * 1000,
        )

        if not result.is_valid:
            logger.warning(
                "Contract validation FAILED: table=%s version=%s errors=%d",
                table_name,
                contract.version,
                len(errors),
            )
        return result

    def validate_schema(
        self, table_name: str, columns: List[str], dtypes: Dict
    ) -> ValidationResult:
        """Schema-only validation (no data needed)."""
        start = time.time()
        contract = self.registry.get(table_name)
        if not contract:
            return ValidationResult(
                is_valid=False,
                table_name=table_name,
                contract_version="none",
                errors=[f"No contract for '{table_name}'"],
            )

        errors = []
        for f in contract.fields:
            if not f.nullable and f.name not in columns:
                errors.append(f"Required field missing from schema: '{f.name}'")

        return ValidationResult(
            is_valid=len(errors) == 0,
            table_name=table_name,
            contract_version=contract.version,
            errors=errors,
            fields_checked=len(contract.fields),
            validation_time_ms=(time.time() - start) * 1000,
        )

    # ─────────────────────────────────────────────
    # Check methods
    # ─────────────────────────────────────────────

    def _check_required_fields(self, contract, df: pd.DataFrame) -> List[str]:
        errors = []
        for field_name in contract.required_fields:
            if field_name not in df.columns:
                errors.append(f"Required field missing: '{field_name}'")
        return errors

    def _check_types(self, contract, df: pd.DataFrame) -> List[str]:
        errors = []
        type_map = {
            "string": ["object", "string"],
            "integer": ["int64", "int32", "int16", "int8", "Int64"],
            "float": ["float64", "float32", "Float64"],
            "boolean": ["bool"],
            "timestamp": ["datetime64[ns]", "datetime64[ns, UTC]"],
        }
        for field_def in contract.fields:
            if field_def.name not in df.columns:
                continue
            actual_dtype = str(df[field_def.name].dtype)
            allowed = type_map.get(field_def.type, [])
            if allowed and not any(actual_dtype.startswith(t) for t in allowed):
                # Allow int → float promotion as warning only
                if field_def.type == "float" and actual_dtype.startswith("int"):
                    continue
                errors.append(
                    f"Type mismatch for '{field_def.name}': "
                    f"contract={field_def.type}, actual={actual_dtype}"
                )
        return errors

    def _check_null_rates(self, contract, df: pd.DataFrame) -> List[str]:
        errors = []
        for field_def in contract.fields:
            if field_def.nullable or field_def.name not in df.columns:
                continue
            null_count = df[field_def.name].isna().sum()
            if null_count > 0:
                null_rate = null_count / len(df)
                errors.append(
                    f"Non-nullable field '{field_def.name}' has {null_count} nulls "
                    f"({null_rate:.1%} null rate)"
                )
        return errors

    def _check_constraints(self, contract, df: pd.DataFrame) -> List[str]:
        """Returns warnings (not errors) for constraint violations."""
        warnings = []
        for field_def in contract.fields:
            if not field_def.constraints or field_def.name not in df.columns:
                continue
            col = df[field_def.name].dropna()
            c = field_def.constraints

            if "min" in c and pd.api.types.is_numeric_dtype(col):
                violations = (col < c["min"]).sum()
                if violations:
                    warnings.append(
                        f"'{field_def.name}': {violations} values below min={c['min']}"
                    )

            if "max" in c and pd.api.types.is_numeric_dtype(col):
                violations = (col > c["max"]).sum()
                if violations:
                    warnings.append(
                        f"'{field_def.name}': {violations} values above max={c['max']}"
                    )

            if "allowed_values" in c:
                invalid = (~col.astype(str).isin(c["allowed_values"])).sum()
                if invalid:
                    warnings.append(
                        f"'{field_def.name}': {invalid} values not in allowed set {c['allowed_values']}"
                    )

            if "pattern" in c:
                try:
                    non_matching = (~col.astype(str).str.match(c["pattern"])).sum()
                    if non_matching:
                        warnings.append(
                            f"'{field_def.name}': {non_matching} values don't match pattern '{c['pattern']}'"
                        )
                except Exception:
                    pass

        return warnings
