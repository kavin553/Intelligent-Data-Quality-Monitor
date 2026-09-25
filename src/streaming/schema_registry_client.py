"""
Lightweight Schema Registry client.
Compatible with Confluent Schema Registry REST API.
No paid dependency required.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    subject: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "subject": self.subject,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class SchemaRegistryClient:
    """
    HTTP client for a Confluent-compatible Schema Registry.
    Falls back to in-memory storage when registry is unavailable.
    """

    def __init__(self, registry_url: str = "http://localhost:8081"):
        self.registry_url = registry_url.rstrip("/")
        self._local_cache: Dict[str, List[Dict]] = {}  # subject → [schemas]
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            import urllib.request

            urllib.request.urlopen(f"{self.registry_url}/subjects", timeout=2)
            return True
        except Exception:
            logger.warning(
                "Schema Registry at %s unreachable — using local cache",
                self.registry_url,
            )
            return False

    def _http_post(self, path: str, body: dict) -> Optional[dict]:
        try:
            import urllib.request, urllib.error

            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                f"{self.registry_url}{path}",
                data=data,
                headers={"Content-Type": "application/vnd.schemaregistry.v1+json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.error("Schema Registry POST failed: %s", e)
            return None

    def _http_get(self, path: str) -> Optional[dict]:
        try:
            import urllib.request

            with urllib.request.urlopen(
                f"{self.registry_url}{path}", timeout=5
            ) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.error("Schema Registry GET failed: %s", e)
            return None

    def register_schema(self, subject: str, schema: dict) -> int:
        """Register a schema. Returns schema ID."""
        if self._available:
            result = self._http_post(
                f"/subjects/{subject}/versions",
                {"schema": json.dumps(schema)},
            )
            if result:
                return result.get("id", -1)

        # Local fallback
        if subject not in self._local_cache:
            self._local_cache[subject] = []
        self._local_cache[subject].append(schema)
        schema_id = len(self._local_cache[subject])
        logger.info("Registered schema locally: subject=%s id=%d", subject, schema_id)
        return schema_id

    def get_schema(self, subject: str, version: str = "latest") -> Optional[dict]:
        """Get schema by subject and version."""
        if self._available:
            result = self._http_get(f"/subjects/{subject}/versions/{version}")
            if result:
                return json.loads(result.get("schema", "{}"))

        # Local fallback
        versions = self._local_cache.get(subject, [])
        if not versions:
            return None
        return versions[-1] if version == "latest" else (
            versions[int(version) - 1] if int(version) <= len(versions) else None
        )

    def validate_schema(self, subject: str, data: dict) -> ValidationResult:
        """Validate data against the registered schema for a subject."""
        schema = self.get_schema(subject)
        if not schema:
            return ValidationResult(
                is_valid=False,
                subject=subject,
                errors=[f"No schema registered for subject: {subject}"],
            )

        errors = []
        warnings = []
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        for field_name in required_fields:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")

        for field_name, field_schema in properties.items():
            if field_name not in data:
                if field_name not in required_fields:
                    warnings.append(f"Optional field absent: {field_name}")
                continue

            value = data[field_name]
            expected_type = field_schema.get("type")
            if expected_type and not self._check_type(value, expected_type):
                errors.append(
                    f"Type mismatch for '{field_name}': expected {expected_type}, got {type(value).__name__}"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            subject=subject,
            errors=errors,
            warnings=warnings,
        )

    def list_subjects(self) -> List[str]:
        if self._available:
            result = self._http_get("/subjects")
            if result:
                return result
        return list(self._local_cache.keys())

    def _check_type(self, value: Any, expected_type: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected = type_map.get(expected_type)
        return isinstance(value, expected) if expected else True
