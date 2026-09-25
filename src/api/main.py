"""
DataShield REST API v0.3.0
Real-time data observability platform.

New in v0.3.0:
  - OpenTelemetry distributed tracing (Jaeger)
  - Self-healing pipeline remediation endpoints
  - Data contract validation endpoints
  - GNN cascade prediction endpoint
  - Kafka streaming status endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quality_engine.schema import SchemaDiscovery
from quality_engine.anomaly_detector import AnomalyDetector
from lineage.database import LineageDB
from lineage.blast_radius import BlastRadiusCalculator
from ml_features.ml_anomaly_detector import MLAnomalyDetector
from remediation.engine import RemediationEngine
from contracts.registry import ContractRegistry, DataContract, FieldContract
from contracts.validator import ContractValidator
from gnn.cascade_predictor import GNNCascadePredictor

try:
    from observability.tracing import (
        setup_tracing,
        trace_span,
        record_anomaly_metric,
        record_blast_radius_metric,
    )
    setup_tracing("datashield", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"))
    TRACING_ENABLED = True
except Exception:
    TRACING_ENABLED = False

    class _Noop:
        def __call__(self, *a, **kw):
            return self
        def __enter__(self): return self
        def __exit__(self, *a): pass

    trace_span = _Noop()

    def record_anomaly_metric(*a, **kw): pass
    def record_blast_radius_metric(*a, **kw): pass


app = FastAPI(
    title="DataShield API",
    description="Real-time data observability — anomaly detection, lineage, blast radius, self-healing",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────

lineage_db: Optional[LineageDB] = None
discovery = SchemaDiscovery()
baseline_metadata: dict = {}
ml_detector = MLAnomalyDetector(contamination=0.1)
remediation_engine = RemediationEngine(dry_run=False)
contract_registry = ContractRegistry()
contract_validator = ContractValidator(contract_registry)
gnn_predictor = GNNCascadePredictor()

# ── Request / Response Models ─────────────────────────────────

class SchemaDiscoveryRequest(BaseModel):
    table_name: str
    data: dict

class SchemaDiscoveryResponse(BaseModel):
    table_name: str
    row_count: int
    column_count: int
    columns: dict
    discovered_at: str

class AnomalyDetectionRequest(BaseModel):
    table_name: str
    data: dict

class AnomalyDetectionResponse(BaseModel):
    table_name: str
    alerts: List[dict]
    total_alerts: int
    detection_time_ms: float

class BlastRadiusRequest(BaseModel):
    source_table_id: int
    max_depth: int = 10

class BlastRadiusResponse(BaseModel):
    source_table_name: str
    total_affected: int
    critical_affected: int
    affected_tables: List[dict]
    escalation_channels: dict
    computation_time_ms: float

class ContractRequest(BaseModel):
    table_name: str
    version: str
    producer: str
    consumers: List[str]
    fields: List[dict]
    description: str = ""

class ContractValidationRequest(BaseModel):
    table_name: str
    data: dict

# ── Health ─────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.3.0",
        "components": {
            "quality_engine": "ready",
            "lineage_graph": "ready" if lineage_db else "not_initialized",
            "ml_detector": "ready",
            "remediation_engine": "ready",
            "contract_registry": f"{len(contract_registry.list_subjects())} contracts registered",
            "gnn_predictor": "trained" if gnn_predictor.is_trained else "untrained",
            "tracing": "enabled" if TRACING_ENABLED else "disabled",
        },
    }

# ── Quality Engine ─────────────────────────────────────────────

@app.post("/api/quality/discover", response_model=SchemaDiscoveryResponse)
async def discover_schema(request: SchemaDiscoveryRequest):
    with trace_span("quality.discover", {"table": request.table_name}):
        try:
            df = pd.DataFrame(request.data)
            metadata = discovery.discover(df, request.table_name)
            baseline_metadata[request.table_name] = metadata
            return SchemaDiscoveryResponse(
                table_name=metadata.table_name,
                row_count=metadata.row_count,
                column_count=metadata.column_count,
                columns={k: v.to_dict() for k, v in metadata.columns.items()},
                discovered_at=metadata.discovered_at,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/quality/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    with trace_span("quality.detect", {"table": request.table_name}) as span:
        if request.table_name not in baseline_metadata:
            raise HTTPException(status_code=400, detail=f"No baseline for '{request.table_name}'. Call /api/quality/discover first.")
        try:
            df = pd.DataFrame(request.data)
            baseline = baseline_metadata[request.table_name]
            detector = AnomalyDetector(baseline)
            start = time.time()
            alerts = detector.detect(df)
            elapsed = (time.time() - start) * 1000
            record_anomaly_metric(request.table_name, len(alerts), elapsed)
            return AnomalyDetectionResponse(
                table_name=request.table_name,
                alerts=[a.to_dict() for a in alerts],
                total_alerts=len(alerts),
                detection_time_ms=elapsed,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

# ── ML Detection ───────────────────────────────────────────────

@app.post("/api/ml/detect")
async def detect_ml_anomalies(request: AnomalyDetectionRequest):
    with trace_span("ml.detect", {"table": request.table_name}):
        try:
            df = pd.DataFrame(request.data)
            start = time.time()
            ml_alerts = ml_detector.detect(df)
            elapsed = (time.time() - start) * 1000
            record_anomaly_metric(request.table_name, len(ml_alerts), elapsed)
            return {
                "table_name": request.table_name,
                "detection_method": "ML (Isolation Forest + LOF + Temporal + Multivariate)",
                "ml_alerts": [a.to_dict() for a in ml_alerts],
                "total_alerts": len(ml_alerts),
                "detection_time_ms": elapsed,
                "alert_breakdown": {
                    "isolation_forest": len([a for a in ml_alerts if a.anomaly_type.value == "isolation_forest"]),
                    "local_outlier_factor": len([a for a in ml_alerts if a.anomaly_type.value == "local_outlier_factor"]),
                    "temporal_pattern": len([a for a in ml_alerts if a.anomaly_type.value == "temporal_pattern"]),
                    "multivariate": len([a for a in ml_alerts if a.anomaly_type.value == "multivariate"]),
                },
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/ml/compare")
async def compare_ml_vs_statistical(request: AnomalyDetectionRequest):
    if request.table_name not in baseline_metadata:
        raise HTTPException(status_code=400, detail="No baseline. Call /api/quality/discover first.")
    try:
        df = pd.DataFrame(request.data)
        baseline = baseline_metadata[request.table_name]
        stat_start = time.time()
        stat_alerts = AnomalyDetector(baseline).detect(df)
        stat_time = (time.time() - stat_start) * 1000
        ml_start = time.time()
        ml_alerts = ml_detector.detect(df)
        ml_time = (time.time() - ml_start) * 1000
        comparison = ml_detector.compare_with_statistical(len(stat_alerts), len(ml_alerts))
        return {
            "table_name": request.table_name,
            "statistical_detection": {"alerts": len(stat_alerts), "time_ms": stat_time, "details": [a.to_dict() for a in stat_alerts]},
            "ml_detection": {"alerts": len(ml_alerts), "time_ms": ml_time, "details": [a.to_dict() for a in ml_alerts]},
            "comparison": comparison,
            "recommendation": "Use ML detection for unknown patterns; Statistical for strict threshold rules",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── Lineage Graph ──────────────────────────────────────────────

@app.post("/api/lineage/initialize")
async def initialize_lineage():
    global lineage_db
    lineage_db = LineageDB()
    return {"status": "initialized", "message": "Lineage graph created"}

@app.post("/api/lineage/add-table")
async def add_table(
    table_name: str, table_type: str, owner: str,
    owner_email: str, criticality: str, refresh_frequency: str
):
    if lineage_db is None:
        raise HTTPException(status_code=400, detail="Lineage not initialized")
    try:
        table_id = lineage_db.add_table(table_name, table_type, owner, owner_email, criticality, refresh_frequency)
        return {"table_id": table_id, "table_name": table_name, "status": "added"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/lineage/add-dependency")
async def add_dependency(
    upstream_table_id: int, downstream_table_id: int,
    latency_minutes: int, dependency_type: str = "direct"
):
    if lineage_db is None:
        raise HTTPException(status_code=400, detail="Lineage not initialized")
    try:
        lineage_db.add_dependency(upstream_table_id, downstream_table_id, latency_minutes, dependency_type)
        return {"status": "added", "upstream_id": upstream_table_id, "downstream_id": downstream_table_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/lineage/blast-radius", response_model=BlastRadiusResponse)
async def calculate_blast_radius(request: BlastRadiusRequest):
    if lineage_db is None:
        raise HTTPException(status_code=400, detail="Lineage not initialized")
    with trace_span("lineage.blast_radius"):
        try:
            calculator = BlastRadiusCalculator(lineage_db)
            report = calculator.calculate(request.source_table_id, max_depth=request.max_depth)
            record_blast_radius_metric(report.source_table_name, report.total_affected, report.computation_time_ms)
            return BlastRadiusResponse(
                source_table_name=report.source_table_name,
                total_affected=report.total_affected,
                critical_affected=report.critical_affected,
                affected_tables=[t.to_dict() for t in report.affected_tables],
                escalation_channels=report.escalation_channels,
                computation_time_ms=report.computation_time_ms,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

# ── Self-Healing Remediation ───────────────────────────────────

@app.post("/api/remediation/remediate")
async def remediate_anomaly(request: AnomalyDetectionRequest):
    """Detect anomalies AND automatically remediate them."""
    if request.table_name not in baseline_metadata:
        raise HTTPException(status_code=400, detail="No baseline. Call /api/quality/discover first.")
    try:
        df = pd.DataFrame(request.data)
        baseline = baseline_metadata[request.table_name]
        alerts = AnomalyDetector(baseline).detect(df)

        if not alerts:
            return {"table_name": request.table_name, "message": "No anomalies detected — no remediation needed", "remediations": []}

        results = remediation_engine.remediate_batch(alerts)
        return {
            "table_name": request.table_name,
            "anomalies_detected": len(alerts),
            "remediations": [r.to_dict() for r in results],
            "engine_stats": remediation_engine.get_stats(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/remediation/history")
async def remediation_history():
    return {
        "history": [r.to_dict() for r in remediation_engine.get_remediation_history()],
        "stats": remediation_engine.get_stats(),
    }

# ── Data Contracts ─────────────────────────────────────────────

@app.post("/api/contracts/register")
async def register_contract(request: ContractRequest):
    """Register a data contract for a table."""
    try:
        fields = [
            FieldContract(
                name=f["name"],
                type=f["type"],
                nullable=f.get("nullable", True),
                description=f.get("description", ""),
                constraints=f.get("constraints", {}),
            )
            for f in request.fields
        ]
        contract = DataContract(
            table_name=request.table_name,
            version=request.version,
            producer=request.producer,
            consumers=request.consumers,
            fields=fields,
            description=request.description,
        )
        contract_id = contract_registry.register(contract)
        return {"contract_id": contract_id, "table_name": request.table_name, "version": request.version, "status": "registered"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/contracts/validate")
async def validate_contract(request: ContractValidationRequest):
    """Validate data against the registered contract for a table."""
    try:
        df = pd.DataFrame(request.data)
        result = contract_validator.validate(request.table_name, df)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/contracts/list")
async def list_contracts():
    subjects = contract_registry.list_subjects()
    return {
        "count": len(subjects),
        "subjects": {s: contract_registry.get_versions(s) for s in subjects},
    }

@app.get("/api/contracts/{table_name}")
async def get_contract(table_name: str, version: str = "latest"):
    contract = contract_registry.get(table_name, version)
    if not contract:
        raise HTTPException(status_code=404, detail=f"No contract found for '{table_name}' v{version}")
    return contract.to_dict()

# ── GNN Cascade Prediction ─────────────────────────────────────

@app.post("/api/gnn/train")
async def train_gnn(n_incidents: int = 5000, epochs: int = 100):
    """Train the GNN on the current lineage graph."""
    if lineage_db is None or len(lineage_db.tables) < 2:
        raise HTTPException(status_code=400, detail="Initialize lineage graph with at least 2 tables first.")
    try:
        metrics = gnn_predictor.train_on_lineage_graph(lineage_db, n_synthetic_incidents=n_incidents, epochs=epochs)
        return {"status": "trained", "model_info": gnn_predictor.get_model_info(), "training_metrics": metrics.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/gnn/predict")
async def gnn_predict(source_table_id: int):
    """Predict cascade failure probabilities using the trained GNN."""
    if not gnn_predictor.is_trained:
        raise HTTPException(status_code=400, detail="GNN not trained. Call POST /api/gnn/train first.")
    if lineage_db is None:
        raise HTTPException(status_code=400, detail="Lineage not initialized.")
    try:
        prediction = gnn_predictor.predict_cascade(source_table_id, lineage_db)
        return prediction.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/gnn/compare")
async def gnn_compare(source_table_id: int):
    """Compare GNN predictions vs exponential decay heuristic."""
    if not gnn_predictor.is_trained:
        raise HTTPException(status_code=400, detail="GNN not trained.")
    if lineage_db is None:
        raise HTTPException(status_code=400, detail="Lineage not initialized.")
    try:
        result = gnn_predictor.predict_vs_heuristic(source_table_id, lineage_db)
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── Root ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "DataShield API",
        "version": "0.3.0",
        "endpoints": {
            "health": "GET /health",
            "quality_discover": "POST /api/quality/discover",
            "quality_detect": "POST /api/quality/detect",
            "ml_detect": "POST /api/ml/detect",
            "ml_compare": "POST /api/ml/compare",
            "lineage_initialize": "POST /api/lineage/initialize",
            "lineage_add_table": "POST /api/lineage/add-table",
            "lineage_add_dependency": "POST /api/lineage/add-dependency",
            "lineage_blast_radius": "POST /api/lineage/blast-radius",
            "remediation_auto": "POST /api/remediation/remediate",
            "remediation_history": "GET /api/remediation/history",
            "contracts_register": "POST /api/contracts/register",
            "contracts_validate": "POST /api/contracts/validate",
            "contracts_list": "GET /api/contracts/list",
            "contracts_get": "GET /api/contracts/{table_name}",
            "gnn_train": "POST /api/gnn/train",
            "gnn_predict": "POST /api/gnn/predict",
            "gnn_compare": "POST /api/gnn/compare",
        },
        "docs": "http://localhost:8000/docs",
        "new_in_v0.3": [
            "OpenTelemetry distributed tracing → Jaeger",
            "Self-healing pipeline remediation (POST /api/remediation/remediate)",
            "Data contract validation (POST /api/contracts/validate)",
            "GNN cascade prediction (POST /api/gnn/predict)",
            "Kafka streaming consumer (src/streaming/)",
            "Kubernetes Helm chart (helm/datashield/)",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
