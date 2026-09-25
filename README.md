# DataShield

**Real-time data-observability platform: a data-quality engine, lineage / blast-radius tracking, and ML anomaly detection, exposed over a FastAPI service. Evaluated on a real, messy public dataset — not just synthetic data.**

Python 3.11+ · FastAPI · MIT licensed.

<p align="center">
  <img src="assets/hero.png" width="620"><br>
  <sub>What the quality engine actually finds in 541,909 real Online Retail transactions — not a synthetic frame it was handed.</sub>
</p>

DataShield watches tabular data flowing through a pipeline and answers three questions in one pass:

1. **Is this batch broken?** Statistical and ML detectors flag row-count spikes, null explosions, cardinality collapse, distribution shift, schema drift, and PII leakage.
2. **What else breaks if it is?** A lineage graph computes the downstream blast radius (which tables, dashboards, and models are affected) and the probability that the failure propagates to each.
3. **Who needs to know, and can we auto-fix it?** Severity-ranked escalation plus a remediation engine for the cases that have a safe automated response.

It runs three ways with increasing infrastructure: a **zero-dependency demo**, a **full FastAPI service** (Postgres-backed, with a graceful in-memory fallback), and a **streaming mode** (Kafka + Postgres via Docker Compose).

---

## Real data + honest evaluation

Most portfolio DQ projects only ever run on data they generated themselves, which
proves nothing — of course a detector finds the anomaly you just injected into a
clean synthetic frame. DataShield is evaluated on a **real, genuinely messy public
dataset**, and its detection accuracy is measured against **ground truth** instead
of asserted.

- **Base dataset is REAL:** the [UCI *Online Retail*](https://archive.ics.uci.edu/dataset/352/online+retail)
  dataset — 541,909 actual e-commerce transactions from a UK online retailer
  (2010–2011). It ships real mess: **24.93% of rows have no CustomerID**, 1.71% are
  cancelled invoices, 1.96% have negative quantities (returns), plus zero/negative
  prices, 5,268 exact duplicates, and extreme outlier amounts. The quality engine
  runs over all of it and reports what it actually finds.
- **Detection is evaluated the honest way:** you can't measure precision/recall
  without labels, and the real feed has none. So DataShield takes **clean slices of
  the real data and injects KNOWN, labelled faults** (null spike, distribution
  drift, a numeric→text type change, injected PII emails, cardinality collapse),
  then scores the detector against those labels. The base rows are real; the
  injected fault is the synthetic-with-ground-truth part — that's how you evaluate a
  data-quality system.

Reproduce it (no paid keys, no infra):

```bash
python scripts/download_data.py    # cache the real dataset under data/ (gitignored)
python run_demo.py                 # real-data profile + injected-fault precision/recall
```

Measured detection on the injected faults (real detector output vs. ground truth):

| Injected fault | Precision | Recall |
|---|---:|---:|
| null_spike · distribution_drift · schema_type_change · pii_injection · cardinality_collapse | 1.00 | 1.00 |
| any-fault vs. clean control batches | 1.00 | 1.00 |

Blunt caveat: these faults are large and unambiguous, so perfect scores mean the
**pipeline is correct**, not that the detector is hard to fool. The simple
statistical thresholds have real limits (e.g. the >3σ distribution check is
insensitive on Online Retail's heavy-tailed amounts and needs ~8× drift to fire).
See [METRICS.md](./METRICS.md) for the full real-vs-synthetic breakdown and the
threshold characteristics found while building this.

---

## TL;DR — Quickstart

> Commands assume the repo's local virtualenv at `./venv`. Substitute your own interpreter if you manage environments differently.

```bash
# 0. Setup (once)
python3 -m venv venv
./venv/bin/pip install -r requirements-full.txt   # full service + data/eval deps; demo-only deps live in requirements.txt
```

### 1. Zero-infra demo on REAL data (no Postgres, no Kafka)

The fastest way to see every layer working against the real dataset:

```bash
./venv/bin/python scripts/download_data.py   # once — caches UCI Online Retail
./venv/bin/python run_demo.py                # real profile + injected-fault eval
```

`run_demo.py --profile-only` prints just the real quality issues; `--eval-only`
prints just the precision/recall. An interactive **Streamlit** dashboard
(`streamlit_app.py`) is also included — pick the *Online Retail (REAL)* dataset
(after downloading) or the synthetic one, inject incidents with sliders, and watch
the detectors fire.

### 2. Full API service

```bash
./venv/bin/uvicorn src.api.main:app --port 8000
# interactive docs: http://localhost:8000/docs
```

The API prefers a Postgres connection (`DATABASE_URL`) for lineage persistence and **falls back to in-memory state when Postgres is absent**, so it will boot for a demo. OpenTelemetry tracing self-disables if no OTLP collector is reachable. See the [dependency matrix](#dependency-matrix) for what works in each mode.

### 3. Streaming mode (Kafka + Postgres)

```bash
docker compose up                       # core: postgres + api
docker compose --profile streaming up   # + Kafka, Zookeeper, Schema Registry, Kafka UI
docker compose --profile full up        # everything incl. Jaeger, Prometheus, Grafana
```

`docker compose config` validates cleanly.

### 4. Tests

```bash
./venv/bin/python -m pytest tests/unit tests/integration -q
# 22 passed in ~1.5s — no infra required
```

---

## The problem

Data pipelines fail silently. A schema changes upstream, an ETL job double-runs, a column starts arriving 80% null — and nothing throws an error. The bad data lands in a warehouse, flows into dashboards and ML features, and the first signal is a confused stakeholder hours later. By then the question isn't just "what broke" but "what did it touch."

Most quality tooling answers only the first half. You learn a table is bad but not which of your 200 downstream assets just inherited the problem, or which incident deserves the on-call page versus a ticket.

DataShield couples **detection** with **lineage-aware impact analysis** so a single bad batch produces a ranked, scoped answer: what failed, what it cascades into, how likely, and how urgent.

---

## Architecture

```
                      ┌─────────────────────────────────────────────┐
   Events / batches   │              QUALITY ENGINE                 │
   (Kafka topic or    │   Schema discovery → baseline metadata      │
    direct DataFrame) │   Statistical detectors (8 checks)          │
        ─────────────►│   ML detectors (Isolation Forest, LOF,      │
                      │     temporal, multivariate)                 │
                      │   Contract validation                       │
                      └───────────────────┬─────────────────────────┘
                                          │ alerts (typed, severity-ranked)
                                          ▼
                      ┌─────────────────────────────────────────────┐
                      │              LINEAGE GRAPH                  │
                      │   Dependency tracking (tables → tables)     │
                      │   Blast radius (BFS over the graph)         │
                      │   Probabilistic propagation (latency-aware) │
                      │   Escalation routing by criticality         │
                      └───────────────────┬─────────────────────────┘
                                          │ impact report
                                          ▼
                      ┌─────────────────────────────────────────────┐
                      │           FastAPI SERVICE / ALERTS          │
                      │   REST endpoints + /docs (OpenAPI)          │
                      │   Remediation engine (auto-fix where safe)  │
                      │   OpenTelemetry traces → Jaeger (optional)  │
                      └─────────────────────────────────────────────┘
```

```mermaid
flowchart LR
    K[Kafka topic / DataFrame] --> Q
    subgraph Q[Quality Engine]
        SD[Schema discovery] --> ST[Statistical detectors]
        SD --> ML[ML detectors]
        SD --> CT[Contract validation]
    end
    Q -- alerts --> L
    subgraph L[Lineage Graph]
        BR[Blast radius BFS] --> PP[Probabilistic propagation]
    end
    L -- impact report --> A[FastAPI / alerts / remediation]
```

---

## Dependency matrix

Not every feature needs every service. This table is the contract for what runs in each mode.

| Capability | Standalone (demo / library) | Needs Postgres | Needs Kafka | Notes |
|---|:---:|:---:|:---:|---|
| Schema discovery | ✅ | — | — | Pure pandas/NumPy |
| Statistical anomaly detection (8 checks) | ✅ | — | — | scipy + pandas |
| ML anomaly detection (4 methods) | ✅ | — | — | scikit-learn |
| Contract registration & validation | ✅ | — | — | In-memory registry |
| Lineage graph + blast radius | ✅ | optional | — | In-memory by default; Postgres persists it |
| Probabilistic propagation | ✅ | optional | — | Runs on the in-memory graph |
| Remediation engine | ✅ | — | — | Operates on detected alerts |
| `run_demo.py` (real-data profile + eval) | ✅ | — | — | Runs on cached UCI Online Retail; injected-fault precision/recall |
| `streamlit_app.py` (interactive dashboard) | ✅ | — | — | Real or synthetic dataset, slider-driven incidents |
| FastAPI service + `/docs` | ✅ (fallback) | recommended | — | Boots without Postgres via in-memory fallback |
| Persisted lineage across restarts | — | ✅ | — | `DATABASE_URL` → Postgres |
| Streaming ingestion (real-time consume) | — | ✅ | ✅ | `--profile streaming`; Schema Registry for Avro |
| Distributed tracing (Jaeger) | — | — | — | Optional OTLP collector; auto-disabled if absent |

**Rule of thumb:** the entire detection + lineage core is pure Python and runs standalone. Postgres buys you persistence; Kafka buys you real-time streaming. Everything degrades gracefully when those are missing.

---

## API surface

The FastAPI app (`src/api/main.py`, v0.3.0) exposes the full platform. Highlights:

| Method & path | Purpose |
|---|---|
| `GET /health` | Component readiness (quality engine, lineage, ML, contracts, tracing) |
| `POST /api/quality/discover` | Learn a baseline schema for a table |
| `POST /api/quality/detect` | Statistical anomaly detection against the baseline |
| `POST /api/ml/detect` | ML detection (Isolation Forest + LOF + temporal + multivariate) |
| `POST /api/ml/compare` | Run both detection families and compare results |
| `POST /api/lineage/initialize` · `/add-table` · `/add-dependency` | Build the lineage graph |
| `POST /api/lineage/blast-radius` | Compute downstream impact + escalation channels |
| `POST /api/remediation/remediate` | Detect then auto-remediate where safe |
| `POST /api/contracts/register` · `/validate` | Data-contract registry + validation |
| `POST /api/gnn/train` · `/predict` · `/compare` | Experimental GNN cascade prediction vs. heuristic |

Full, browsable schema lives at `http://localhost:8000/docs` once the service is running.

---

## Using it as a library

```python
import pandas as pd
from quality_engine.schema import SchemaDiscovery
from quality_engine.anomaly_detector import AnomalyDetector
from lineage.database import LineageDB
from lineage.blast_radius import BlastRadiusCalculator

# 1. Learn a baseline, then detect drift on a new batch
baseline = SchemaDiscovery().discover(df_yesterday, "transactions")
alerts = AnomalyDetector(baseline).detect(df_today)

# 2. Score downstream impact of a failing table
db = LineageDB()
raw = db.add_table("raw_events", "source", "data_eng", "de@co.com", "critical", "real-time")
clean = db.add_table("cleaned_events", "transformation", "data_eng", "de@co.com", "high", "hourly")
db.add_dependency(raw, clean, latency_minutes=5)

report = BlastRadiusCalculator(db).calculate(raw)
print(report.total_affected, report.critical_affected)
```

(`pyproject.toml` sets `pythonpath = ["src"]`, so these imports resolve when running under the project venv / pytest.)

---

## Testing & honest metrics

This section is deliberately literal about what is and isn't verified. Full
numbers and reproduction steps are in [METRICS.md](./METRICS.md).

| Claim | Status |
|---|---|
| **Real base dataset** | UCI Online Retail, 541,909 real transactions. Real quality issues counted directly (24.93% null CustomerID, 1.71% cancellations, 1.96% negative quantities, 5,268 duplicates, outlier amounts). Reproduce with `python run_demo.py --profile-only`. |
| **Detection precision/recall** | **Real**, measured against injected faults with ground-truth labels (`src/eval`). All five injected fault types caught at 1.00/1.00; clean controls not false-flagged. The faults are large/unambiguous — perfect scores mean the pipeline is correct, not that the detector is hard to fool. See the blunt caveats in METRICS.md. |
| **Test suite** | **26 pass with no infrastructure** (`tests/unit` + `tests/integration`), including 4 new real-data eval tests that skip cleanly without the dataset cache. `tests/load` / `tests/chaos` need running services and aren't in this count. |
| **Latency numbers** | Blast-radius / detection timings are fast locally on small-to-mid graphs, but the sub-millisecond / 100K-table figures were extrapolations, not load-tested, and have been removed. Run `tests/load/load_test_100k_tables.py` to measure on your hardware. |
| **`docker compose config`** | Validates cleanly. |

Removed from prior versions: unsourced ROI/dollar claims ("$2–5M annual losses
prevented", "detects failures 8 hours before humans", MTTR-in-dollars tables).
They were not backed by any code or measurement in this repo and have been cut.
The old synthetic-only "~94% accuracy" figure is superseded by the real
injected-fault evaluation above.

Run the verified suite:

```bash
./venv/bin/python -m pytest tests/unit tests/integration -q
```

---

## Tech stack

| Layer | Tools | Role |
|---|---|---|
| API | FastAPI, Uvicorn, Pydantic | Async REST service, request/response validation, OpenAPI docs |
| Data & ML | pandas, NumPy, SciPy, scikit-learn | Schema discovery, statistical checks, Isolation Forest / LOF / temporal / multivariate detectors |
| Persistence | PostgreSQL, SQLAlchemy, Alembic | Lineage metadata storage and migrations (optional; in-memory fallback) |
| Streaming | Kafka via `confluent-kafka`, Confluent Schema Registry | Real-time event ingestion and Avro schema management |
| Observability | OpenTelemetry (SDK + OTLP exporter), Jaeger, Prometheus, Grafana | Distributed tracing and metrics (all optional, auto-disable if absent) |
| Packaging & ops | Docker, Docker Compose, Helm / Kubernetes | Local profiles (core / streaming / observability / full) and a K8s chart under `helm/datashield` |

---

## Repository layout

```
DataShield/
├── run_demo.py                 # Zero-infra demo on REAL data (profile + injected-fault eval)
├── streamlit_app.py            # Interactive dashboard (real or synthetic dataset)
├── scripts/download_data.py    # Cache the real UCI Online Retail dataset under data/
├── data/                       # Cached dataset (gitignored; reproduce via the script)
├── docker-compose.yml          # Profiles: core / streaming / observability / full
├── Dockerfile
├── requirements.txt            # Streamlit demo deps ; requirements-full.txt = service + data/eval
├── pyproject.toml              # pythonpath=["src"], pytest config
├── src/
│   ├── eval/                   # REAL-data profiling + fault injection + precision/recall eval
│   │   ├── real_data.py        #   load + profile Online Retail, run the quality engine on it
│   │   ├── fault_injection.py  #   inject KNOWN labelled faults into clean real slices
│   │   └── evaluate.py         #   score detector vs ground truth -> precision/recall
│   ├── api/main.py             # FastAPI app (all endpoints, v0.3.0)
│   ├── quality_engine/         # schema discovery + statistical detectors (+ type-drift check)
│   ├── ml_features/            # ML anomaly detector (4 methods)
│   ├── lineage/                # graph DB, blast radius, graph optimizer
│   ├── contracts/              # contract registry + validator
│   ├── remediation/            # auto-remediation engine + actions
│   ├── streaming/              # Kafka producer/consumer, schema registry client
│   ├── observability/          # OpenTelemetry tracing setup
│   └── gnn/                    # experimental GNN cascade predictor
├── tests/
│   ├── unit/                   # quality engine, ML detector, graph optimizer, real-data eval
│   ├── integration/            # lineage graph end-to-end
│   ├── load/                   # 100K-table load test (needs running services)
│   └── chaos/                  # resilience tests (needs running services)
├── helm/datashield/            # Kubernetes Helm chart
├── benchmarks/ · docs/         # benchmark notes and deep-dive write-ups
├── METRICS.md                  # honest, reproduced metrics (real vs synthetic)
└── demo.ipynb                  # Jupyter walkthrough
```

---

## License

MIT. See the badge above; add a `LICENSE` file if distributing.

---

*Built by Koutilya Yenumula ([@koutilyaY](https://github.com/koutilyaY)). This is a portfolio project demonstrating data-engineering systems design — read the [metrics section](#testing--honest-metrics) for the verified-vs-aspirational breakdown.*
