"""
DataShield — Live Demo (Streamlit Community Cloud)
==================================================

A zero-infrastructure, browser-based demonstration of the DataShield
real-time data-observability platform. No Postgres, no Kafka, no FastAPI
server required — this app imports the *real* DataShield engine modules
directly and runs them in-process on a sample dataset you control.

What's running under the hood (all real repo code from `src/`):
  - quality_engine.SchemaDiscovery   -> learns a "known-good" baseline schema
  - quality_engine.AnomalyDetector   -> 8 statistical data-quality checks
  - ml_features.ml_anomaly_detector  -> Isolation Forest / LOF / temporal / multivariate ML
  - lineage.LineageDB + BlastRadiusCalculator -> downstream blast-radius (BFS)

Entry point for Streamlit Community Cloud (repo root: `streamlit_app.py`).
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# --- Make the repo's src/ importable (works locally and on Streamlit Cloud) ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# --- Import the REAL DataShield engine modules (zero-infra code paths) ---
from quality_engine import SchemaDiscovery, AnomalyDetector, SeverityLevel  # noqa: E402
from ml_features.ml_anomaly_detector import MLAnomalyDetector  # noqa: E402
from lineage.database import LineageDB  # noqa: E402
from lineage.blast_radius import BlastRadiusCalculator  # noqa: E402


# =============================================================================
# Page config + theme
# =============================================================================
st.set_page_config(
    page_title="DataShield — Live Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 2rem; }
.ds-hero h1 { font-size: 2.4rem; margin-bottom: .25rem;
  background: linear-gradient(135deg,#3b82f6,#8b5cf6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.ds-sub { color:#94a3b8; font-size:1.05rem; }
.ds-pill { display:inline-block; padding:2px 10px; border-radius:999px;
  font-size:.78rem; font-weight:600; margin-right:6px; }
.ds-pill.ok { background:#064e3b; color:#6ee7b7; }
.ds-pill.crit { background:#7f1d1d; color:#fca5a5; }
.ds-pill.warn { background:#78350f; color:#fcd34d; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# Sample data generation
# =============================================================================
COLUMNS = ["transaction_id", "amount", "customer_id", "status"]


@st.cache_data(show_spinner=False)
def make_baseline(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """A clean, known-good 'transactions' dataset — the baseline DataShield learns.

    NOTE: this is SYNTHETIC data used only as an easy, always-available baseline
    for the interactive slider demo. The REAL evaluation on the UCI Online Retail
    dataset lives in `run_demo.py` / `src/eval` (real base rows + injected faults
    with ground truth). See the README.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "transaction_id": np.arange(1, n + 1),
            "amount": rng.normal(100, 20, n).round(2),
            "customer_id": rng.integers(1000, 9999, n),
            "status": rng.choice(["completed", "pending", "failed"], n),
        }
    )


@st.cache_data(show_spinner=False)
def load_real_online_retail():
    """Load the REAL UCI Online Retail clean slice if the cache is present.

    Returns (baseline_df, available_flag). The columns are renamed to the demo's
    transaction schema so the same detectors run unchanged.
    """
    cache = os.path.join(REPO_ROOT, "data", "online_retail.parquet")
    if not os.path.exists(cache):
        return None, False
    raw = pd.read_parquet(cache)
    invoice = raw["InvoiceNo"].astype(str)
    mask = (
        raw["CustomerID"].notna()
        & (raw["Quantity"] > 0)
        & (raw["UnitPrice"] > 0)
        & ~invoice.str.startswith("C")
    )
    clean = raw.loc[mask]
    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, len(clean) + 1),
            "amount": (clean["Quantity"].values * clean["UnitPrice"].values).round(2),
            "customer_id": clean["CustomerID"].astype("int64").values,
            "status": clean["Country"].astype(str).values,
        }
    )
    return df, True


def make_observed(
    n: int,
    seed: int,
    *,
    row_spike_pct: float,
    null_rate: float,
    amount_shift: float,
    cardinality_collapse: bool,
    inject_pii: bool,
    drop_column: bool,
    source_df: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Generate/sample a fresh batch and optionally inject quality incidents.

    If `source_df` is given (the REAL Online Retail clean slice), the batch is a
    real sample and `amount_shift` is applied additively to the real amounts.
    Otherwise the batch is synthetic.
    """
    base_rows = int(n * (1 + row_spike_pct / 100.0))
    base_rows = max(base_rows, 50)
    rng = np.random.default_rng(seed + 1)

    if source_df is not None:
        df = source_df.sample(n=min(base_rows, len(source_df)),
                              random_state=seed + 1, replace=base_rows > len(source_df)
                              ).reset_index(drop=True)
        df["transaction_id"] = np.arange(1, len(df) + 1)
        df["amount"] = (df["amount"] + amount_shift).round(2)
    else:
        df = pd.DataFrame(
            {
                "transaction_id": np.arange(1, base_rows + 1),
                "amount": rng.normal(100 + amount_shift, 20, base_rows).round(2),
                "customer_id": rng.integers(1000, 9999, base_rows),
                "status": rng.choice(["completed", "pending", "failed"], base_rows),
            }
        )

    # Null explosion on `amount`
    if null_rate > 0:
        mask = rng.random(base_rows) < (null_rate / 100.0)
        df.loc[mask, "amount"] = np.nan

    # Cardinality collapse on customer_id (everyone becomes one customer)
    if cardinality_collapse:
        df["customer_id"] = 1001

    # PII exposure: overwrite `status` with email addresses
    if inject_pii:
        df["status"] = [f"user{ i }@example.com" for i in range(base_rows)]

    # Schema drift: drop a required column
    if drop_column:
        df = df.drop(columns=["customer_id"])

    return df


# =============================================================================
# Sidebar — controls
# =============================================================================
with st.sidebar:
    st.header("⚙️ Demo controls")
    st.caption("Pick a dataset, then inject incidents to see DataShield react.")

    _real_df, _real_ok = load_real_online_retail()
    _options = ["transactions (synthetic)"]
    if _real_ok:
        _options.insert(0, "Online Retail (REAL, UCI)")
    dataset = st.selectbox(
        "Sample dataset",
        _options,
        help="'Online Retail (REAL)' learns the baseline from a clean slice of the "
        "real UCI dataset (available after running scripts/download_data.py). "
        "'transactions (synthetic)' is generated data for the slider demo.",
    )
    if not _real_ok:
        st.caption("Run `python scripts/download_data.py` to unlock the REAL "
                   "Online Retail dataset here.")
    batch_size = st.slider("Observed batch size (rows)", 200, 12_000, 8_000, 200)
    seed = st.number_input("Random seed", 0, 9999, 7, 1)

    st.divider()
    st.subheader("💥 Inject incidents")
    row_spike = st.slider("Row-count spike / drop (%)", -90, 200, 0, 5,
                          help="±20% is tolerated; beyond that triggers a row-count alert.")
    null_rate = st.slider("Null rate on `amount` (%)", 0, 90, 0, 5,
                          help="Baseline has ~0% nulls; >0% will explode the null check.")
    amount_shift = st.slider("Mean shift on `amount` ($)", 0, 300, 0, 10,
                             help="Shifts the distribution; >3σ triggers distribution-shift.")
    cardinality_collapse = st.toggle("Collapse `customer_id` cardinality")
    inject_pii = st.toggle("Inject PII into `status` (emails)")
    drop_column = st.toggle("Drop `customer_id` (schema drift)")

    st.divider()
    contamination = st.slider("ML contamination (expected outlier %)", 1, 20, 5, 1) / 100.0
    run = st.button("▶️ Run DataShield scan", type="primary", use_container_width=True)


# =============================================================================
# Header
# =============================================================================
st.markdown(
    """
<div class="ds-hero">
  <h1>🛡️ DataShield — Live Observability Demo</h1>
  <div class="ds-sub">Real-time data-quality checks, ML anomaly detection, and
  blast-radius lineage — running the <b>real engine code</b> in-process, zero infra.</div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption(
    "Baseline schema is learned from 10,000 clean transactions, then your observed "
    "batch is scored against it using `quality_engine`, `ml_features`, and `lineage`."
)


# =============================================================================
# Run pipeline
# =============================================================================
SEV_RANK = {"critical": 3, "warning": 2, "info": 1}
SEV_BADGE = {"critical": "🔴", "warning": "🟠", "info": "🔵"}


@st.cache_resource(show_spinner=False)
def get_baseline_metadata(use_real: bool = False):
    if use_real:
        real_df, ok = load_real_online_retail()
        if ok:
            base = real_df.sample(n=min(10_000, len(real_df)), random_state=42).reset_index(drop=True)
            meta = SchemaDiscovery().discover(base, "online_retail")
            return base, meta
    base = make_baseline()
    meta = SchemaDiscovery().discover(base, "transactions")
    return base, meta


def quality_score(alerts) -> int:
    """0-100 health score: each alert costs points by severity."""
    penalty = 0
    for a in alerts:
        penalty += {"critical": 25, "warning": 10, "info": 4}.get(a.severity.value, 5)
    return max(0, 100 - penalty)


if run:
    use_real = dataset.startswith("Online Retail")
    base_df, baseline_meta = get_baseline_metadata(use_real=use_real)

    observed = make_observed(
        batch_size,
        int(seed),
        row_spike_pct=row_spike,
        null_rate=null_rate,
        amount_shift=amount_shift,
        cardinality_collapse=cardinality_collapse,
        inject_pii=inject_pii,
        drop_column=drop_column,
        source_df=base_df if use_real else None,
    )

    # --- REAL statistical quality checks ---
    detector = AnomalyDetector(baseline_meta)
    alerts = detector.detect(observed)

    # --- REAL ML anomaly detection ---
    ml_detector = MLAnomalyDetector(contamination=contamination)
    ml_alerts = ml_detector.detect(observed)

    score = quality_score(alerts)
    crit = sum(1 for a in alerts if a.severity.value == "critical")
    warn = sum(1 for a in alerts if a.severity.value == "warning")

    # ---------------------------------------------------------------- KPIs
    st.subheader("📊 Observability dashboard")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Quality score", f"{score}/100",
              delta="healthy" if score >= 85 else "degraded",
              delta_color="normal" if score >= 85 else "inverse")
    k2.metric("Statistical alerts", len(alerts))
    k3.metric("Critical", crit)
    k4.metric("ML anomalies", len(ml_alerts))
    k5.metric("Rows scanned", f"{len(observed):,}")

    if score >= 85:
        st.success("✅ Data is healthy — passes DataShield quality gates.")
    elif crit:
        st.error(f"🚨 {crit} critical incident(s) detected — downstream consumers at risk.")
    else:
        st.warning("⚠️ Degraded quality — warnings detected, review before promoting.")

    st.divider()

    # ------------------------------------------------ Check matrix (pass/fail)
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### ✅ Quality checks (statistical engine)")
        all_checks = [
            ("Row-count stability", "row_count_spike"),
            ("Null-rate guard", "null_rate_explosion"),
            ("Cardinality guard", "cardinality_collapse"),
            ("Distribution shift", "distribution_shift"),
            ("Schema drift", "schema_drift"),
            ("PII exposure", "pii_exposure"),
        ]
        failed_types = {a.anomaly_type.value: a for a in alerts}
        rows = []
        for label, atype in all_checks:
            if atype in failed_types:
                a = failed_types[atype]
                rows.append({"Check": label, "Status": "FAIL",
                             "Severity": a.severity.value, "Detail": a.message})
            else:
                rows.append({"Check": label, "Status": "PASS",
                             "Severity": "-", "Detail": "within tolerance"})
        cdf = pd.DataFrame(rows)

        def _style(row):
            color = "#064e3b" if row["Status"] == "PASS" else (
                "#7f1d1d" if row["Severity"] == "critical" else "#78350f")
            return [f"background-color: {color}"] * len(row)

        st.dataframe(cdf.style.apply(_style, axis=1),
                     use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### 🤖 ML anomaly detectors")
        if ml_alerts:
            ml_rows = [{
                "Detector": a.anomaly_type.value.replace("_", " ").title(),
                "Column": a.column,
                "Severity": a.severity.value,
                "Score": round(a.score, 3),
                "Rows flagged": len(a.affected_indices),
            } for a in ml_alerts]
            st.dataframe(pd.DataFrame(ml_rows),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No ML anomalies flagged at this contamination level.")
        st.caption("Isolation Forest · Local Outlier Factor · Temporal · Multivariate "
                   "(scikit-learn, real `MLAnomalyDetector`).")

    st.divider()

    # ------------------------------------------------ Alerts feed
    st.markdown("#### 🔔 Alert feed")
    if alerts:
        for a in sorted(alerts, key=lambda x: -SEV_RANK[x.severity.value]):
            badge = SEV_BADGE[a.severity.value]
            with st.container(border=True):
                st.markdown(f"{badge} **{a.anomaly_type.value.replace('_',' ').title()}** "
                            f"· `{a.severity.value}`"
                            + (f" · column `{a.column_name}`" if a.column_name else ""))
                st.write(a.message)
                if a.deviation_percent is not None:
                    st.caption(f"Deviation: {a.deviation_percent:+.1f}%  "
                               f"(baseline {a.baseline_value} → observed {a.observed_value})")
    else:
        st.success("No statistical anomalies — all checks within tolerance.")

    st.divider()

    # ------------------------------------------------ Chart: amount distribution
    st.markdown("#### 📈 `amount` distribution: baseline vs. observed")
    bins = np.linspace(0, 300, 40)
    base_hist = np.histogram(base_df["amount"].dropna(), bins=bins)[0]
    if "amount" in observed:
        obs_hist = np.histogram(observed["amount"].dropna(), bins=bins)[0]
    else:
        obs_hist = np.zeros(len(bins) - 1)
    centers = ((bins[:-1] + bins[1:]) / 2).round(1)
    chart_df = pd.DataFrame(
        {"baseline": base_hist, "observed": obs_hist},
        index=centers,
    )
    chart_df.index.name = "amount ($)"
    st.bar_chart(chart_df, color=["#3b82f6", "#f97316"])

    # ------------------------------------------------ Data preview w/ flagged rows
    st.markdown("#### 🔎 Observed batch (ML-flagged rows highlighted)")
    flagged = set()
    for a in ml_alerts:
        flagged.update(a.affected_indices)
    preview = observed.head(200).copy()
    preview.insert(0, "⚠️", ["⚠️" if i in flagged else "" for i in preview.index])
    st.dataframe(preview, use_container_width=True, height=320, hide_index=True)
    st.caption(f"{len(flagged)} of {len(observed):,} rows flagged by ML detectors "
               f"(showing first 200).")

    st.divider()

    # ------------------------------------------------ Blast radius (lineage)
    st.markdown("#### 🌐 Blast radius — downstream impact if `transactions` fails")
    db = LineageDB()
    raw = db.add_table("raw_events", "source", "data_eng", "de@co.com", "critical", "real-time")
    clean = db.add_table("cleaned_events", "transformation", "data_eng", "de@co.com", "high", "hourly")
    summary = db.add_table("order_summary", "transformation", "analytics", "analytics@co.com", "high", "daily")
    report = db.add_table("revenue_report", "dashboard", "finance", "finance@co.com", "critical", "daily")
    dash = db.add_table("executive_dashboard", "dashboard", "finance", "finance@co.com", "critical", "daily")
    db.add_dependency(raw, clean, 5)
    db.add_dependency(clean, summary, 10)
    db.add_dependency(summary, report, 60)
    db.add_dependency(report, dash, 120)

    calc = BlastRadiusCalculator(db)
    blast = calc.calculate(raw)
    b1, b2, b3 = st.columns(3)
    b1.metric("Tables affected", blast.total_affected)
    b2.metric("Critical downstream", blast.critical_affected)
    b3.metric("Compute time", f"{blast.computation_time_ms:.2f} ms")
    if blast.affected_tables:
        bl_df = pd.DataFrame([t.to_dict() for t in blast.affected_tables])[
            ["table_name", "criticality", "owner", "depth", "latency_minutes", "path"]
        ]
        st.dataframe(bl_df, use_container_width=True, hide_index=True)
    st.caption("BFS over the in-memory lineage graph — answers 'who do I page, and when "
               "does each downstream asset break?'")

    st.divider()
    st.caption(f"Scan completed {datetime.utcnow().isoformat()}Z · "
               "DataShield engine running in-process · zero infrastructure.")

else:
    st.info("👈 Configure the dataset and incidents in the sidebar, then click "
            "**Run DataShield scan**.")
    st.markdown(
        """
**Try this:** crank up *Null rate on `amount`* to 60%, set *Mean shift* to 150,
toggle *Inject PII*, and watch the quality score crater while the alert feed,
ML detectors, and blast-radius panel all light up.
"""
    )
