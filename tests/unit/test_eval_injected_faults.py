"""
Tests for the real-data evaluation harness (src/eval).

These tests need the cached UCI Online Retail dataset. If it is not present they
skip with a clear message (run scripts/download_data.py to enable them). They
verify that:
  - injected-fault batches carry correct ground-truth labels,
  - the detector's type-drift check catches a numeric->string column change,
  - the evaluation produces sane precision/recall on the known faults.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, "src")

from eval.fault_injection import (  # noqa: E402
    FaultType,
    make_clean_slice,
    build_evaluation_batches,
    _inject,
)
from eval.evaluate import evaluate_detector, _learn_baseline  # noqa: E402
from quality_engine import SchemaDiscovery, AnomalyDetector, AnomalyType  # noqa: E402

CACHE = Path("data/online_retail.parquet")
pytestmark = pytest.mark.skipif(
    not CACHE.exists(),
    reason="Online Retail cache missing; run scripts/download_data.py",
)


@pytest.fixture(scope="module")
def real_df():
    return pd.read_parquet(CACHE)


def test_clean_slice_is_actually_clean(real_df):
    clean = make_clean_slice(real_df)
    assert len(clean) > 0
    assert clean["CustomerID"].isna().sum() == 0
    assert (clean["Quantity"] > 0).all()
    assert (clean["UnitPrice"] > 0).all()
    assert "LineAmount" in clean.columns


def test_batches_carry_ground_truth_labels(real_df):
    batches = build_evaluation_batches(real_df, batch_size=1_000, reps=2)
    faults = {b.fault for b in batches}
    # every fault type plus the clean control should be represented
    for f in FaultType:
        assert f in faults
    assert any(b.is_faulty for b in batches)
    assert any(not b.is_faulty for b in batches)


def test_type_drift_is_detected(real_df):
    """A numeric column arriving as text must raise a schema-drift alert."""
    clean = make_clean_slice(real_df).head(2_000)
    baseline = SchemaDiscovery().discover(clean, "t")
    detector = AnomalyDetector(baseline)

    import numpy as np

    broken = _inject(clean.copy(), FaultType.SCHEMA_TYPE_CHANGE, np.random.default_rng(0))
    alerts = detector.detect(broken)
    drift = [a for a in alerts if a.anomaly_type == AnomalyType.SCHEMA_DRIFT]
    assert any("type changed" in a.message.lower() for a in drift)


def test_evaluation_produces_real_metrics(real_df):
    result = evaluate_detector(real_df, batch_size=1_500, reps=3)
    assert result.n_batches > 0
    assert result.n_faulty > 0 and result.n_clean > 0
    # Each injected fault should be caught at least sometimes (recall > 0).
    for fault, m in result.per_fault.items():
        assert m.recall == m.recall, f"{fault} recall is NaN"  # not NaN
        assert m.recall > 0.0, f"{fault} never detected (recall=0)"
    # Clean controls should not all be false-flagged: overall precision high.
    assert result.overall_precision >= 0.8
