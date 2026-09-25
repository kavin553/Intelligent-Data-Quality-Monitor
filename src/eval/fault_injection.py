"""
Inject KNOWN, labelled faults into clean slices of the real data.

Why this exists: you cannot measure a detector's precision/recall without ground
truth, and the real Online Retail feed does not come with labels saying "this
batch is broken". So we take a clean slice of the *real* data and produce copies
with specific, known faults injected. Each copy carries a label of which fault
was injected. That label is the ground truth the evaluation scores against.

To be blunt: the base rows are real, but the fault injection is synthetic. That
is the standard and honest way to evaluate a data-quality system — the alternative
(hand-labelling hundreds of production incidents) is not available here.

Clean slice = rows from Online Retail with:
  - a non-null CustomerID
  - a positive Quantity (drop returns/cancellations)
  - a positive UnitPrice
This gives a genuinely tidy base so that an injected fault is the only anomaly
in a batch, which is what makes precision measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class FaultType(str, Enum):
    """The known faults we inject. 'clean' means no fault (a control batch)."""

    CLEAN = "clean"
    NULL_SPIKE = "null_spike"
    DISTRIBUTION_DRIFT = "distribution_drift"
    SCHEMA_TYPE_CHANGE = "schema_type_change"
    PII_INJECTION = "pii_injection"
    CARDINALITY_COLLAPSE = "cardinality_collapse"


# Map an injected fault to the detector alert type(s) that should catch it.
# The detector emits AnomalyType values; we consider a batch "caught" if it
# raises at least one alert of a type in this set.
FAULT_TO_ALERT_TYPES = {
    FaultType.NULL_SPIKE: {"null_rate_explosion"},
    FaultType.DISTRIBUTION_DRIFT: {"distribution_shift"},
    # A numeric column arriving as text is caught by the engine's type-drift
    # check, which raises a schema_drift alert naming the changed column.
    FaultType.SCHEMA_TYPE_CHANGE: {"schema_drift"},
    FaultType.PII_INJECTION: {"pii_exposure"},
    FaultType.CARDINALITY_COLLAPSE: {"cardinality_collapse"},
}


@dataclass
class InjectedBatch:
    """One evaluation batch: a DataFrame plus the fault label (ground truth)."""

    df: pd.DataFrame
    fault: FaultType
    seed: int

    @property
    def is_faulty(self) -> bool:
        return self.fault != FaultType.CLEAN


def make_clean_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Return a genuinely clean subset of the real data (real rows, no faults)."""
    invoice = df["InvoiceNo"].astype(str)
    mask = (
        df["CustomerID"].notna()
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & ~invoice.str.startswith("C")
    )
    clean = df.loc[mask].copy()
    # Add a numeric line-amount column so numeric detectors have something real
    # to work with (drift, cardinality etc. operate on numeric columns).
    clean["LineAmount"] = (clean["Quantity"] * clean["UnitPrice"]).round(2)
    clean["CustomerID"] = clean["CustomerID"].astype("int64")
    return clean.reset_index(drop=True)


def _sample(clean: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return clean.sample(n=min(n, len(clean)), random_state=seed).reset_index(drop=True)


def _inject(batch: pd.DataFrame, fault: FaultType, rng: np.random.Generator) -> pd.DataFrame:
    """Apply one known fault to a clean batch."""
    b = batch.copy()
    n = len(b)

    if fault == FaultType.CLEAN:
        return b

    if fault == FaultType.NULL_SPIKE:
        # Blow out CustomerID nulls in the clean slice (clean slice has 0%).
        mask = rng.random(n) < 0.6
        b.loc[mask, "CustomerID"] = np.nan
        return b

    if fault == FaultType.DISTRIBUTION_DRIFT:
        # Multiply the amount distribution far past 3 sigma of the baseline.
        b["LineAmount"] = b["LineAmount"] * 8.0 + 500.0
        b["UnitPrice"] = b["UnitPrice"] * 8.0
        return b

    if fault == FaultType.SCHEMA_TYPE_CHANGE:
        # Quantity arrives as text ("12 units") instead of int: a real-world
        # upstream type break. Numeric checks can no longer read the column.
        b["Quantity"] = b["Quantity"].astype(str) + " units"
        return b

    if fault == FaultType.PII_INJECTION:
        # A column starts leaking email addresses where it should not.
        b["Description"] = [f"user{i}@example.com" for i in range(n)]
        return b

    if fault == FaultType.CARDINALITY_COLLAPSE:
        # Every row collapses onto a single CustomerID (a broken join / default).
        b["CustomerID"] = b["CustomerID"].iloc[0]
        return b

    return b


def build_evaluation_batches(
    df: pd.DataFrame,
    batch_size: int = 5_000,
    reps: int = 5,
    faults: Optional[List[FaultType]] = None,
    include_clean: bool = True,
    seed: int = 42,
) -> List[InjectedBatch]:
    """
    Build a labelled evaluation set from the real data.

    For each fault type we produce `reps` batches (different random samples of
    the real clean slice). We also produce `reps` clean control batches so that
    precision is measurable (a detector that fires on clean batches is punished).

    Returns a flat list of InjectedBatch, each carrying its ground-truth label.
    """
    clean = make_clean_slice(df)
    if faults is None:
        faults = [f for f in FaultType if f != FaultType.CLEAN]

    batches: List[InjectedBatch] = []
    counter = 0

    labels = list(faults)
    if include_clean:
        labels = labels + [FaultType.CLEAN]

    for fault in labels:
        for r in range(reps):
            s = seed + counter
            counter += 1
            base = _sample(clean, batch_size, s)
            rng = np.random.default_rng(s)
            injected = _inject(base, fault, rng)
            batches.append(InjectedBatch(df=injected, fault=fault, seed=s))

    return batches
