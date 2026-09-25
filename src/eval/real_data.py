"""
Load and profile the REAL UCI Online Retail dataset, and run the existing
DataShield quality engine over it.

Nothing here is synthetic. The numbers this module prints are whatever is
actually in the ~541,909-row Online Retail file: real null CustomerIDs, real
cancellations, real negative quantities, real outlier amounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from quality_engine.schema import SchemaDiscovery
from quality_engine.anomaly_detector import AnomalyDetector

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE = DATA_DIR / "online_retail.parquet"


def load_online_retail() -> pd.DataFrame:
    """Load the cached real dataset. Run scripts/download_data.py first."""
    if not CACHE.exists():
        raise FileNotFoundError(
            f"{CACHE} not found. Run:  python scripts/download_data.py"
        )
    df = pd.read_parquet(CACHE)
    # Parse the invoice date if it came through as text.
    if df["InvoiceDate"].dtype == object:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


@dataclass
class RealDataReport:
    """Actual quality issues found in the real Online Retail data."""

    n_rows: int
    n_cols: int
    columns: List[str]
    null_rates: Dict[str, float]
    n_cancelled: int
    cancelled_rate: float
    n_negative_qty: int
    negative_qty_rate: float
    n_nonpositive_price: int
    n_duplicate_rows: int
    quantity_min: int
    quantity_max: int
    price_min: float
    price_max: float
    outlier_amount_rows: int
    engine_alerts: List[dict] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        lines = [
            f"Real UCI Online Retail: {self.n_rows:,} rows x {self.n_cols} cols",
            "Null rates (real): "
            + ", ".join(
                f"{c}={r:.1%}" for c, r in self.null_rates.items() if r > 0
            ),
            f"Cancelled invoices (InvoiceNo starts 'C'): "
            f"{self.n_cancelled:,} ({self.cancelled_rate:.2%})",
            f"Negative quantity rows: {self.n_negative_qty:,} "
            f"({self.negative_qty_rate:.2%})  [returns/cancellations]",
            f"Non-positive UnitPrice rows: {self.n_nonpositive_price:,}",
            f"Exact duplicate rows: {self.n_duplicate_rows:,}",
            f"Quantity range: [{self.quantity_min:,}, {self.quantity_max:,}]  "
            f"UnitPrice range: [{self.price_min:.2f}, {self.price_max:.2f}]",
            f"Rows with outlier line-amount (|z|>3 on Quantity*UnitPrice): "
            f"{self.outlier_amount_rows:,}",
            f"Quality-engine alerts on the raw feed: {len(self.engine_alerts)}",
        ]
        return lines


def profile_real_data(df: pd.DataFrame) -> RealDataReport:
    """Compute the real quality issues present in Online Retail."""
    invoice = df["InvoiceNo"].astype(str)
    cancelled = invoice.str.startswith("C")
    neg_qty = df["Quantity"] < 0
    nonpos_price = df["UnitPrice"] <= 0

    amount = df["Quantity"] * df["UnitPrice"]
    a = amount.to_numpy(dtype="float64")
    std = a.std()
    if std > 0:
        z = np.abs((a - a.mean()) / std)
        outliers = int((z > 3).sum())
    else:
        outliers = 0

    return RealDataReport(
        n_rows=len(df),
        n_cols=df.shape[1],
        columns=list(df.columns),
        null_rates={c: float(df[c].isna().mean()) for c in df.columns},
        n_cancelled=int(cancelled.sum()),
        cancelled_rate=float(cancelled.mean()),
        n_negative_qty=int(neg_qty.sum()),
        negative_qty_rate=float(neg_qty.mean()),
        n_nonpositive_price=int(nonpos_price.sum()),
        n_duplicate_rows=int(df.duplicated().sum()),
        quantity_min=int(df["Quantity"].min()),
        quantity_max=int(df["Quantity"].max()),
        price_min=float(df["UnitPrice"].min()),
        price_max=float(df["UnitPrice"].max()),
        outlier_amount_rows=outliers,
    )


def run_quality_engine(df: pd.DataFrame, table_name: str = "online_retail") -> RealDataReport:
    """
    Run the existing SchemaDiscovery + AnomalyDetector over the real data and
    attach the engine's alerts to the profile report.

    We learn a baseline from a clean-ish first month, then run detection on the
    whole feed. On real data the engine legitimately fires: distribution shifts
    across the year, cardinality changes, etc. Those alerts are real signal from
    the real feed, not injected.
    """
    report = profile_real_data(df)

    # Baseline from the earliest 30 days of activity (approximates "known-good"
    # early state); detect on the full feed.
    dated = df.dropna(subset=["InvoiceDate"]).sort_values("InvoiceDate")
    if len(dated) > 0:
        cutoff = dated["InvoiceDate"].min() + pd.Timedelta(days=30)
        baseline_df = dated[dated["InvoiceDate"] < cutoff]
    else:
        baseline_df = df.head(len(df) // 12)

    if len(baseline_df) < 100:
        baseline_df = df.head(min(len(df), 50_000))

    baseline = SchemaDiscovery().discover(baseline_df, table_name)
    detector = AnomalyDetector(baseline)
    alerts = detector.detect(df)
    report.engine_alerts = [a.to_dict() for a in alerts]
    return report


if __name__ == "__main__":
    df = load_online_retail()
    rep = run_quality_engine(df)
    print("\n".join(rep.summary_lines()))
    print("\nEngine alerts:")
    for a in rep.engine_alerts:
        print(f"  [{a['severity']}] {a['anomaly_type']}: {a['message']}")
