"""
Download the UCI Online Retail dataset and cache it under data/.

Online Retail is a real transactional dataset from a UK-based online retailer:
~541,909 rows of actual e-commerce transactions between 2010-12-01 and 2011-12-09.
It is genuinely messy — missing CustomerIDs, cancelled orders (negative
quantities, InvoiceNo starting with 'C'), inconsistent free-text descriptions,
and the odd negative/zero unit price. That messiness is the point: it lets us
run the quality engine on real data instead of something we generated ourselves.

Source: UCI Machine Learning Repository, "Online Retail", dataset id 352.
  https://archive.ics.uci.edu/dataset/352/online+retail
Citation: Daqing Chen. Online Retail. UCI Machine Learning Repository, 2015.

The download is cached to data/online_retail.parquet. The data/ directory is
gitignored; only this script is committed, so anyone can reproduce the cache.

Two paths, tried in order:
  1. ucimlrepo (pip install ucimlrepo) -> fetch_ucirepo(id=352)
  2. Direct download of the .xlsx from the UCI archive (needs openpyxl)

No API keys required for either path.
"""

from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "online_retail.parquet"

# Direct archive URL (fallback if ucimlrepo is unavailable).
XLSX_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Give columns stable names regardless of the source path."""
    rename = {
        "Invoice": "InvoiceNo",
        "Customer ID": "CustomerID",
        "Price": "UnitPrice",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Keep a consistent column order when present.
    preferred = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
    ]
    cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    return df[cols]


def _via_ucimlrepo() -> pd.DataFrame | None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return None
    print("Fetching via ucimlrepo (id=352)...")
    ds = fetch_ucirepo(id=352)
    # ucimlrepo splits into features/targets; Online Retail has no target,
    # so everything is in .data.features (or .data.original).
    df = getattr(ds.data, "original", None)
    if df is None:
        df = ds.data.features
    return _normalize(df)


def _via_archive() -> pd.DataFrame:
    print(f"Downloading .xlsx from UCI archive:\n  {XLSX_URL}")
    req = urllib.request.Request(XLSX_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    return _normalize(df)


def download(force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if CACHE.exists() and not force:
        print(f"Already cached: {CACHE}")
        return CACHE

    df = _via_ucimlrepo()
    if df is None:
        print("ucimlrepo not installed; falling back to direct archive download.")
        df = _via_archive()

    df.to_parquet(CACHE, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {CACHE}")
    return CACHE


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = download(force=force)
    df = pd.read_parquet(path)
    print(f"\nCached dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("Columns:", list(df.columns))
