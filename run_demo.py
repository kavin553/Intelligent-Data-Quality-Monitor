"""
DataShield zero-infra demo — now on REAL data (UCI Online Retail).

Runs end to end with no Postgres, no Kafka, no server:

  1. Loads the real Online Retail dataset (541,909 e-commerce transactions).
     If the cache is missing it tells you to run scripts/download_data.py.
  2. Profiles the REAL quality issues in it (null CustomerIDs, cancellations,
     negative quantities, outlier amounts) and runs the quality engine on it.
  3. Builds labelled evaluation batches by injecting KNOWN faults into clean
     slices of the real data, then reports the detector's real precision/recall.

Usage:
    python run_demo.py                 # full demo (real profile + injected-fault eval)
    python run_demo.py --profile-only  # just the real-data quality report
    python run_demo.py --eval-only     # just the injected-fault precision/recall
    python run_demo.py --batch-size 5000 --reps 5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from eval.real_data import load_online_retail, run_quality_engine  # noqa: E402
from eval.evaluate import evaluate_detector  # noqa: E402


BANNER = "=" * 68


def _section(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def main() -> int:
    ap = argparse.ArgumentParser(description="DataShield real-data demo")
    ap.add_argument("--profile-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=5_000)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    try:
        df = load_online_retail()
    except FileNotFoundError as e:
        print(e)
        print("\nDownload the real dataset first:\n  python scripts/download_data.py")
        return 1

    if not args.eval_only:
        _section("REAL DATA  (UCI Online Retail — real, unmodified)")
        report = run_quality_engine(df)
        for line in report.summary_lines():
            print(line)
        if report.engine_alerts:
            print("\nQuality-engine alerts on the raw real feed:")
            for a in report.engine_alerts:
                print(f"  [{a['severity']}] {a['anomaly_type']}: {a['message']}")

    if not args.profile_only:
        _section(
            "DETECTION EVALUATION  (real rows + KNOWN injected faults = ground truth)"
        )
        result = evaluate_detector(
            df, batch_size=args.batch_size, reps=args.reps
        )
        for line in result.summary_lines():
            print(line)
        print(
            "\nBase rows are REAL. The injected faults are synthetic-with-labels "
            "so precision/recall can be measured. Numbers above are the detector's "
            "real scores on catching those known faults."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
