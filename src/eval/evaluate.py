"""
Score the DataShield detector on the injected-fault batches and report real
precision/recall.

Ground truth: each batch is labelled with the fault injected into it (or CLEAN).
For each fault type we ask: of the batches carrying this fault, how many did the
detector catch (recall), and of all the batches it flagged for this fault type,
how many actually carried it (precision)?

Definitions used here:
  - A batch is "flagged for fault F" if the detector raises at least one alert
    whose AnomalyType is in FAULT_TO_ALERT_TYPES[F].
  - Recall(F)    = caught faulty batches of type F / total faulty batches of type F
  - Precision(F) = caught faulty batches of type F / batches flagged as F
                   (a clean batch that trips F's alert types is a false positive)

We also report an overall "any fault vs clean" precision/recall: did the detector
raise ANY alert on a faulty batch, and did it stay silent on clean ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from quality_engine.schema import SchemaDiscovery
from quality_engine.anomaly_detector import AnomalyDetector

from .fault_injection import (
    FaultType,
    InjectedBatch,
    FAULT_TO_ALERT_TYPES,
    make_clean_slice,
    build_evaluation_batches,
)


@dataclass
class PerFaultMetrics:
    fault: FaultType
    tp: int = 0  # faulty batch of this type, correctly flagged for this type
    fn: int = 0  # faulty batch of this type, missed
    fp: int = 0  # non-fault batch flagged for this type

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (p == p) or not (r == r) or (p + r) == 0:  # nan-safe
            return float("nan")
        return 2 * p * r / (p + r)


@dataclass
class EvalResult:
    per_fault: Dict[FaultType, PerFaultMetrics]
    overall_precision: float
    overall_recall: float
    n_batches: int
    n_faulty: int
    n_clean: int
    batch_size: int

    def summary_lines(self) -> List[str]:
        lines = [
            f"Evaluation on real-data slices with injected faults "
            f"({self.n_batches} batches of ~{self.batch_size:,} rows: "
            f"{self.n_faulty} faulty, {self.n_clean} clean control)",
            "",
            f"{'Fault':<22}{'Precision':>10}{'Recall':>9}{'F1':>7}",
            "-" * 48,
        ]
        for fault, m in self.per_fault.items():
            def fmt(x):
                return "  n/a" if x != x else f"{x:6.2f}"
            lines.append(
                f"{fault.value:<22}{fmt(m.precision):>10}"
                f"{fmt(m.recall):>9}{fmt(m.f1):>7}"
            )
        lines.append("-" * 48)
        lines.append(
            f"{'any-fault vs clean':<22}"
            f"{self.overall_precision:>10.2f}{self.overall_recall:>9.2f}"
        )
        return lines


def _learn_baseline(df: pd.DataFrame, batch_size: int, seed: int) -> AnomalyDetector:
    """Learn a known-good baseline from a clean sample of the real data."""
    clean = make_clean_slice(df)
    base_sample = clean.sample(n=min(batch_size, len(clean)), random_state=seed)
    meta = SchemaDiscovery().discover(base_sample.reset_index(drop=True), "online_retail_clean")
    return AnomalyDetector(meta)


def _alert_types(detector: AnomalyDetector, batch: pd.DataFrame) -> set:
    alerts = detector.detect(batch)
    return {a.anomaly_type.value for a in alerts}


def evaluate_detector(
    df: pd.DataFrame,
    batch_size: int = 5_000,
    reps: int = 5,
    seed: int = 42,
) -> EvalResult:
    """Build labelled batches, run the detector, and compute precision/recall."""
    detector = _learn_baseline(df, batch_size, seed=seed - 1)
    batches: List[InjectedBatch] = build_evaluation_batches(
        df, batch_size=batch_size, reps=reps, seed=seed
    )

    per_fault: Dict[FaultType, PerFaultMetrics] = {
        f: PerFaultMetrics(fault=f) for f in FAULT_TO_ALERT_TYPES
    }

    overall_tp = overall_fn = overall_fp = overall_tn = 0

    for batch in batches:
        fired = _alert_types(detector, batch.df)

        # Per-fault scoring: for each fault type, was this batch flagged for it?
        for fault, target_types in FAULT_TO_ALERT_TYPES.items():
            flagged_for_fault = bool(fired & target_types)
            if batch.fault == fault:
                if flagged_for_fault:
                    per_fault[fault].tp += 1
                else:
                    per_fault[fault].fn += 1
            else:
                if flagged_for_fault:
                    per_fault[fault].fp += 1

        # Overall any-fault vs clean.
        any_alert = len(fired) > 0
        if batch.is_faulty:
            if any_alert:
                overall_tp += 1
            else:
                overall_fn += 1
        else:
            if any_alert:
                overall_fp += 1
            else:
                overall_tn += 1

    overall_precision = (
        overall_tp / (overall_tp + overall_fp)
        if (overall_tp + overall_fp)
        else float("nan")
    )
    overall_recall = (
        overall_tp / (overall_tp + overall_fn)
        if (overall_tp + overall_fn)
        else float("nan")
    )

    n_faulty = sum(1 for b in batches if b.is_faulty)
    return EvalResult(
        per_fault=per_fault,
        overall_precision=overall_precision,
        overall_recall=overall_recall,
        n_batches=len(batches),
        n_faulty=n_faulty,
        n_clean=len(batches) - n_faulty,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    from .real_data import load_online_retail

    df = load_online_retail()
    result = evaluate_detector(df)
    print("\n".join(result.summary_lines()))
