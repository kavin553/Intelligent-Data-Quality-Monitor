"""
Real-data evaluation for DataShield.

This package takes the REAL, messy UCI Online Retail dataset and does two things:

  real_data.py      profile the real data and run the quality engine over it,
                    reporting the actual quality issues it surfaces.

  fault_injection.py  take a clean slice of the real data and inject KNOWN,
                      labelled faults (null spikes, distribution drift, a
                      type/schema change, injected PII, cardinality collapse).
                      These copies are synthetic-with-ground-truth: the only
                      way to measure a detector's precision/recall is to know
                      exactly which batches are broken and how.

  evaluate.py       run the detectors over the injected-fault batches and
                    compute real precision/recall against that ground truth.
"""

from .real_data import load_online_retail, profile_real_data, run_quality_engine
from .fault_injection import (
    FaultType,
    InjectedBatch,
    make_clean_slice,
    build_evaluation_batches,
)
from .evaluate import evaluate_detector, EvalResult

__all__ = [
    "load_online_retail",
    "profile_real_data",
    "run_quality_engine",
    "FaultType",
    "InjectedBatch",
    "make_clean_slice",
    "build_evaluation_batches",
    "evaluate_detector",
    "EvalResult",
]
