# DataShield: Honest Metrics

This file reports only measured numbers. Every figure below comes from running
the code in this repo on the real UCI Online Retail dataset. There are no dollar
figures, no "X hours faster than a human", no ROI estimates — those were removed
because nothing in this repo measures them.

Reproduce everything with:

```bash
python scripts/download_data.py     # cache the real dataset (~541,909 rows)
python run_demo.py                  # real profile + injected-fault evaluation
python -m pytest tests/unit tests/integration -q
```

---

## What is real vs. synthetic

| Thing | Real or synthetic | Why |
|---|---|---|
| Base dataset (UCI Online Retail, 541,909 rows) | **Real** | Actual UK online-retailer transactions, 2010-12 to 2011-12. Genuinely messy. |
| Quality issues reported below (nulls, cancellations, negatives, duplicates) | **Real** | Counted directly from the real file. |
| Injected faults used for the precision/recall evaluation | **Synthetic, with ground-truth labels** | You cannot measure a detector without knowing which batches are broken. Base rows are real; the fault is added and labelled. |
| Precision/recall numbers | **Real measurements** | Real detector output scored against the injected-fault labels. |

The base rows are real e-commerce transactions. The faults are injected on top of
clean slices of that real data specifically so the detection metrics can be
measured against known ground truth. That is the standard, honest way to evaluate
a data-quality system.

---

## Real quality issues found in Online Retail

Counted directly from the 541,909-row real dataset (`src/eval/real_data.py`):

| Issue | Value |
|---|---|
| Rows x columns | 541,909 x 8 |
| CustomerID null rate | **24.93%** (135,080 rows) |
| Description null rate | 0.27% (1,454 rows) |
| Cancelled invoices (InvoiceNo starts with 'C') | 9,288 (1.71%) |
| Negative-quantity rows (returns/cancellations) | 10,624 (1.96%) |
| Non-positive UnitPrice rows | 2,517 |
| Exact duplicate rows | 5,268 |
| Quantity range | [-80,995, 80,995] |
| UnitPrice range | [-11,062.06, 38,970.00] |
| Outlier line-amount rows (\|z\|>3 on Quantity x UnitPrice) | 403 |

Running the quality engine on the raw feed (baseline learned from the first 30
days, detection on the full year) fires a `row_count_spike` alert, which is
correct — the full year is ~13x the size of the first month.

---

## Detection precision/recall on injected faults

`run_demo.py` builds 30 evaluation batches (~5,000 rows each): 5 repetitions of
each of 5 fault types plus 5 clean control batches. It learns a baseline from a
clean slice of the real data, then scores the detector against the ground-truth
labels (`src/eval/evaluate.py`).

| Injected fault | Precision | Recall | F1 |
|---|---:|---:|---:|
| null_spike (60% CustomerID nulls) | 1.00 | 1.00 | 1.00 |
| distribution_drift (8x amount, past 3 sigma) | 1.00 | 1.00 | 1.00 |
| schema_type_change (Quantity -> text) | 1.00 | 1.00 | 1.00 |
| pii_injection (emails in Description) | 1.00 | 1.00 | 1.00 |
| cardinality_collapse (all one CustomerID) | 1.00 | 1.00 | 1.00 |
| **any-fault vs clean** | **1.00** | **1.00** | — |

### Be blunt: what these numbers do and do not show

- They show the detection **pipeline is correct**: every injected fault type is
  caught, and clean control batches are not false-flagged. That is a real,
  non-trivial result — before this work the engine could not catch a
  numeric-to-text type change at all (a dedicated type-drift check was added to
  `AnomalyDetector` to fix that).
- They **do not** claim the detector is hard. The injected faults are large and
  unambiguous (60% nulls, 8x drift). Perfect scores here mean the faults are
  clearly out of distribution, not that the detector would catch subtle
  real-world corruption.
- Known threshold characteristics, measured while building this:
  - **Null check is trivially sensitive when the baseline is 0% null.** The rule
    is "alert if null rate more than doubles"; doubling 0 is still ~0, so any
    nonzero null trips it. Good for catching a clean column going bad, but it
    would be noisy on a column that is normally slightly null.
  - **Distribution-shift check is insensitive on heavy-tailed real data.** Online
    Retail line-amounts have a huge standard deviation (~$103), so the >3-sigma
    rule needs a very large mean shift to fire. A 3x mean shift was **not**
    caught in testing; it took ~8x. On long-tailed real distributions this check
    is conservative.

These are honest limitations of simple statistical thresholds, not bugs.

---

## Test suite

`tests/unit` + `tests/integration` run with no infrastructure:

```
26 passed
```

This includes 4 new tests in `tests/unit/test_eval_injected_faults.py` that
exercise the real-data eval harness (they skip cleanly if the dataset cache is
absent). The `tests/load` and `tests/chaos` suites require running services and
are not part of this count.

---

## Lineage / blast radius

The blast-radius BFS is real and runs in-process on the in-memory lineage graph
(`src/lineage`). Its timing on the small demo graph is sub-millisecond, but the
headline "scales to 100K tables at N ms" figures from earlier drafts were
extrapolations, not load-tested guarantees, and have been removed. Run
`tests/load/load_test_100k_tables.py` to measure on your own hardware.
