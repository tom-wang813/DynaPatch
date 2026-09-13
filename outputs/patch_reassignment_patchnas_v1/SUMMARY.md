# patch_reassignment_patchnas_v1

Decision: **INCONCLUSIVE**

| condition | median RR | original − null | W/T/L | original > null 97.5% | eligible | moved | macro RR | cov@.5 | margin gain | flip rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 0.5465 | +0.0000 | 0/36/0 | — | 1.000 | 0.000 | 0.3515 | 0.7947 | 13.4627 | 0.9709 |
| global_effect_shuffle | 0.1821 | +0.3261 | 36/0/0 | 29 | 1.000 | 0.991 | 0.1451 | 0.1025 | 4.3812 | 0.7562 |
| global_direction_shuffle | 0.1801 | +0.3271 | 36/0/0 | 30 | 1.000 | 0.991 | 0.1422 | 0.0858 | 4.4615 | 0.7652 |
| global_magnitude_shuffle | 0.5281 | +0.0106 | 28/3/5 | 8 | 1.000 | 0.991 | 0.3340 | 0.7809 | 14.8210 | 0.9438 |
| within_true_class_shuffle | 0.4911 | +0.0387 | 30/1/5 | 14 | 0.831 | 0.643 | 0.3349 | 0.7468 | 12.1979 | 0.9233 |
| within_failure_type_shuffle | 0.5214 | +0.0091 | 26/6/4 | 8 | 0.628 | 0.453 | 0.3475 | 0.7816 | 13.4627 | 0.9664 |

The grouped shuffles retain singleton groups. Read them together with `eligible` and `moved`; they diagnose granularity rather than provide a fair global null.

## Eligible-restricted (singleton groups excluded)

Same comparison, but BOTH `original` and the shuffled draws are restricted to only the rows whose group had >1 member and so were actually reassigned to a DIFFERENT failure's patch -- a singleton group's one row is shuffled to itself (a no-op), which silently dilutes the table above toward "no effect" regardless of whether direction matters within a group. Answers: does direction matter WITHIN a failure type, specifically on the failures where that question was actually put to the test?

| condition | cells w/ eligible rows | median RR (eligible) | original − null (eligible) | W/T/L (eligible) | original > null 97.5% (eligible) |
|---|---:|---:|---:|---:|---:|
| original | 36 | 0.5465 | +0.0000 | 0/36/0 | — |
| global_effect_shuffle | 36 | 0.1821 | +0.3261 | 36/0/0 | 29 |
| global_direction_shuffle | 36 | 0.1801 | +0.3271 | 36/0/0 | 30 |
| global_magnitude_shuffle | 36 | 0.5281 | +0.0106 | 28/3/5 | 8 |
| within_true_class_shuffle | 36 | 0.5756 | +0.0502 | 30/1/5 | 14 |
| within_failure_type_shuffle | 36 | 0.6730 | +0.0120 | 26/6/4 | 8 |

Maximum reconstruction error: `0.000e+00`.
