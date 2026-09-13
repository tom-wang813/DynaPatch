# patch_reassignment_v1

Decision: **SUPPORT**

| condition | median RR | original − null | W/T/L | original > null 97.5% | eligible | moved | macro RR | cov@.5 | margin gain | flip rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 0.5032 | +0.0000 | 0/36/0 | — | 1.000 | 0.000 | 0.3482 | 0.6583 | 4.7962 | 0.6219 |
| global_effect_shuffle | 0.1623 | +0.3279 | 36/0/0 | 35 | 1.000 | 0.992 | 0.1252 | 0.0331 | 0.9158 | 0.2570 |
| global_direction_shuffle | 0.1641 | +0.3370 | 36/0/0 | 35 | 1.000 | 0.992 | 0.1255 | 0.0633 | 1.0457 | 0.2672 |
| global_magnitude_shuffle | 0.4578 | +0.0216 | 29/0/7 | 10 | 1.000 | 0.991 | 0.3289 | 0.6035 | 4.5397 | 0.5894 |
| within_true_class_shuffle | 0.4496 | +0.0320 | 34/0/2 | 12 | 0.831 | 0.644 | 0.3252 | 0.5408 | 4.5007 | 0.5807 |
| within_failure_type_shuffle | 0.4970 | +0.0089 | 25/6/5 | 1 | 0.628 | 0.452 | 0.3473 | 0.6491 | 4.7962 | 0.6230 |

The grouped shuffles retain singleton groups. Read them together with `eligible` and `moved`; they diagnose granularity rather than provide a fair global null.

## Eligible-restricted (singleton groups excluded)

Same comparison, but BOTH `original` and the shuffled draws are restricted to only the rows whose group had >1 member and so were actually reassigned to a DIFFERENT failure's patch -- a singleton group's one row is shuffled to itself (a no-op), which silently dilutes the table above toward "no effect" regardless of whether direction matters within a group. Answers: does direction matter WITHIN a failure type, specifically on the failures where that question was actually put to the test?

| condition | cells w/ eligible rows | median RR (eligible) | original − null (eligible) | W/T/L (eligible) | original > null 97.5% (eligible) |
|---|---:|---:|---:|---:|---:|
| original | 36 | 0.5032 | +0.0000 | 0/36/0 | — |
| global_effect_shuffle | 36 | 0.1623 | +0.3279 | 36/0/0 | 35 |
| global_direction_shuffle | 36 | 0.1641 | +0.3370 | 36/0/0 | 35 |
| global_magnitude_shuffle | 36 | 0.4578 | +0.0216 | 29/0/7 | 10 |
| within_true_class_shuffle | 36 | 0.5214 | +0.0415 | 34/0/2 | 12 |
| within_failure_type_shuffle | 36 | 0.6136 | +0.0146 | 25/6/5 | 1 |

Maximum reconstruction error: `0.000e+00`.
