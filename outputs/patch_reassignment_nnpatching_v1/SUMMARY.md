# patch_reassignment_nnpatching_v1

Decision: **INCONCLUSIVE**

| condition | median RR | original − null | W/T/L | original > null 97.5% | eligible | moved | macro RR | cov@.5 | margin gain | flip rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 0.5889 | +0.0000 | 0/36/0 | — | 1.000 | 0.000 | 0.4302 | 0.8944 | 11.3567 | 0.9685 |
| global_effect_shuffle | 0.1693 | +0.4337 | 36/0/0 | 29 | 1.000 | 0.991 | 0.1521 | 0.0647 | 1.3208 | 0.6342 |
| global_direction_shuffle | 0.1661 | +0.4367 | 35/0/1 | 30 | 1.000 | 0.992 | 0.1483 | 0.0618 | 1.6571 | 0.6422 |
| global_magnitude_shuffle | 0.5584 | +0.0119 | 25/3/8 | 10 | 1.000 | 0.992 | 0.4074 | 0.8933 | 11.9794 | 0.9275 |
| within_true_class_shuffle | 0.5361 | +0.0319 | 31/3/2 | 19 | 0.831 | 0.644 | 0.4122 | 0.8128 | 10.1096 | 0.9131 |
| within_failure_type_shuffle | 0.5596 | +0.0018 | 21/9/6 | 8 | 0.628 | 0.452 | 0.4136 | 0.8730 | 11.3567 | 0.9580 |

The grouped shuffles retain singleton groups. Read them together with `eligible` and `moved`; they diagnose granularity rather than provide a fair global null.

## Eligible-restricted (singleton groups excluded)

Same comparison, but BOTH `original` and the shuffled draws are restricted to only the rows whose group had >1 member and so were actually reassigned to a DIFFERENT failure's patch -- a singleton group's one row is shuffled to itself (a no-op), which silently dilutes the table above toward "no effect" regardless of whether direction matters within a group. Answers: does direction matter WITHIN a failure type, specifically on the failures where that question was actually put to the test?

| condition | cells w/ eligible rows | median RR (eligible) | original − null (eligible) | W/T/L (eligible) | original > null 97.5% (eligible) |
|---|---:|---:|---:|---:|---:|
| original | 36 | 0.5889 | +0.0000 | 0/36/0 | — |
| global_effect_shuffle | 36 | 0.1693 | +0.4337 | 36/0/0 | 29 |
| global_direction_shuffle | 36 | 0.1661 | +0.4367 | 35/0/1 | 30 |
| global_magnitude_shuffle | 36 | 0.5584 | +0.0119 | 25/3/8 | 10 |
| within_true_class_shuffle | 36 | 0.5924 | +0.0359 | 31/3/2 | 19 |
| within_failure_type_shuffle | 36 | 0.7425 | +0.0024 | 21/9/6 | 8 |

Maximum reconstruction error: `0.000e+00`.
