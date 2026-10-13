# Validation

[`row_independence_audit.py`](row_independence_audit.py) is a generic audit for any `predict(frame) -> probabilities`
function: single row vs batch, chunk sizes, permutation, duplicated rows, scrambled unrelated rows, repeat calls. It reports
the maximum difference per check. [`tests/test_row_independence_audit.py`](../tests/test_row_independence_audit.py) exercises it
with row-local predictors (must pass) and four kinds of transductive leakage (must be caught).

The competition required row independence: the value for a row may not change with the other rows in the batch. For the
CatBoost-only graph the maximum difference is exactly 0; with float32 torch members it is around 1e-8, which is why the
tolerance is a parameter.
