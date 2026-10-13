# Fitting recipes

The two central fitting recipes of the solution, stated compactly. The code that fitted the full set of frozen artifacts
is not part of this repository; the inference graph that consumes them is in [`pitch_control/`](../pitch_control).

| File | Recipe |
| --- | --- |
| [`reference/anchor_reference.py`](reference/anchor_reference.py) | The anchor: five-class CatBoost (96 trees, depth 8, learning rate 0.055) with recent-season sample weights, three seeds, probability of the success class, and the final R5 / R8 blend (0.938 / 0.062 after the additive offset) |
| [`reference/rc3_corrector_reference.py`](reference/rc3_corrector_reference.py) | The residual corrector: centred residual target, eight capped CatBoost regressors, and the final combination `clip(p0 + 1.2 * c_b - 0.3 * c_v)` |

Both files run on synthetic data in [`tests/test_reference_training.py`](../tests/test_reference_training.py).
