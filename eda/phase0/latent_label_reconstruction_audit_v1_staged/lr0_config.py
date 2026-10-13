"""LR0 configuration - latent label reconstruction integrity audit.

Two facts make this possible.

The main row_id is chronological: 0.6642 of consecutive
row pairs form a valid within-plate-appearance transition, against ~0.664
expected at a mean plate appearance of 3.9 pitches.

The data profile established that asof_pitcher_n is career-cumulative:
100.0% of pitcher-seasons start at or above the previous season's maximum,
and the 35.044% that start at zero equal 792 debut seasons out of 2,260
pitcher-seasons exactly.

Therefore, for consecutive rows of the same pitcher,

    count_i = asof_rate_i * asof_n_i
    outcome_i = count_{i+1} - count_i

recovers the per-pitch outcome of every counter the organisers exposed as a
cumulative rate.  The reconstruction is self-validating: the success counter
must reproduce the official `control_success` column exactly.

The organizers allow future rows of the training file to be used to refine
training targets.  This is training-only: the same reconstruction is not valid
on evaluation rows, and model inputs stay row-local.  Unlike an estimated
latent variable, the failure types are observed here.

Not named config.py - the strong pipeline's modules import `config`.
"""
from __future__ import annotations

ROOT = "."
TRAIN_PATH = f"{ROOT}/data/train.csv"
RUNS_DIR = f"{ROOT}/runs"
RUN_PREFIX = "latent_label_reconstruction_audit_v1_staged"
STAGE_NAME = "stage1_reconstruction_integrity"

SEASON = "season"
TARGET = "control_success"
PITCHER = "pitcher_id"
BATTER = "batter_id"

# entity -> (count column, {label: rate column})
COUNTERS = {
    "pitcher": (
        "asof_pitcher_n",
        {"success": "asof_pitcher_success_rate",
         "reverse": "asof_pitcher_reverse_rate",
         "middle":  "asof_pitcher_middle_rate",
         "ball":    "asof_pitcher_ball_rate",
         "strike":  "asof_pitcher_strike_rate"},
    ),
    "pitchmix": (
        "asof_pitcher_pitchmix_n",
        {"fastball": "asof_pitcher_fastball_rate",
         "breaking": "asof_pitcher_breaking_rate",
         "offspeed": "asof_pitcher_offspeed_rate"},
    ),
    "batter": (
        "asof_batter_n",
        {"bsuccess": "asof_batter_success_rate",
         "bmiddle":  "asof_batter_middle_rate"},
    ),
}
ENTITY_KEY = {"pitcher": PITCHER, "pitchmix": PITCHER, "batter": BATTER}

# round(rate * n) must land on an integer. With n up to 15,449 a CSV stored
# to 6 decimals leaves at most 7.7e-3 of drift, which is safe; 4 decimals
# would leave 0.77 and break. Measured, not assumed.
ROUND_TOLERANCE = 0.02

# Pass requires the reconstruction to reproduce the official target and to
# cover essentially every row.
GATE_MIN_SUCCESS_AGREEMENT = 0.999
GATE_MIN_RECOVERABLE = 0.95

REF_ASSUMED = 0.2498
