# Exploratory data analysis

Scripts for looking at the training data. Run them from the project root with the competition files in `data/`
(git-ignored). Outputs go to `runs/` (git-ignored); nothing here uses evaluation data (two scripts read the five-row `test.csv`
schema sample only to check the schema).

| Script | Question it answers |
| --- | --- |
| [`from_scratch_eda_v1.py`](from_scratch_eda_v1.py) | Baseline profile: target by season, count, platoon, inning phase, leverage; numeric and categorical association with the target |
| [`visual_eda_atlas_v1.py`](visual_eda_atlas_v1.py) | The aggregate figures used in the README: season drift, season x count heat map, as-of reliability curves, cold start, pitcher profiles, pitch-mix atlas |
| [`full_data_information_audit_v2.py`](full_data_information_audit_v2.py) | What information is in the files: schema and contract checks, ordering of the training file, how the running as-of columns relate to the labels |
| [`year_relationship_identity_continuity_audit_v1.py`](year_relationship_identity_continuity_audit_v1.py) | How similar the seasons are to each other, and how persistent pitcher and batter identities, teams and leagues are across seasons |

## phase0/: the first week

Written 2026-08-05 to 08-11; the numbered `E*` units answer one question each.

| Unit | Question it answers |
| --- | --- |
| [`latent_label_reconstruction_audit_v1_staged`](phase0/latent_label_reconstruction_audit_v1_staged) | LR0: can the per-pitch labels (success, reverse, middle, ball, strike) be recovered inside the training file from the change of the running as-of columns between consecutive rows? Run first: the units below read its output |
| [`full_data_profile_e0_staged`](phase0/full_data_profile_e0_staged) | E0: profile of every column of `train.csv` and `trackman_history.csv`: missingness, distributions, association with the target, season drift |
| [`pitcher_deep_profile_e1_staged`](phase0/pitcher_deep_profile_e1_staged) | E1: how much of the pitcher-identity variance is real skill? Split-half reliability, year-to-year persistence, empirical-Bayes shrinkage, cold-start convergence |
| [`drift_regime_audit_e2_staged`](phase0/drift_regime_audit_e2_staged) | E2: drift and regimes in depth, and the risk of extrapolating to the next season: which features cannot leave their time range |
| [`team_catcher_and_failure_anatomy_e3_staged`](phase0/team_catcher_and_failure_anatomy_e3_staged) | E3: is a "reverse" failure the opposite of the catcher's target (a team-level calling hypothesis), what is hiding in `other_failure`, and what does the batter axis add |
| [`physical_geometry_atlas_e4_staged`](phase0/physical_geometry_atlas_e4_staged) | E4: physical and spatial map of the TrackMan history (release, movement, speed) |
| [`fatigue_workload_rhythm_e5_staged`](phase0/fatigue_workload_rhythm_e5_staged) | E5: fatigue, appearance rhythm and workload from the within-game pitch order in the TrackMan history |
| [`reconstructed_label_domain_eda_v1_staged`](phase0/reconstructed_label_domain_eda_v1_staged) | Domain EDA on the reconstructed labels: failure mode (reverse vs middle) and the outcome axes |

These units are the scripts as they were run, with private paths replaced by project-relative ones. They read each other's
output folders under `runs/` and pick the newest matching run (E1 to E5 and the label EDA read the LR0 output; E4 and E5
also read a TrackMan ID-alignment run that is not included), so they are a record of the analysis rather than a
one-command pipeline.

What the EDA found, in one paragraph: the success rate drifts down year after year (about 56.5% in 2019 to 48.6% in 2024),
the drift is not uniform across counts, the as-of channels are monotone but heavily shrunk predictors, the regular-season (`R`)
and second-team (`F`) populations behave differently and drift differently, and a pitcher's effect persists from one
season to the next. Those four observations are the backbone of the architecture described in
the [main README](../README.md).
