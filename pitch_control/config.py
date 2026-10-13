"""Column groups used by the feature graph."""
from __future__ import annotations

TARGET='control_success'
ROW_ID='row_id'

BASE_CATEGORICAL=[
    'top_bottom','game_type','base_state','pitcher_id','batter_id',
    'pitcher_hand','batter_hand','pitcher_team_id','batter_team_id',
]
ENGINEERED_CATEGORICAL=[
    'count_state','base_out_state','inning_phase','platoon_matchup',
    'pressure_state','score_state','runners_state','cold_start_state',
    'recent_state','game_type_x_count',
]
RATE_FEATURES=[
    'asof_pitcher_success_rate','asof_pitcher_reverse_rate',
    'asof_pitcher_middle_rate','asof_pitcher_ball_rate',
    'asof_pitcher_strike_rate','asof_pitcher_prev1_game_success_rate',
    'asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate',
    'asof_pitcher_prev1_game_middle_rate','asof_pitcher_prev3_game_middle_rate',
    'asof_pitcher_prev5_game_middle_rate','asof_batter_success_rate',
    'asof_batter_middle_rate','asof_pitcher_fastball_rate',
    'asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate',
]
