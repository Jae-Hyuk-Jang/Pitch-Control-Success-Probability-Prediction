# EDA2 - domain EDA on LR0 reconstructed labels (PROBE0-confirmed schema)
#
# 스키마 점검(PROBE0)으로 확정한 사실:
#   - 10 cols lab_* float32, row_id 없음 -> train.csv와 위치 정렬 (anchor 1.0 검증)
#   - NaN 1,584행(0.107%) = 미복원 행 -> 분석에서 제외
#   - lab_bsuccess ≡ lab_success, lab_bmiddle ≡ lab_middle (완전 중복) -> 제외
#   - axis A: success와 (reverse|middle) 배타. reverse∩middle 50,185행.
#     success=0 & reverse=0 & middle=0 인 'other_fail' 193,963행(13.2%) 존재
#   - ball/strike 배타, 나머지는 in-play (18.7%) -> lab_inplay 파생
#   - 구종군 3분할 완전 (합=1)
#
# CPU ~3-5분. 출력: runs/reconstructed_label_domain_eda_v2_<stamp>/

import json
import os
import time

import numpy as np
import pandas as pd

ROOT = '.'
def _latest(pattern):
    """Newest run folder (or file) matching the glob; the unit that writes it must have been run first."""
    import glob
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else pattern


LR0 = _latest(ROOT + '/runs/latent_label_reconstruction_audit_v1_staged_*'
             '/stage1_reconstruction_integrity/reconstructed_labels.parquet')
REF = 0.24963554

STAMP = time.strftime('%Y%m%d_%H%M%S')
OUT = f'{ROOT}/runs/reconstructed_label_domain_eda_v2_{STAMP}'
os.makedirs(OUT, exist_ok=True)
print('OUT_EDA2=' + OUT, flush=True)
t0 = time.time()

TRAIN_COLS = [
    'row_id', 'season', 'game_type', 'inning', 'top_bottom',
    'balls_before', 'strikes_before', 'outs_before', 'base_state',
    'num_runners_on', 'li', 'score_diff_pitcher_team',
    'pitcher_id', 'batter_id', 'pitcher_hand', 'batter_hand',
    'asof_pitcher_n', 'asof_pitcher_success_rate', 'asof_pitcher_reverse_rate',
    'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate',
    'asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate',
    'asof_pitcher_offspeed_rate', 'asof_batter_n', 'asof_batter_success_rate',
    'control_success',
]
train = pd.read_csv(ROOT + '/data/train.csv', usecols=TRAIN_COLS)
lab = pd.read_parquet(LR0, columns=[
    'lab_success', 'lab_reverse', 'lab_middle', 'lab_ball', 'lab_strike',
    'lab_fastball', 'lab_breaking', 'lab_offspeed'])
assert len(train) == len(lab) == 1475092
df = pd.concat([train.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
print(f'loaded {df.shape} in {time.time()-t0:.0f}s', flush=True)

m = df['lab_success'].notna()
anchor = float((df.loc[m, 'lab_success'] == df.loc[m, 'control_success']).mean())
print(f'anchor={anchor:.8f} labelled={int(m.sum())}', flush=True)
assert anchor > 0.9999
df = df[m].copy()

# ---- derived labels
df['lab_inplay'] = 1.0 - df['lab_ball'] - df['lab_strike']
df['axisA'] = np.select(
    [df['lab_success'] == 1,
     (df['lab_reverse'] == 1) & (df['lab_middle'] == 1),
     df['lab_reverse'] == 1,
     df['lab_middle'] == 1],
    ['success', 'rev_mid', 'reverse', 'middle'], default='other_fail')
df['pitch_group'] = np.select(
    [df['lab_fastball'] == 1, df['lab_breaking'] == 1, df['lab_offspeed'] == 1],
    ['fastball', 'breaking', 'offspeed'], default='NA')

fail = df['lab_success'] == 0
for src, dst in (('lab_reverse', 'rev_gf'), ('lab_middle', 'mid_gf')):
    df[dst] = np.where(fail, df[src], np.nan)
df['oth_gf'] = np.where(fail, (df['axisA'] == 'other_fail').astype(float), np.nan)

# ---- derived context
df['count_state'] = (df['balls_before'].astype(int).astype(str) + '-'
                     + df['strikes_before'].astype(int).astype(str))
df['platoon'] = ('P' + df['pitcher_hand'].astype(str)
                 + '_B' + df['batter_hand'].astype(str))
df['inning_bucket'] = pd.cut(df['inning'], [0, 3, 6, 99],
                             labels=['inn1-3', 'inn4-6', 'inn7+'])
df['li_q'] = pd.qcut(df['li'], 4, labels=['li_q1', 'li_q2', 'li_q3', 'li_q4'],
                     duplicates='drop')
df['score_bucket'] = df['score_diff_pitcher_team'].clip(-3, 3).astype(int)
df['p_cold'] = pd.cut(df['asof_pitcher_n'].fillna(0),
                      [-1, 0, 99, 999, 9999, 10**9],
                      labels=['n0', 'n1-99', 'n100-999', 'n1k-10k', 'n10k+'])
warm = df['asof_pitcher_n'] >= 300
df['p_decile'] = np.nan
df.loc[warm, 'p_decile'] = pd.qcut(
    df.loc[warm, 'asof_pitcher_success_rate'], 10, labels=False, duplicates='drop')
df['pball_decile'] = np.nan
df.loc[warm, 'pball_decile'] = pd.qcut(
    df.loc[warm, 'asof_pitcher_ball_rate'], 10, labels=False, duplicates='drop')

YC = ['lab_success', 'lab_reverse', 'lab_middle', 'lab_ball', 'lab_strike',
      'lab_inplay']
GF = ['rev_gf', 'mid_gf', 'oth_gf']


def rate_table(d, by, min_n=200):
    g = d.groupby(by, observed=True, dropna=False)
    out = g.size().rename('n').to_frame()
    for y in YC + GF:
        out[y] = g[y].mean()
    return out[out['n'] >= min_n].round(5).reset_index()


def between_var(d, key, y):
    g = d.dropna(subset=[y]).groupby(key, observed=True)[y]
    n = g.count().astype(float)
    mu = g.mean()
    N = n.sum()
    if N == 0 or len(n) < 2:
        return 0.0, int(len(n)), float('nan')
    gm = float((n * mu).sum() / N)
    return float((n * (mu - gm) ** 2).sum() / N), int(len(n)), gm


# ---- A0: axis-A composition overall
df['axisA'].value_counts(normalize=True).round(5).to_csv(f'{OUT}/A0_axisA.csv')
print('\naxisA composition:', flush=True)
print(df['axisA'].value_counts(normalize=True).round(4).to_string(), flush=True)

# ---- rate tables
TABLES = {
    'A1_count': 'count_state',
    'A3_platoon': 'platoon',
    'A3b_count_x_platoon': ['count_state', 'platoon'],
    'A4_pitcher_decile': 'p_decile',
    'A4b_coldstart': 'p_cold',
    'A5_base_state': 'base_state',
    'A5b_li': 'li_q',
    'A5c_inning': 'inning_bucket',
    'A5d_score': 'score_bucket',
    'A5e_outs': 'outs_before',
    'A6_season': 'season',
    'A6b_gametype': 'game_type',
    'A2_count_x_pitchgroup': ['count_state', 'pitch_group'],
    'A7_count_x_pballdecile': ['count_state', 'pball_decile'],
}
for name, by in TABLES.items():
    try:
        t = rate_table(df, by)
        t.to_csv(f'{OUT}/{name}.csv', index=False)
        print(f'[{name}] {len(t)} cells', flush=True)
    except Exception as e:
        print(f'[{name}] FAILED {e}', flush=True)

# pitch policy by count (what gets thrown when)
mix = (df[df['pitch_group'] != 'NA']
       .groupby(['count_state', 'pitch_group'], observed=True).size()
       .unstack(fill_value=0))
mix = (mix.T / mix.sum(axis=1)).T.round(4)
mix.to_csv(f'{OUT}/A2b_pitch_policy_by_count.csv')
print('\npitch policy by count:', flush=True)
print(mix.to_string(), flush=True)

# ---- variance decomposition
AXES = [
    ('count_state', 'count_state'), ('pitch_group_POST', 'pitch_group'),
    ('platoon', 'platoon'), ('base_state', 'base_state'),
    ('outs', 'outs_before'), ('inning_bucket', 'inning_bucket'),
    ('li_quartile', 'li_q'), ('score_bucket', 'score_bucket'),
    ('season', 'season'), ('game_type', 'game_type'),
    ('pitcher_decile', 'p_decile'), ('pitcher_coldstart', 'p_cold'),
    ('pitcher_id_RAW', 'pitcher_id'), ('batter_id_RAW', 'batter_id'),
    ('count_x_pitchgroup_POST', ['count_state', 'pitch_group']),
    ('count_x_platoon', ['count_state', 'platoon']),
    ('count_x_pdecile', ['count_state', 'p_decile']),
    ('platoon_x_pdecile', ['platoon', 'p_decile']),
    ('count_x_pballdecile', ['count_state', 'pball_decile']),
    ('season_x_count', ['season', 'count_state']),
]
rows = []
for aname, key in AXES:
    for y in YC:
        try:
            bv, k, gm = between_var(df, key, y)
            rows.append({'axis': aname, 'label': y.replace('lab_', ''),
                         'cells': k, 'global_rate': round(gm, 5),
                         'bv_x1e4': round(bv * 1e4, 4),
                         'score_potential': round(1e5 * bv / REF, 1)})
        except Exception as e:
            print(f'[VD {aname}/{y}] FAILED {e}', flush=True)
vd = pd.DataFrame(rows).sort_values(['label', 'bv_x1e4'], ascending=[True, False])
vd.to_csv(f'{OUT}/VD_variance_decomposition.csv', index=False)
for lbl in ('success', 'ball', 'middle', 'reverse'):
    print(f'\n=== between-cell variance: {lbl} ===', flush=True)
    print(vd[vd['label'] == lbl].head(12).to_string(index=False), flush=True)

# ---- key stdout tables for review
print('\n=== A1 count table ===', flush=True)
print(pd.read_csv(f'{OUT}/A1_count.csv').to_string(index=False), flush=True)
print('\n=== A4 pitcher decile ===', flush=True)
print(pd.read_csv(f'{OUT}/A4_pitcher_decile.csv').to_string(index=False),
      flush=True)
print('\n=== A6 season ===', flush=True)
print(pd.read_csv(f'{OUT}/A6_season.csv').to_string(index=False), flush=True)

summary = {'stamp': STAMP, 'rows': int(len(df)), 'anchor': anchor,
           'axisA': df['axisA'].value_counts(normalize=True).round(5).to_dict(),
           'top_success_axes': vd[vd['label'] == 'success'].head(12)
           [['axis', 'cells', 'bv_x1e4', 'score_potential']].to_dict('records'),
           'runtime_sec': round(time.time() - t0, 1)}
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=1)
with open(f'{OUT}/findings.md', 'w') as f:
    f.write('# EDA2 findings\n\n' + json.dumps(summary, indent=1))

print(f'\ndecision=EDA2_COMPLETE runtime={time.time()-t0:.0f}s', flush=True)
print('OUT_EDA2=' + OUT, flush=True)
