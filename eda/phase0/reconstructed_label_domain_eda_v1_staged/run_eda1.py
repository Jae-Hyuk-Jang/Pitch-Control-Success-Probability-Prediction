# EDA1 - domain EDA on LR0 reconstructed per-pitch labels
#
# Question: how does the FAILURE MODE (reverse vs middle), the axis-B outcome
# (ball/strike) and the pitch-group policy vary with pre-pitch context?
# Output: rate tables per axis + a between-cell variance decomposition that
# ranks every axis by how much success-rate variance it could justify.
#
# CPU only, ~3-5 min. Writes runs/reconstructed_label_domain_eda_v1_staged_<stamp>/

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
REF = 0.24963554  # holdout reference Brier

STAMP = time.strftime('%Y%m%d_%H%M%S')
OUT = f'{ROOT}/runs/reconstructed_label_domain_eda_v1_staged_{STAMP}'
os.makedirs(OUT, exist_ok=True)
print('OUT_EDA1=' + OUT, flush=True)

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
print(f'train {train.shape} in {time.time()-t0:.0f}s', flush=True)

labels = pd.read_parquet(LR0)
print(f'labels {labels.shape} cols={list(labels.columns)}', flush=True)

# ---- align labels to train (merge on row_id when available, else positional)
if 'row_id' in labels.columns:
    df = train.merge(labels, on='row_id', how='left', validate='one_to_one')
else:
    assert len(labels) == len(train), 'positional align needs equal length'
    df = pd.concat([train.reset_index(drop=True),
                    labels.reset_index(drop=True)], axis=1)


def find_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

L = {
    'success':  find_col(['lab_success', 'success', 'y_success']),
    'reverse':  find_col(['lab_reverse', 'reverse', 'y_reverse']),
    'middle':   find_col(['lab_middle', 'middle', 'y_middle']),
    'ball':     find_col(['lab_ball', 'ball', 'y_ball']),
    'strike':   find_col(['lab_strike', 'strike', 'y_strike']),
    'fastball': find_col(['lab_fastball', 'fastball', 'y_fastball']),
    'breaking': find_col(['lab_breaking', 'breaking', 'y_breaking']),
    'offspeed': find_col(['lab_offspeed', 'offspeed', 'y_offspeed']),
}
print('label mapping:', json.dumps(L), flush=True)
YCOLS = [v for k, v in L.items() if v is not None and k in
         ('success', 'reverse', 'middle', 'ball', 'strike')]

# ---- integrity anchor: reconstructed success must equal official target
if L['success']:
    m = df[L['success']].notna()
    anchor = float((df.loc[m, L['success']] == df.loc[m, 'control_success']).mean())
    print(f'anchor success==control_success: {anchor:.8f} on {int(m.sum())} rows',
          flush=True)

# ---- derived context columns
df['count_state'] = (df['balls_before'].astype(int).astype(str) + '-'
                     + df['strikes_before'].astype(int).astype(str))
df['platoon'] = ('P' + df['pitcher_hand'].astype(str)
                 + '_B' + df['batter_hand'].astype(str))
df['inning_bucket'] = pd.cut(df['inning'], [0, 3, 6, 99],
                             labels=['inn1-3', 'inn4-6', 'inn7+'])
df['li_q'] = pd.qcut(df['li'], 4, duplicates='drop')
df['score_bucket'] = df['score_diff_pitcher_team'].clip(-3, 3)
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

if L['fastball'] and L['breaking'] and L['offspeed']:
    df['pitch_group'] = np.select(
        [df[L['fastball']] == 1, df[L['breaking']] == 1, df[L['offspeed']] == 1],
        ['fastball', 'breaking', 'offspeed'], default=None)
else:
    df['pitch_group'] = None

fail = df[L['success']] == 0
df['rev_given_fail'] = np.where(fail, df[L['reverse']], np.nan)
df['mid_given_fail'] = np.where(fail, df[L['middle']], np.nan)


# ---- helpers
def rate_table(d, by, ycols, min_n=200):
    g = d.groupby(by, observed=True, dropna=False)
    out = g.size().rename('n').to_frame()
    for y in ycols:
        out[y] = g[y].mean()
    out = out[out['n'] >= min_n]
    return out.reset_index()


def between_var(d, key, y):
    s = d.dropna(subset=[y] if isinstance(key, str) else [y])
    g = s.groupby(key, observed=True)[y]
    n = g.count().astype(float)
    mu = g.mean()
    N = n.sum()
    if N == 0 or len(n) < 2:
        return 0.0, int(len(n)), float('nan')
    gm = float((n * mu).sum() / N)
    bv = float((n * (mu - gm) ** 2).sum() / N)
    return bv, int(len(n)), gm


AXES = [
    ('count_state', 'count_state'),
    ('pitch_group_POSTPITCH', 'pitch_group'),
    ('platoon', 'platoon'),
    ('base_state', 'base_state'),
    ('outs', 'outs_before'),
    ('inning_bucket', 'inning_bucket'),
    ('li_quartile', 'li_q'),
    ('score_bucket', 'score_bucket'),
    ('season', 'season'),
    ('game_type', 'game_type'),
    ('pitcher_decile', 'p_decile'),
    ('pitcher_coldstart', 'p_cold'),
    ('count_x_pitchgroup_POSTPITCH', ['count_state', 'pitch_group']),
    ('count_x_platoon', ['count_state', 'platoon']),
    ('count_x_pitcher_decile', ['count_state', 'p_decile']),
    ('platoon_x_pitcher_decile', ['platoon', 'p_decile']),
    ('count_x_pballdecile', ['count_state', 'pball_decile']),
]

# ---- main rate tables
tables = {
    'A1_count': ('count_state', None),
    'A3_platoon': ('platoon', None),
    'A4_pitcher_decile': ('p_decile', None),
    'A4b_coldstart': ('p_cold', None),
    'A5_base_state': ('base_state', None),
    'A5b_li': ('li_q', None),
    'A5c_inning': ('inning_bucket', None),
    'A5d_score': ('score_bucket', None),
    'A6_season': ('season', None),
    'A6b_gametype': ('game_type', None),
}
extra = ['rev_given_fail', 'mid_given_fail']
for name, (by, _flt) in tables.items():
    try:
        t = rate_table(df, by, YCOLS + extra)
        t.to_csv(f'{OUT}/{name}.csv', index=False)
        print(f'[{name}] {len(t)} cells', flush=True)
    except Exception as e:
        print(f'[{name}] FAILED {e}', flush=True)

# pitch-group tables (post-pitch observation - usable only via policy models)
try:
    if df['pitch_group'].notna().any():
        t = rate_table(df.dropna(subset=['pitch_group']),
                       ['count_state', 'pitch_group'], YCOLS + extra)
        t.to_csv(f'{OUT}/A2_count_x_pitchgroup.csv', index=False)
        mix = (df.dropna(subset=['pitch_group'])
               .groupby(['count_state', 'pitch_group'], observed=True)
               .size().unstack(fill_value=0))
        mix = (mix.T / mix.sum(axis=1)).T.reset_index()
        mix.to_csv(f'{OUT}/A2b_pitch_policy_by_count.csv', index=False)
        print('[A2/A2b] pitch-group tables written', flush=True)
except Exception as e:
    print(f'[A2] FAILED {e}', flush=True)

# ---- variance decomposition across axes x labels
rows = []
for aname, key in AXES:
    for lname in ('success', 'reverse', 'middle', 'ball', 'strike'):
        y = L[lname]
        if y is None:
            continue
        try:
            bv, k, gm = between_var(df, key, y)
        except Exception as e:
            print(f'[VD {aname}/{lname}] FAILED {e}', flush=True)
            continue
        rows.append({'axis': aname, 'label': lname, 'cells': k,
                     'global_rate': round(gm, 6),
                     'between_var': bv,
                     'bv_x1e4': round(bv * 1e4, 4),
                     'score_potential': round(1e5 * bv / REF, 1)})
vd = pd.DataFrame(rows).sort_values(['label', 'between_var'],
                                    ascending=[True, False])
vd.to_csv(f'{OUT}/VD_variance_decomposition.csv', index=False)
print('\n=== between-cell variance, label=success (score_potential = upper'
      ' bound if fully exploited & calibrated) ===', flush=True)
print(vd[vd['label'] == 'success'].to_string(index=False), flush=True)
print('\n=== label=ball (axis-B probe) ===', flush=True)
print(vd[vd['label'] == 'ball'].head(8).to_string(index=False), flush=True)

# ---- compact findings
lines = ['# EDA1 findings (auto-generated)', '',
         f'run: {STAMP}  rows: {len(df)}  anchor: {anchor:.8f}', '',
         '## success variance by axis (top 10)', '']
top = vd[vd['label'] == 'success'].head(10)
lines += [top.to_string(index=False), '',
          '## failure-mode composition by count', '']
try:
    a1 = pd.read_csv(f'{OUT}/A1_count.csv')
    lines += [a1.to_string(index=False), '']
except Exception:
    pass
lines += ['## notes',
          '- pitch_group axes use the POST-PITCH observed group: usable only',
          '  through policy models P(k|x) or frozen pitcher-level rates.',
          '- rev_given_fail / mid_given_fail are conditional on failure.']
with open(f'{OUT}/findings.md', 'w') as f:
    f.write('\n'.join(lines))

summary = {'stamp': STAMP, 'rows': int(len(df)), 'anchor': anchor,
           'label_mapping': L,
           'top_success_axes': top[['axis', 'bv_x1e4', 'score_potential']]
           .to_dict('records'),
           'runtime_sec': round(time.time() - t0, 1)}
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=1)

print(f'\ndecision=EDA1_COMPLETE runtime={time.time()-t0:.0f}s', flush=True)
print('OUT_EDA1=' + OUT, flush=True)
