# E2 - 드리프트 / regime 심층 + 2025 외삽 위험 정량화
#
# 질문: 2025 평가 데이터에 대해 무엇이 위험한가. 어떤 피처가 시간 밖으로 못 나가는가.
#
#  1) 월 단위 연속 추적 (season x month, 36포인트): 라벨 5종의 시계열 + 구조 단절 탐지
#  2) 규정 변화 시점 검정: 각 인접 시즌 경계에서 라벨 점프 크기 + 월 단위 급변 위치
#  3) 피처별 외삽 위험 점수:
#     - trend_slope: 시즌 인덱스에 대한 선형 추세 (2025 외삽 시 밖으로 나가는 정도)
#     - out_of_range_2025: 2025 선형 외삽값이 학습 관측 범위를 벗어나는가
#     - pit_id_novelty: 각 시즌의 신규 투수 비율 -> 2025 신규 투수 추정
#  4) 투수 구성 회전율: 시즌간 로스터 생존율, 신규/이탈, ID 시간순 발급 검증
#  5) regime별 개인 순위 안정성: 2019-22 vs 2023 vs 2024 구간에서 투수 랭킹 상관
#     (라벨 체계가 바뀌면 개인 순위도 깨지는가 = CE 동결 아티팩트의 유효기간)
#  6) 시간 역행 검증: 과거 N시즌으로 학습 -> 다음 시즌 예측 시 성능 감쇠 곡선
#     (단순 로지스틱, 빠름) -> "학습 창을 얼마나 최근으로 잡아야 하는가"의 EDA 근거
#
# CPU ~5-10분. 출력: runs/drift_regime_audit_e2_<stamp>/

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
STAMP = time.strftime('%Y%m%d_%H%M%S')
OUT = f'{ROOT}/runs/drift_regime_audit_e2_{STAMP}'
os.makedirs(OUT, exist_ok=True)
print('OUT_E2=' + OUT, flush=True)
t0 = time.time()

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8, "axes.titlesize": 9})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=140)
    plt.close(fig)
    print(f"[chart] {name}", flush=True)


tr = pd.read_csv(f'{ROOT}/data/train.csv', low_memory=False)
lab = pd.read_parquet(LR0, columns=['lab_success', 'lab_reverse', 'lab_middle',
                                    'lab_ball', 'lab_strike'])
df = pd.concat([tr.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
df = df[df['lab_success'].notna()].copy()
df['lab_inplay'] = 1.0 - df['lab_ball'] - df['lab_strike']
LBL = ['lab_success', 'lab_reverse', 'lab_middle', 'lab_ball', 'lab_inplay']
print(f'rows={len(df)} in {time.time()-t0:.0f}s', flush=True)

# ---- 1) monthly continuous tracking
mo = (df.groupby(['season', 'game_month'])
      .agg(n=('lab_success', 'size'), **{c: (c, 'mean') for c in LBL})
      .reset_index())
mo = mo[mo.n >= 3000].reset_index(drop=True)
mo['t'] = np.arange(len(mo))
mo['tag'] = mo.season.astype(str) + '-' + mo.game_month.astype(str).str.zfill(2)
mo.to_csv(f'{OUT}/e2_monthly_labels.csv', index=False)

fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
for ax, c, col in zip(axes, LBL, PAL):
    ax.plot(mo.t, mo[c], color=col, lw=2, marker='o', ms=3)
    for s in sorted(df.season.unique())[1:]:
        b = mo[mo.season == s].t.min()
        ax.axvline(b - 0.5, color=INK2, lw=0.8, ls=':')
    ax.set_ylabel(c.replace('lab_', ''))
    ax.set_title(c.replace('lab_', ''), loc='left')
axes[-1].set_xticks(mo.t[::3])
axes[-1].set_xticklabels(mo.tag[::3], rotation=90, fontsize=6)
save(fig, 'e2_monthly_labels.png')

# season-boundary jumps
seas = df.groupby('season')[LBL].mean()
jumps = seas.diff().dropna().round(5)
jumps.to_csv(f'{OUT}/e2_season_jumps.csv')
print('\nseason-to-season label jumps:', flush=True)
print(jumps.to_string(), flush=True)

# largest month-to-month moves per label
moves = {}
for c in LBL:
    d = mo[c].diff().abs()
    i = int(d.iloc[1:].idxmax())
    moves[c.replace('lab_', '')] = dict(
        at=f"{mo.tag[i-1]} -> {mo.tag[i]}", delta=round(float(mo[c][i] - mo[c][i-1]), 5))
print('largest month-to-month move:', json.dumps(moves), flush=True)

# ---- 2) roster turnover / id novelty
pit_by_season = df.groupby('season')['pitcher_id'].apply(set)
seasons = sorted(pit_by_season.index)
turn = []
seen = set()
for s in seasons:
    cur = pit_by_season[s]
    new = cur - seen
    prev = pit_by_season[s - 1] if (s - 1) in pit_by_season.index else set()
    turn.append(dict(season=int(s), pitchers=len(cur),
                     new_vs_all_history=len(new),
                     new_rate=round(len(new) / len(cur), 4),
                     survived_from_prev=len(cur & prev) if prev else None,
                     survival_rate=(round(len(cur & prev) / len(prev), 4)
                                    if prev else None)))
    seen |= cur
turn = pd.DataFrame(turn)
turn.to_csv(f'{OUT}/e2_roster_turnover.csv', index=False)
print('\nroster turnover:', flush=True)
print(turn.to_string(index=False), flush=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(turn.season.astype(str), turn.new_rate, color=PAL[0], label='new pitcher rate')
ax.plot(turn.season.astype(str), turn.survival_rate.fillna(np.nan),
        color=PAL[1], lw=2, marker='o', ms=5, label='survival from prev season')
ax.set_title('pitcher roster turnover (2025 will look like this too)', loc='left')
ax.legend(frameon=False)
save(fig, 'e2_roster_turnover.png')

# id ordering check: does mean pitcher_id rise with debut season?
debut = df.groupby('pitcher_id')['season'].min()
idcheck = pd.DataFrame(dict(pitcher_id=debut.index, debut=debut.values))
gd = idcheck.groupby('debut')['pitcher_id'].agg(['mean', 'min', 'max', 'size'])
print('\npitcher_id by debut season (시간순 발급 검증):', flush=True)
print(gd.round(1).to_string(), flush=True)
id_rank_corr = float(np.corrcoef(idcheck.debut, idcheck.pitcher_id)[0, 1])
print(f'corr(debut_season, pitcher_id) = {id_rank_corr:.4f}', flush=True)

# ---- 3) per-feature extrapolation risk
NUM = [c for c in tr.columns
       if pd.api.types.is_numeric_dtype(tr[c])
       and c not in ('control_success',)]
rows = []
for c in NUM:
    sm = df.groupby('season')[c].mean()
    x = sm.index.values.astype(float)
    sl = float(np.polyfit(x, sm.values, 1)[0])
    sd = float(df[c].std())
    pred2025 = float(np.polyval(np.polyfit(x, sm.values, 1), 2025.0))
    lo, hi = float(sm.min()), float(sm.max())
    rows.append(dict(column=c, slope_per_season=round(sl, 5),
                     slope_in_sd=round(sl / sd, 5) if sd > 0 else np.nan,
                     season_mean_min=round(lo, 5), season_mean_max=round(hi, 5),
                     pred_2025_mean=round(pred2025, 5),
                     outside_observed_range=bool(pred2025 < lo or pred2025 > hi),
                     extrapolation_gap_sd=round((pred2025 - hi) / sd, 5) if sd > 0 else np.nan))
risk = pd.DataFrame(rows)
risk['abs_slope_sd'] = risk.slope_in_sd.abs()
risk = risk.sort_values('abs_slope_sd', ascending=False)
risk.to_csv(f'{OUT}/e2_extrapolation_risk.csv', index=False)
print('\n2025 extrapolation risk (top 15 by |slope| in SD):', flush=True)
print(risk.head(15).drop(columns=['abs_slope_sd']).to_string(index=False), flush=True)

top = risk.head(14)
fig, ax = plt.subplots(figsize=(8, 6))
colors = [PAL[1] if o else PAL[0] for o in top.outside_observed_range[::-1]]
ax.barh(top.column[::-1], top.slope_in_sd[::-1], color=colors)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel('drift slope (SD per season)')
ax.set_title('2025 extrapolation risk  orange = 2025 projection outside observed range',
             loc='left')
save(fig, 'e2_extrapolation_risk.png')

# ---- 4) regime-wise pitcher ranking stability
REG = {'2019-22': [2019, 2020, 2021, 2022], '2023': [2023], '2024': [2024]}
rk = {}
for name, ss in REG.items():
    sub = df[df.season.isin(ss)]
    g = sub.groupby('pitcher_id')
    m = g.size()
    rk[name] = pd.DataFrame({c: g[c].mean() for c in LBL})[m >= 400]
stab = {}
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
for ax, c in zip(axes, LBL):
    pairs = []
    for a, b in (('2019-22', '2023'), ('2023', '2024'), ('2019-22', '2024')):
        j = rk[a].index.intersection(rk[b].index)
        r = float(np.corrcoef(rk[a].loc[j, c], rk[b].loc[j, c])[0, 1])
        pairs.append((f'{a}~{b}', round(r, 4), len(j)))
    stab[c.replace('lab_', '')] = pairs
    ax.bar([p[0] for p in pairs], [p[1] for p in pairs], color=PAL[:3])
    ax.set_ylim(0, 1)
    ax.set_title(c.replace('lab_', ''), loc='left')
    ax.tick_params(axis='x', rotation=30, labelsize=6)
save(fig, 'e2_regime_rank_stability.png')
print('\nregime rank stability:', json.dumps(stab, ensure_ascii=False), flush=True)

# ---- 5) temporal decay: train on <=S, test on S+1..S+k (simple logistic)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATS = ['asof_pitcher_success_rate', 'asof_pitcher_reverse_rate',
         'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate',
         'asof_batter_success_rate', 'balls_before', 'strikes_before',
         'pitcher_hand', 'batter_hand', 'inning']
sub = df[FEATS + ['season', 'control_success']].dropna()
decay = []
for tr_end in (2020, 2021, 2022, 2023):
    trm = sub.season <= tr_end
    sc = StandardScaler().fit(sub.loc[trm, FEATS])
    lr = LogisticRegression(max_iter=200).fit(
        sc.transform(sub.loc[trm, FEATS]), sub.loc[trm, 'control_success'])
    for gap in range(1, 2025 - tr_end):
        s = tr_end + gap
        te = sub.season == s
        if te.sum() < 1000:
            continue
        p = lr.predict_proba(sc.transform(sub.loc[te, FEATS]))[:, 1]
        y = sub.loc[te, 'control_success'].to_numpy()
        r = float(y.mean())
        br = float(np.mean((p - y) ** 2))
        decay.append(dict(train_end=tr_end, test_season=int(s), gap=gap,
                          brier=round(br, 8),
                          bss=round(1e5 * (1 - br / (r * (1 - r))), 1),
                          mean_pred=round(float(p.mean()), 5), actual=round(r, 5),
                          calib_gap=round(float(p.mean()) - r, 5)))
decay = pd.DataFrame(decay)
decay.to_csv(f'{OUT}/e2_temporal_decay.csv', index=False)
print('\ntemporal decay (train<=S -> test S+gap):', flush=True)
print(decay.to_string(index=False), flush=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for (te, grp), col in zip(decay.groupby('train_end'), PAL):
    axes[0].plot(grp.gap, grp.bss, color=col, lw=2, marker='o', ms=4,
                 label=f'train<={te}')
    axes[1].plot(grp.gap, grp.calib_gap, color=col, lw=2, marker='o', ms=4,
                 label=f'train<={te}')
axes[0].set_xlabel('seasons ahead')
axes[0].set_ylabel('BSS')
axes[0].set_title('skill decay with forecast horizon', loc='left')
axes[0].legend(frameon=False, fontsize=7)
axes[1].axhline(0, color=INK2, lw=0.8)
axes[1].set_xlabel('seasons ahead')
axes[1].set_ylabel('mean_pred - actual')
axes[1].set_title('calibration drift (positive = over-predicting)', loc='left')
save(fig, 'e2_temporal_decay.png')

summary = dict(
    stamp=STAMP, rows=int(len(df)),
    season_jumps=jumps.to_dict(),
    largest_monthly_moves=moves,
    roster_turnover=turn.to_dict('records'),
    id_debut_corr=round(id_rank_corr, 4),
    regime_rank_stability=stab,
    top_extrapolation_risk=risk.head(12)[
        ['column', 'slope_in_sd', 'pred_2025_mean', 'outside_observed_range']
    ].to_dict('records'),
    temporal_decay=decay.to_dict('records'),
    runtime_sec=round(time.time() - t0, 1))
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=1, ensure_ascii=False, default=str)
with open(f'{OUT}/findings.md', 'w') as f:
    f.write('# E2 findings (auto)\n\n'
            + json.dumps(summary, indent=1, ensure_ascii=False, default=str))
print(f'\ndecision=E2_COMPLETE runtime={time.time()-t0:.0f}s', flush=True)
print('OUT_E2=' + OUT, flush=True)
