# E1 - 투수 단위 심층 프로파일
#
# 질문: "투수 정체성 축의 분산 중 진짜 실력은 몇 %인가"
#  1) 투수별 커리어 프로파일 테이블 (성공/reverse/middle/ball + 구종군 믹스)
#  2) Funnel: 성공률 vs 표본수 + 이항 노이즈 밴드 -> 밴드 밖 비율 = 실제 이질성
#  3) Split-half 신뢰도 (짝/홀 투구): 라벨별 실력 분산 비율
#  4) 시즌간 안정성: 연속 시즌 쌍 r (success/reverse/middle/ball) — reverse=실력,
#     middle=랜덤 가설의 정량 검증
#  5) Empirical-Bayes 수축 상수 M (pseudo-count): 스무딩 설계 직결값
#  6) 콜드스타트 수렴: |asof rate - 최종 커리어 rate| vs 누적 n 곡선
#
# CPU ~4-8분. 출력: runs/pitcher_deep_profile_e1_<stamp>/

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
OUT = f'{ROOT}/runs/pitcher_deep_profile_e1_{STAMP}'
os.makedirs(OUT, exist_ok=True)
print('OUT_E1=' + OUT, flush=True)
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


tr = pd.read_csv(f'{ROOT}/data/train.csv', usecols=[
    'pitcher_id', 'season', 'pitcher_hand', 'control_success',
    'asof_pitcher_n', 'asof_pitcher_success_rate'], low_memory=False)
lab = pd.read_parquet(LR0, columns=[
    'lab_success', 'lab_reverse', 'lab_middle', 'lab_ball',
    'lab_fastball', 'lab_breaking', 'lab_offspeed'])
assert len(tr) == len(lab)
df = pd.concat([tr.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
df = df[df['lab_success'].notna()].copy()
print(f'rows={len(df)} in {time.time()-t0:.0f}s', flush=True)

LBL = ['lab_success', 'lab_reverse', 'lab_middle', 'lab_ball']

# ---- 1) career profiles
g = df.groupby('pitcher_id')
prof = g.size().rename('n').to_frame()
prof['hand'] = g['pitcher_hand'].first()
prof['n_seasons'] = g['season'].nunique()
for c in LBL + ['lab_fastball', 'lab_breaking', 'lab_offspeed']:
    prof[c.replace('lab_', '')] = g[c].mean()
prof = prof.reset_index()
prof.to_csv(f'{OUT}/pitcher_profiles.csv', index=False)
print(f'pitchers={len(prof)}  n>=1000: {(prof.n >= 1000).sum()}', flush=True)

p0 = float(df['lab_success'].mean())

# ---- 2) funnel: rate vs n with binomial band
big = prof[prof.n >= 100]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(big.n, big.success, s=8, alpha=0.45, color=PAL[0], edgecolors='none')
ns = np.geomspace(100, big.n.max(), 200)
for z, ls in ((1.96, '-'), (2.81, ':')):
    ax.plot(ns, p0 + z * np.sqrt(p0 * (1 - p0) / ns), color=PAL[1], lw=1.5, ls=ls)
    ax.plot(ns, p0 - z * np.sqrt(p0 * (1 - p0) / ns), color=PAL[1], lw=1.5, ls=ls)
ax.set_xscale('log')
ax.set_xlabel('career pitches (log)')
ax.set_ylabel('career success rate')
ax.set_title('funnel: pitcher success vs sample size (orange: 95% / 99.5% binomial band)',
             loc='left')
save(fig, 'e1_funnel_success.png')
z = (big.success - p0) / np.sqrt(p0 * (1 - p0) / big.n)
outside95 = float((np.abs(z) > 1.96).mean())
print(f'outside 95% band: {outside95:.3f} (노이즈만이면 0.05)', flush=True)

# ---- 3) split-half reliability (even/odd pitches per pitcher)
df['_ix'] = df.groupby('pitcher_id').cumcount()
half = df[df.groupby('pitcher_id')['pitcher_id'].transform('size') >= 1000]
rel = {}
for c in LBL:
    a = half[half['_ix'] % 2 == 0].groupby('pitcher_id')[c].mean()
    b = half[half['_ix'] % 2 == 1].groupby('pitcher_id')[c].mean()
    j = a.index.intersection(b.index)
    r = float(np.corrcoef(a[j], b[j])[0, 1])
    rel[c.replace('lab_', '')] = dict(
        split_half_r=round(r, 4),
        spearman_brown=round(2 * r / (1 + r), 4), pitchers=int(len(j)))
print('split-half reliability:', json.dumps(rel), flush=True)

# ---- 4) year-to-year stability
ps = df.groupby(['pitcher_id', 'season']).agg(
    n=('lab_success', 'size'),
    **{c.replace('lab_', ''): (c, 'mean') for c in LBL}).reset_index()
ps_q = ps[ps.n >= 300].copy()
ps_q['next_season'] = ps_q.season + 1
pairs = ps_q.merge(ps_q, left_on=['pitcher_id', 'next_season'],
                   right_on=['pitcher_id', 'season'], suffixes=('_y1', '_y2'))
y2y = {}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, c in zip(axes, [c.replace('lab_', '') for c in LBL]):
    x_, y_ = pairs[f'{c}_y1'], pairs[f'{c}_y2']
    r = float(np.corrcoef(x_, y_)[0, 1])
    y2y[c] = dict(r=round(r, 4), pairs=int(len(pairs)))
    ax.scatter(x_, y_, s=7, alpha=0.4, color=PAL[0], edgecolors='none')
    lim = [min(x_.min(), y_.min()), max(x_.max(), y_.max())]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls=':')
    ax.set_title(f'{c}  y2y r={r:.3f}', loc='left')
    ax.set_xlabel('season t')
    ax.set_ylabel('season t+1')
save(fig, 'e1_year_to_year.png')
print('year-to-year:', json.dumps(y2y), flush=True)

# ---- 5) empirical-Bayes shrinkage constant per label
eb = {}
for c in LBL:
    m = prof[prof.n >= 200]
    rates, ns_ = m[c.replace('lab_', '')].to_numpy(), m.n.to_numpy()
    mu = float(np.average(rates, weights=ns_))
    var_obs = float(np.average((rates - mu) ** 2, weights=ns_))
    noise = float(np.average(mu * (1 - mu) / ns_, weights=ns_))
    var_true = max(var_obs - noise, 1e-9)
    M = mu * (1 - mu) / var_true  # pseudo-count prior strength
    eb[c.replace('lab_', '')] = dict(
        mu=round(mu, 5), sd_true=round(float(np.sqrt(var_true)), 5),
        prior_strength_M=round(M, 1),
        signal_share=round(var_true / var_obs, 4))
print('empirical-Bayes:', json.dumps(eb), flush=True)

# ---- 6) cold-start convergence: |asof rate - final| by cumulative n
final = df.groupby('pitcher_id')['lab_success'].mean().rename('final_rate')
dd = df[['pitcher_id', 'asof_pitcher_n', 'asof_pitcher_success_rate']].join(
    final, on='pitcher_id')
dd = dd[dd['asof_pitcher_success_rate'].notna()
        & (dd.groupby('pitcher_id')['pitcher_id'].transform('size') >= 3000)]
dd['err'] = (dd['asof_pitcher_success_rate'] - dd['final_rate']).abs()
buckets = [0, 50, 100, 200, 500, 1000, 2000, 5000, 100000]
dd['nb'] = pd.cut(dd['asof_pitcher_n'], buckets)
conv = dd.groupby('nb', observed=True)['err'].agg(['mean', 'median', 'size'])
conv.to_csv(f'{OUT}/e1_coldstart_convergence.csv')
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(len(conv)), conv['mean'], color=PAL[0], lw=2, marker='o', ms=4,
        label='mean |asof − final|')
ax.plot(range(len(conv)), conv['median'], color=PAL[1], lw=2, marker='o', ms=4,
        label='median')
ax.set_xticks(range(len(conv)))
ax.set_xticklabels([str(i) for i in conv.index], rotation=30, fontsize=7)
ax.set_title('cold-start: asof rate error vs cumulative n (career>=3000 pitchers)',
             loc='left')
ax.legend(frameon=False)
save(fig, 'e1_coldstart_convergence.png')
print(conv.to_string(), flush=True)

# ---- distribution + hand/mix quartiles
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].hist(prof[prof.n >= 1000].success, bins=40, color=PAL[0])
axes[0].set_title('career success rate (n>=1000)', loc='left')
axes[1].hist(prof[prof.n >= 1000].reverse, bins=40, color=PAL[1])
axes[1].set_title('career reverse rate', loc='left')
axes[2].hist(prof[prof.n >= 1000].middle, bins=40, color=PAL[3])
axes[2].set_title('career middle rate', loc='left')
save(fig, 'e1_rate_distributions.png')

summary = dict(stamp=STAMP, pitchers=int(len(prof)), p0=round(p0, 5),
               funnel_outside95=outside95, split_half=rel, year_to_year=y2y,
               empirical_bayes=eb,
               runtime_sec=round(time.time() - t0, 1))
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=1, ensure_ascii=False)
with open(f'{OUT}/findings.md', 'w') as f:
    f.write('# E1 findings (auto)\n\n' + json.dumps(summary, indent=1, ensure_ascii=False))
print(f'\ndecision=E1_COMPLETE runtime={time.time()-t0:.0f}s', flush=True)
print('OUT_E1=' + OUT, flush=True)
