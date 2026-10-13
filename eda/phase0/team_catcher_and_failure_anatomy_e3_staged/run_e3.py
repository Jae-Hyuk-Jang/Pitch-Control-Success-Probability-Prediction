# E3 - 팀/포수 콜링 가설 + other_fail 정체 + 타자 축
#
# 가설 H1 (핵심): reverse = "포수 요구 방향의 반대". 포수는 팀 소속이고 투수는 보통
#   한 팀에 머무르므로, E1이 잰 reverse의 높은 개인 신뢰도(r=0.932) 중 일부가
#   실은 팀(=포수진 콜링 스타일) 효과일 수 있다.
#   자연 실험: 팀을 옮긴 투수. reverse가 선수를 따라가는가, 팀을 따라가는가.
#
#   S1 분산분해: reverse를 팀 / 팀내 투수로 분해. success와 대조.
#   S2 이적 투수 within-pitcher 대조: 같은 투수의 이적 전/후 Δreverse를
#      "이적한 두 팀의 리그 대비 편차 차이"로 예측할 수 있는가.
#      기울기 beta ~= 1이면 팀이 원인, ~= 0이면 순수 개인 특성.
#      대조군: 팀을 안 옮긴 투수의 같은 기간 Δ (노이즈 하한)
#   S3 팀 효과의 시간 안정성: 팀별 reverse 편차가 시즌간 유지되는가
#      (유지되면 포수진/코칭 스타일, 흔들리면 우연)
#
# 가설 H2: other_fail(실패인데 reverse도 middle도 아님, 13.2%) = "존에서 과도한 벗어남"
#   검증: other_fail의 ball 비율이 다른 실패 유형보다 압도적으로 높아야 한다.
#
# 가설 H3: batter_id의 분산은 진짜 타자 효과인가, 상대 투수 구성의 프록시인가
#   검증: 타자 축 실력/노이즈 분해(E1과 동형) + 투수 품질 통제 후 잔차 분산
#
# CPU ~5-10분. 출력: runs/team_catcher_failure_anatomy_e3_<stamp>/

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
OUT = f'{ROOT}/runs/team_catcher_failure_anatomy_e3_{STAMP}'
os.makedirs(OUT, exist_ok=True)
print('OUT_E3=' + OUT, flush=True)
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
    'pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id', 'season',
    'pitcher_hand', 'batter_hand', 'control_success'], low_memory=False)
lab = pd.read_parquet(LR0, columns=['lab_success', 'lab_reverse', 'lab_middle',
                                    'lab_ball', 'lab_strike'])
df = pd.concat([tr.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
df = df[df['lab_success'].notna()].copy()
df['lab_inplay'] = 1.0 - df['lab_ball'] - df['lab_strike']
df['lab_other_fail'] = ((df.lab_success == 0) & (df.lab_reverse == 0)
                        & (df.lab_middle == 0)).astype(float)
print(f'rows={len(df)} in {time.time()-t0:.0f}s', flush=True)

# 주 구단 10곳만 (E0: 3개 팀은 합계 0.37%)
tc = df.pitcher_team_id.value_counts()
MAIN_TEAMS = sorted(tc[tc >= 50000].index.tolist())
print(f'main teams ({len(MAIN_TEAMS)}): {MAIN_TEAMS}', flush=True)
dm = df[df.pitcher_team_id.isin(MAIN_TEAMS)].copy()

LBL = ['lab_success', 'lab_reverse', 'lab_middle', 'lab_ball']
res = {}

# ============ H2: other_fail 정체 (가벼우니 먼저) ============
fail = df[df.lab_success == 0].copy()
fail['kind'] = np.select(
    [(fail.lab_reverse == 1) & (fail.lab_middle == 1),
     fail.lab_reverse == 1, fail.lab_middle == 1],
    ['rev_mid', 'reverse', 'middle'], default='other_fail')
h2 = fail.groupby('kind').agg(
    n=('lab_ball', 'size'), ball=('lab_ball', 'mean'),
    strike=('lab_strike', 'mean'), inplay=('lab_inplay', 'mean')).round(5)
h2['share'] = (h2.n / len(fail)).round(5)
h2.to_csv(f'{OUT}/e3_h2_other_fail_identity.csv')
print('\n=== H2: 실패 유형별 판정 분포 ===', flush=True)
print(h2.to_string(), flush=True)
succ_ball = float(df.loc[df.lab_success == 1, 'lab_ball'].mean())
print(f'(참고) success 행의 ball 비율 = {succ_ball:.5f}', flush=True)

fig, ax = plt.subplots(figsize=(8, 4))
kinds = list(h2.index)
x = np.arange(len(kinds) + 1)
for i, (col, c) in enumerate(zip(['ball', 'strike', 'inplay'], PAL[:3])):
    vals = list(h2[col]) + [float(df.loc[df.lab_success == 1, f'lab_{col}'].mean())]
    ax.bar(x + (i - 1) * 0.26, vals, width=0.25, color=c, label=col)
ax.set_xticks(x)
ax.set_xticklabels(kinds + ['(success)'])
ax.set_title('H2: 실패 유형별 판정 결과 — other_fail이 "존 밖 과도 이탈"이면 ball이 압도해야',
             loc='left')
ax.legend(frameon=False)
save(fig, 'e3_h2_other_fail.png')
res['H2_by_failure_kind'] = h2.to_dict('index')
res['H2_success_ball_rate'] = round(succ_ball, 5)

# ============ H1-S1: 팀 vs 팀내 투수 분산분해 ============
def between_var(d, key, y):
    g = d.groupby(key, observed=True)[y]
    n = g.count().astype(float)
    mu = g.mean()
    N = n.sum()
    gm = float((n * mu).sum() / N)
    return float((n * (mu - gm) ** 2).sum() / N)

s1 = []
for c in LBL:
    v_team = between_var(dm, 'pitcher_team_id', c)
    v_pit = between_var(dm, 'pitcher_id', c)
    # 팀 평균을 뺀 잔차 위에서의 투수 분산 = 팀 통제 후 개인 효과
    tm_mean = dm.groupby('pitcher_team_id')[c].transform('mean')
    dm['_r'] = dm[c] - tm_mean
    v_pit_within = between_var(dm, 'pitcher_id', '_r')
    s1.append(dict(label=c.replace('lab_', ''),
                   team_bv_x1e4=round(v_team * 1e4, 4),
                   pitcher_bv_x1e4=round(v_pit * 1e4, 4),
                   pitcher_within_team_bv_x1e4=round(v_pit_within * 1e4, 4),
                   team_share_of_pitcher=round(v_team / v_pit, 4),
                   pitcher_var_lost_to_team=round(1 - v_pit_within / v_pit, 4)))
s1 = pd.DataFrame(s1)
s1.to_csv(f'{OUT}/e3_s1_team_vs_pitcher.csv', index=False)
print('\n=== H1-S1: 팀 vs 투수 분산분해 ===', flush=True)
print(s1.to_string(index=False), flush=True)
res['S1_team_vs_pitcher'] = s1.to_dict('records')

# 팀별 라벨 편차 (리그 평균 대비)
tm = dm.groupby('pitcher_team_id')[LBL].mean()
tm_dev = (tm - dm[LBL].mean()).round(5)
tm_dev.to_csv(f'{OUT}/e3_team_deviation.csv')
print('\n팀별 리그 대비 편차:', flush=True)
print(tm_dev.to_string(), flush=True)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, c, col in zip(axes, LBL, PAL):
    d = tm_dev[c].sort_values()
    ax.barh([str(i) for i in d.index], d.values, color=col)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(f'{c.replace("lab_", "")} 팀 편차', loc='left')
save(fig, 'e3_s1_team_deviation.png')

# ============ H1-S3: 팀 효과의 시간 안정성 ============
ts = dm.groupby(['pitcher_team_id', 'season'])[LBL].mean()
lg = dm.groupby('season')[LBL].mean()
dev = (ts - lg.reindex(ts.index.get_level_values('season')).set_index(ts.index)).reset_index()
s3 = {}
for c in LBL:
    p = dev.pivot(index='pitcher_team_id', columns='season', values=c)
    rs = []
    for a, b in zip(sorted(p.columns)[:-1], sorted(p.columns)[1:]):
        m = p[a].notna() & p[b].notna()
        if m.sum() >= 8:
            rs.append(round(float(np.corrcoef(p.loc[m, a], p.loc[m, b])[0, 1]), 4))
    s3[c.replace('lab_', '')] = dict(consecutive_season_r=rs,
                                     mean_r=round(float(np.mean(rs)), 4) if rs else None)
print('\n=== H1-S3: 팀 편차의 시즌간 안정성 ===', flush=True)
print(json.dumps(s3, ensure_ascii=False), flush=True)
res['S3_team_effect_stability'] = s3

# ============ H1-S2: 이적 투수 자연 실험 ============
pt = (dm.groupby(['pitcher_id', 'season'])
      .agg(n=('lab_success', 'size'), team=('pitcher_team_id', 'first'),
           **{c.replace('lab_', ''): (c, 'mean') for c in LBL}).reset_index())
# 시즌 내 팀이 유일한 경우만 (시즌 중 이적은 first가 부정확)
uniq = dm.groupby(['pitcher_id', 'season'])['pitcher_team_id'].nunique()
pt = pt[pt.set_index(['pitcher_id', 'season']).index.map(uniq) == 1]
pt = pt[pt.n >= 200].copy()
pt['next'] = pt.season + 1
pr = pt.merge(pt, left_on=['pitcher_id', 'next'], right_on=['pitcher_id', 'season'],
              suffixes=('_1', '_2'))
pr['moved'] = (pr.team_1 != pr.team_2).astype(int)
print(f'\n연속시즌 쌍={len(pr)}  이적={int(pr.moved.sum())} ({pr.moved.mean():.3f})',
      flush=True)

# 팀 편차(해당 시즌, 그 투수 본인 제외 = leave-one-out)
lo = {}
for c in LBL:
    key = c.replace('lab_', '')
    grp = dm.groupby(['pitcher_team_id', 'season'])[c].agg(['sum', 'count'])
    pit = dm.groupby(['pitcher_id', 'pitcher_team_id', 'season'])[c].agg(['sum', 'count'])
    pit = pit.reset_index().merge(
        grp.reset_index(), on=['pitcher_team_id', 'season'], suffixes=('_p', '_t'))
    pit['loo'] = ((pit['sum_t'] - pit['sum_p'])
                  / (pit['count_t'] - pit['count_p']).replace(0, np.nan))
    lg_s = dm.groupby('season')[c].mean()
    pit['loo_dev'] = pit['loo'] - pit['season'].map(lg_s)
    lo[key] = pit[['pitcher_id', 'pitcher_team_id', 'season', 'loo_dev']]

s2 = {}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, c, col in zip(axes, LBL, PAL):
    key = c.replace('lab_', '')
    L = lo[key]
    a = pr.merge(L, left_on=['pitcher_id', 'team_1', 'season_1'],
                 right_on=['pitcher_id', 'pitcher_team_id', 'season']).rename(
        columns={'loo_dev': 'dev1'})
    a = a.merge(L, left_on=['pitcher_id', 'team_2', 'season_2'],
                right_on=['pitcher_id', 'pitcher_team_id', 'season']).rename(
        columns={'loo_dev': 'dev2'})
    a['d_player'] = a[f'{key}_2'] - a[f'{key}_1']
    a['d_team'] = a.dev2 - a.dev1
    mv = a[a.moved == 1]
    st = a[a.moved == 0]
    out = {}
    for nm, sub in (('moved', mv), ('stayed', st)):
        if len(sub) >= 25 and sub.d_team.std() > 1e-9:
            beta, icept = np.polyfit(sub.d_team, sub.d_player, 1)
            r = float(np.corrcoef(sub.d_team, sub.d_player)[0, 1])
            out[nm] = dict(n=int(len(sub)), beta=round(float(beta), 4),
                           r=round(r, 4), sd_d_player=round(float(sub.d_player.std()), 5))
        else:
            out[nm] = dict(n=int(len(sub)))
    s2[key] = out
    ax.scatter(st.d_team, st.d_player, s=9, alpha=0.28, color=INK2,
               edgecolors='none', label='stayed')
    ax.scatter(mv.d_team, mv.d_player, s=16, alpha=0.75, color=col,
               edgecolors='none', label='moved')
    if 'beta' in out.get('moved', {}):
        xs = np.linspace(mv.d_team.min(), mv.d_team.max(), 20)
        b, i0 = np.polyfit(mv.d_team, mv.d_player, 1)
        ax.plot(xs, b * xs + i0, color=col, lw=2)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel('Δ 팀 편차 (leave-one-out)')
    ax.set_ylabel('Δ 개인 비율')
    bm = out.get('moved', {}).get('beta')
    ax.set_title(f'{key}  이적 beta={bm}', loc='left')
    if c == LBL[0]:
        ax.legend(frameon=False, fontsize=7)
save(fig, 'e3_s2_team_change_natural_experiment.png')
print('\n=== H1-S2: 이적 자연실험 (beta≈1 팀 원인, ≈0 개인 특성) ===', flush=True)
print(json.dumps(s2, ensure_ascii=False, indent=1), flush=True)
res['S2_team_change_experiment'] = s2

# ============ H3: 타자 축 ============
bt = df.groupby('batter_id').agg(
    n=('lab_success', 'size'), **{c.replace('lab_', ''): (c, 'mean') for c in LBL})
bt = bt[bt.n >= 1000]
df['_bx'] = df.groupby('batter_id').cumcount()
h3 = {}
for c in LBL:
    key = c.replace('lab_', '')
    sub = df[df.batter_id.isin(bt.index)]
    a = sub[sub._bx % 2 == 0].groupby('batter_id')[c].mean()
    b = sub[sub._bx % 2 == 1].groupby('batter_id')[c].mean()
    j = a.index.intersection(b.index)
    r = float(np.corrcoef(a[j], b[j])[0, 1])
    rates, ns_ = bt[key].to_numpy(), bt.n.to_numpy()
    mu = float(np.average(rates, weights=ns_))
    var_obs = float(np.average((rates - mu) ** 2, weights=ns_))
    noise = float(np.average(mu * (1 - mu) / ns_, weights=ns_))
    vt = max(var_obs - noise, 1e-9)
    h3[key] = dict(split_half_r=round(r, 4), spearman_brown=round(2 * r / (1 + r), 4),
                   sd_true=round(float(np.sqrt(vt)), 5),
                   prior_M=round(mu * (1 - mu) / vt, 1),
                   signal_share=round(vt / var_obs, 4), batters=int(len(bt)))
print('\n=== H3: 타자 축 실력/노이즈 분해 (E1 투수와 대조) ===', flush=True)
print(json.dumps(h3, ensure_ascii=False, indent=1), flush=True)
res['H3_batter_axis'] = h3

# 투수 품질 통제 후 타자 잔차 분산 (batter_id가 상대 투수 구성의 프록시인가)
pq = df.groupby('pitcher_id')['lab_success'].transform('mean')
df['_resid'] = df['lab_success'] - pq
bv_raw = between_var(df, 'batter_id', 'lab_success')
bv_res = between_var(df, 'batter_id', '_resid')
res['H3_batter_confound'] = dict(
    batter_bv_raw_x1e4=round(bv_raw * 1e4, 4),
    batter_bv_after_pitcher_control_x1e4=round(bv_res * 1e4, 4),
    share_explained_by_opposing_pitchers=round(1 - bv_res / bv_raw, 4))
print('\nH3 교란 검사:', json.dumps(res['H3_batter_confound']), flush=True)

res['stamp'] = STAMP
res['runtime_sec'] = round(time.time() - t0, 1)
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(res, f, indent=1, ensure_ascii=False, default=str)
with open(f'{OUT}/findings.md', 'w') as f:
    f.write('# E3 findings (auto)\n\n'
            + json.dumps(res, indent=1, ensure_ascii=False, default=str))
print(f'\ndecision=E3_COMPLETE runtime={time.time()-t0:.0f}s', flush=True)
print('OUT_E3=' + OUT, flush=True)
