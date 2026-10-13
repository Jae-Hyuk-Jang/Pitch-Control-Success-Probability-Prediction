# E4 v2 - 물리·공간 지도 (Physical Geometry Atlas)
#
# 정직한 한계: trackman에 플레이트 통과 좌표가 없다. 따라서 존 히트맵은 원리적으로
#   불가. 그릴 수 있는 것은 릴리스 포인트(궤적 시작)와 무브먼트(곡률), 구속/회전.
#
# CPU ~8-15분. 출력: runs/physical_geometry_atlas_e4_<stamp>/

import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
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
TM0 = _latest(ROOT + '/runs/trackman_game_sequence_alignment_v1_staged_*'
             '/stage1_game_alignment')
STAMP = time.strftime('%Y%m%d_%H%M%S')
OUT = f'{ROOT}/runs/physical_geometry_atlas_e4_{STAMP}'
os.makedirs(OUT, exist_ok=True)
print('OUT_E4=' + OUT, flush=True)
t0 = time.time()


# ---------- 한글 폰트 ----------
def setup_korean_font():
    for pat in ('/usr/share/fonts/**/*Nanum*.tt[fc]',
                '/usr/share/fonts/**/*Noto*CJK*.tt[fc]',
                '/usr/share/fonts/**/*NotoSansKR*.tt[fc]'):
        for p in glob.glob(pat, recursive=True):
            try:
                fm.fontManager.addfont(p)
            except Exception:
                pass
    have = {f.name for f in fm.fontManager.ttflist}
    for c in ('NanumGothic', 'NanumBarunGothic', 'Noto Sans CJK KR',
              'Noto Sans KR', 'Malgun Gothic', 'AppleGothic'):
        if c in have:
            plt.rcParams['font.family'] = c
            plt.rcParams['axes.unicode_minus'] = False
            return c
    return None


KFONT = setup_korean_font()
KO = KFONT is not None
print(f'korean font = {KFONT}  (없으면 영문 라벨로 자동 전환)', flush=True)


def T(ko, en):
    return ko if KO else en


PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
GROUP_COLOR = {'fastball': PAL[0], 'breaking': PAL[1], 'offspeed': PAL[2],
               'other': "#8a8880"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f6f9", "#9dc0e8", "#2a78d6", "#123a6b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efeb", "#eb6834"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 8, "axes.titlesize": 9})


def save(fig, name, tight=True):
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(f"{OUT}/{name}", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[chart] {name}", flush=True)


def between_var(d, key, y):
    g = d.dropna(subset=[y]).groupby(key, observed=True)[y]
    n = g.count().astype(float)
    mu = g.mean()
    N = n.sum()
    if N == 0 or len(n) < 2:
        return 0.0
    gm = float((n * mu).sum() / N)
    return float((n * (mu - gm) ** 2).sum() / N)


REPORT = {'korean_font': KFONT}
REF = 0.24963554

# ============================================================
# Part A — 물리 지도
# ============================================================
PHYS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
        'extension', 'rel_height', 'rel_side', 'zone_speed']
tm = pd.read_csv(f'{ROOT}/data/trackman_history.csv', low_memory=False,
                 usecols=['season', 'pitcher_trackman_id', 'pitcher_hand',
                          'pitch_type_group', 'trackman_game_id', 'pitch_no',
                          'pitch_of_pa', 'balls_before', 'strikes_before'] + PHYS)
print(f'trackman {tm.shape} in {time.time()-t0:.0f}s', flush=True)

tm = tm[(tm.balls_before <= 3) & (tm.strikes_before <= 2)]
q = tm[PHYS].quantile([0.001, 0.999])
for c in PHYS:
    tm = tm[(tm[c].isna()) | ((tm[c] >= q.loc[0.001, c]) & (tm[c] <= q.loc[0.999, c]))]
tm = tm.dropna(subset=['rel_side', 'rel_height', 'horz_break',
                       'induced_vert_break', 'rel_speed'])
print(f'after clean {tm.shape}', flush=True)

tm['rel_side_mirror'] = np.where(tm.pitcher_hand == 'Left', -tm.rel_side, tm.rel_side)
tm['horz_break_mirror'] = np.where(tm.pitcher_hand == 'Left', -tm.horz_break,
                                   tm.horz_break)

# --- A1 릴리스 포인트 (포수 시점)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, hand in zip(axes, ['Right', 'Left']):
    s = tm[tm.pitcher_hand == hand]
    hb = ax.hexbin(s.rel_side, s.rel_height, gridsize=70, cmap=SEQ,
                   mincnt=8, linewidths=0)
    ax.plot([0, 0], [0, 2.6], color=INK2, lw=1, ls=':')
    ax.set_xlabel(T('rel_side  (음수=3루측, 양수=1루측)',
                    'rel_side  (neg=3B side, pos=1B side)'))
    ax.set_ylabel('rel_height (m)')
    ax.set_title(T(f'{hand} 투수 릴리스 포인트  n={len(s):,}',
                   f'{hand}-handed release point  n={len(s):,}'), loc='left')
    ax.set_aspect('equal')
fig.colorbar(hb, ax=axes, shrink=0.8, label=T('투구 수', 'pitches'))
fig.suptitle(T('A1. 공이 손을 떠나는 지점 — 포수 시점',
               'A1. Where the ball leaves the hand — catcher view'),
             x=0.01, ha='left', fontsize=11)
save(fig, 'e4_A1_release_point.png', tight=False)

# --- A2 무브먼트 지도
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
for ax, g in zip(axes, ['fastball', 'breaking', 'offspeed']):
    s = tm[tm.pitch_type_group == g]
    ax.hexbin(s.horz_break, s.induced_vert_break, gridsize=60, cmap=SEQ,
              mincnt=8, linewidths=0)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_xlabel('horz_break (cm)')
    ax.set_ylabel('induced_vert_break (cm)')
    ax.set_title(T(f'{g}  n={len(s):,}  평균 {s.rel_speed.mean():.1f}km/h',
                   f'{g}  n={len(s):,}  mean {s.rel_speed.mean():.1f}km/h'), loc='left')
fig.suptitle(T('A2. 공이 어떻게 휘는가 — 구종군별 무브먼트 (포수 시점)',
               'A2. How the ball moves — by pitch group (catcher view)'),
             x=0.01, ha='left', fontsize=11)
save(fig, 'e4_A2_movement_map.png')

# --- A3 구속 x 무브먼트
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for g in ['fastball', 'breaking', 'offspeed']:
    n_g = int((tm.pitch_type_group == g).sum())
    s = tm[tm.pitch_type_group == g].sample(min(60000, n_g), random_state=0)
    axes[0].scatter(s.rel_speed, s.induced_vert_break, s=2, alpha=0.06,
                    color=GROUP_COLOR[g], edgecolors='none')
    axes[1].scatter(s.rel_speed, s.horz_break, s=2, alpha=0.06,
                    color=GROUP_COLOR[g], edgecolors='none')
    axes[2].scatter(s.spin_rate, s.induced_vert_break, s=2, alpha=0.06,
                    color=GROUP_COLOR[g], edgecolors='none')
for ax, (xl, yl) in zip(axes, [('rel_speed (km/h)', 'induced_vert_break'),
                               ('rel_speed (km/h)', 'horz_break'),
                               ('spin_rate (rpm)', 'induced_vert_break')]):
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
h = [plt.Line2D([], [], marker='o', ls='', color=GROUP_COLOR[g], label=g)
     for g in ['fastball', 'breaking', 'offspeed']]
axes[0].legend(handles=h, frameon=False, fontsize=7)
fig.suptitle(T('A3. 구속·회전과 무브먼트의 관계',
               'A3. Velocity / spin vs movement'), x=0.01, ha='left', fontsize=11)
save(fig, 'e4_A3_speed_movement.png')

# --- A4 투수별 아암슬롯
pp = tm.groupby(['pitcher_trackman_id', 'pitcher_hand']).agg(
    n_tm=('rel_speed', 'size'), rel_side=('rel_side', 'mean'),
    rel_height=('rel_height', 'mean'), rel_side_sd=('rel_side', 'std'),
    rel_height_sd=('rel_height', 'std'), speed=('rel_speed', 'mean'),
    ivb=('induced_vert_break', 'mean'), hb=('horz_break', 'mean'),
    hb_m=('horz_break_mirror', 'mean'), ext=('extension', 'mean'),
    spin=('spin_rate', 'mean')).reset_index()
pp = pp[pp.n_tm >= 300]
pp['slot_angle'] = np.degrees(np.arctan2(pp.rel_height, pp.rel_side.abs()))
fig, ax = plt.subplots(figsize=(7.5, 6))
sc = ax.scatter(pp.rel_side, pp.rel_height, s=np.sqrt(pp.n_tm) / 2.2,
                c=pp.speed, cmap=SEQ, alpha=0.85, edgecolors='none')
ax.axvline(0, color=INK2, lw=1, ls=':')
ax.set_xlabel(T('평균 rel_side', 'mean rel_side'))
ax.set_ylabel(T('평균 rel_height', 'mean rel_height'))
ax.set_aspect('equal')
fig.colorbar(sc, label=T('평균 구속 (km/h)', 'mean velocity (km/h)'))
ax.set_title(T(f'A4. 투수 {len(pp)}명의 아암슬롯 지도 (크기=투구수, 색=구속)',
               f'A4. Arm slots of {len(pp)} pitchers (size=pitches, color=velocity)'),
             loc='left')
save(fig, 'e4_A4_pitcher_armslots.png', tight=False)
pp.to_csv(f'{OUT}/e4_pitcher_physics.csv', index=False)
REPORT['A_pitchers_with_physics'] = int(len(pp))

# ============================================================
# Part B — 물리 x 실패 유형 (TM0 정렬)
# ============================================================
pmap = pd.read_csv(f'{TM0}/pitcher_id_map.csv')
print(f'\npitcher_id_map cols={list(pmap.columns)} rows={len(pmap)}', flush=True)
if 'purity' in pmap.columns:
    before = len(pmap)
    pmap = pmap[pmap.purity >= 0.99]
    print(f'purity>=0.99 filter: {before} -> {len(pmap)}', flush=True)
pmap = pmap[['pitcher_id', 'pitcher_tm']].rename(
    columns={'pitcher_tm': 'pitcher_trackman_id'}).dropna()
pmap['pitcher_trackman_id'] = pmap.pitcher_trackman_id.astype(pp.pitcher_trackman_id.dtype)

tr = pd.read_csv(f'{ROOT}/data/train.csv',
                 usecols=['pitcher_id', 'pitcher_team_id', 'batter_team_id',
                          'top_bottom', 'game_dayofweek', 'season',
                          'score_diff_home', 'score_diff_pitcher_team',
                          'control_success'], low_memory=False)
lab = pd.read_parquet(LR0, columns=['lab_success', 'lab_reverse', 'lab_middle',
                                    'lab_ball', 'lab_strike'])
df = pd.concat([tr.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
df = df[df.lab_success.notna()].copy()
df['lab_inplay'] = 1.0 - df.lab_ball - df.lab_strike
df['lab_other_fail'] = ((df.lab_success == 0) & (df.lab_reverse == 0)
                        & (df.lab_middle == 0)).astype(float)
LBL = ['lab_success', 'lab_reverse', 'lab_middle', 'lab_ball', 'lab_other_fail']

prof = df.groupby('pitcher_id').agg(
    n_main=('lab_success', 'size'),
    **{c.replace('lab_', ''): (c, 'mean') for c in LBL}).reset_index()
prof = prof[prof.n_main >= 500]
J = prof.merge(pmap, on='pitcher_id').merge(pp, on='pitcher_trackman_id')
assert 'n_main' in J.columns and 'n_tm' in J.columns, list(J.columns)
print(f'\nB: 물리+라벨 모두 있는 투수 = {len(J)}명', flush=True)
J.to_csv(f'{OUT}/e4_joined_pitcher_physics_labels.csv', index=False)
REPORT['B_joined_pitchers'] = int(len(J))

PHYS_VARS = [('speed', T('평균 구속', 'velocity')),
             ('spin', T('평균 회전수', 'spin rate')),
             ('ivb', T('수직 무브먼트', 'vert break')),
             ('hb_m', T('수평 무브먼트(손 기준)', 'horz break (hand-rel)')),
             ('ext', T('익스텐션', 'extension')),
             ('rel_height', T('릴리스 높이', 'release height')),
             ('slot_angle', T('아암슬롯 각도', 'arm slot angle')),
             ('rel_side_sd', T('릴리스 좌우 산포', 'release side SD')),
             ('rel_height_sd', T('릴리스 높이 산포', 'release height SD'))]
TARGETS = [('success', T('성공', 'success')),
           ('reverse', T('reverse(방향실수)', 'reverse')),
           ('middle', T('middle(실투)', 'middle')),
           ('other_fail', T('other_fail(존밖이탈)', 'other_fail'))]

corr = []
for pv, pl in PHYS_VARS:
    for tv, tl in TARGETS:
        m = J[pv].notna() & J[tv].notna()
        r = float(np.corrcoef(J.loc[m, pv], J.loc[m, tv])[0, 1]) if m.sum() > 30 else np.nan
        corr.append(dict(physics=pv, physics_label=pl, target=tv,
                         r=round(r, 4), n=int(m.sum())))
corr = pd.DataFrame(corr)
corr.to_csv(f'{OUT}/e4_B_physics_label_corr.csv', index=False)
piv = corr.pivot(index='physics', columns='target', values='r').reindex(
    [p[0] for p in PHYS_VARS])[[t[0] for t in TARGETS]]
print('\n=== B: 투수 물리 특성 vs 실패 유형 상관 ===', flush=True)
print(piv.round(3).to_string(), flush=True)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(piv.values, cmap=DIV, vmin=-0.5, vmax=0.5)
ax.set_xticks(range(len(TARGETS)))
ax.set_xticklabels([t[1] for t in TARGETS], rotation=20, ha='right')
ax.set_yticks(range(len(PHYS_VARS)))
ax.set_yticklabels([p[1] for p in PHYS_VARS])
ax.grid(False)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=7,
                    color=INK if abs(v) < 0.3 else SURFACE)
fig.colorbar(im, shrink=0.8)
ax.set_title(T(f'B1. 투수 물리 특성 x 실패 유형 (투수 {len(J)}명)',
               f'B1. Pitcher physics x failure mode ({len(J)} pitchers)'), loc='left')
save(fig, 'e4_B1_physics_label_heatmap.png', tight=False)

# B2 상위 4개 관계 산점
top4 = corr.dropna(subset=['r']).reindex(
    corr.dropna(subset=['r']).r.abs().sort_values(ascending=False).index).head(4)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (_, row) in zip(axes, top4.iterrows()):
    ax.scatter(J[row.physics], J[row.target], s=np.sqrt(J.n_main) / 3,
               alpha=0.55, color=PAL[0], edgecolors='none')
    m = J[row.physics].notna()
    b, a = np.polyfit(J.loc[m, row.physics], J.loc[m, row.target], 1)
    xs = np.linspace(J[row.physics].min(), J[row.physics].max(), 20)
    ax.plot(xs, b * xs + a, color=PAL[1], lw=2)
    ax.set_xlabel(row.physics_label)
    ax.set_ylabel(row.target)
    ax.set_title(f'r={row.r:+.3f}', loc='left')
fig.suptitle(T('B2. 가장 강한 물리-실패 관계 (점 크기=투구수)',
               'B2. Strongest physics-failure relations (size=pitches)'),
             x=0.01, ha='left', fontsize=11)
save(fig, 'e4_B2_top_relations.png')

# B3 사분면
J['q_speed'] = (J.speed > J.speed.median()).map(
    {True: T('고속', 'hi-velo'), False: T('저속', 'lo-velo')})
J['q_ivb'] = (J.ivb > J.ivb.median()).map(
    {True: T('고IVB', 'hi-IVB'), False: T('저IVB', 'lo-IVB')})
J['quad'] = J.q_speed + '/' + J.q_ivb
quad = J.groupby('quad')[[t[0] for t in TARGETS]].mean().round(4)
quad['n_pitchers'] = J.groupby('quad').size()
quad.to_csv(f'{OUT}/e4_B3_quadrant_profile.csv')
print('\n=== B3: 구속x수직무브먼트 사분면별 실패 프로파일 ===', flush=True)
print(quad.to_string(), flush=True)
REPORT['B3_quadrant'] = quad.to_dict('index')

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(quad))
for i, (t, tl) in enumerate(TARGETS[1:]):
    ax.bar(x + (i - 1) * 0.26, quad[t], width=0.25, color=PAL[i], label=tl)
ax.set_xticks(x)
ax.set_xticklabels(quad.index)
ax.set_title(T('B3. 투수 유형 사분면별 실패 유형 구성',
               'B3. Failure composition by pitcher archetype'), loc='left')
ax.legend(frameon=False, fontsize=7)
save(fig, 'e4_B3_quadrants.png')

# ============================================================
# Part C — 미탐색 축
# ============================================================
eq = (df.score_diff_home == df.score_diff_pitcher_team)
chk = eq.groupby(df.top_bottom).mean()
print('\n=== C0: top_bottom별 (score_diff_home == score_diff_pitcher_team) 비율 ===',
      flush=True)
print(chk.to_string(), flush=True)
home_tag = chk.idxmax()
df['pitcher_is_home'] = (df.top_bottom == home_tag).astype(int)
REPORT['C0_home_tag'] = dict(top_bottom_value=str(home_tag),
                             agreement=round(float(chk.max()), 6),
                             other=round(float(chk.min()), 6))
print(f"-> 투수가 홈인 경우: top_bottom == '{home_tag}'  (일치율 {chk.max():.4f})",
      flush=True)

c1 = df.groupby('pitcher_is_home')[LBL + ['lab_inplay']].mean().round(5)
c1['n'] = df.groupby('pitcher_is_home').size()
c1.to_csv(f'{OUT}/e4_C1_home_away.csv')
print('\n=== C1: 홈/원정 (1=투수가 홈) ===', flush=True)
print(c1.to_string(), flush=True)

df['home_team_id'] = np.where(df.pitcher_is_home == 1, df.pitcher_team_id,
                              df.batter_team_id)
vc = df.home_team_id.value_counts()
main_parks = vc[vc >= 50000].index
dpark = df[df.home_team_id.isin(main_parks)]
park_raw = dpark.groupby('home_team_id')[LBL].mean()
park_n = dpark.groupby('home_team_id').size().rename('n')
c2 = (park_raw - park_raw.mean()).round(5).join(park_n)
c2.to_csv(f'{OUT}/e4_C2_park_deviation.csv')
print('\n=== C2: 구장(홈팀 기준) 리그평균 대비 편차 ===', flush=True)
print(c2.to_string(), flush=True)

c3 = df.groupby('game_dayofweek')[LBL].mean().round(5)
c3['n'] = df.groupby('game_dayofweek').size()
c3.to_csv(f'{OUT}/e4_C3_dayofweek.csv')
print('\n=== C3: 요일 (0=월) ===', flush=True)
print(c3.to_string(), flush=True)

vd = []
for name, key, d in [('pitcher_is_home', 'pitcher_is_home', df),
                     ('park_home_team', 'home_team_id', dpark),
                     ('game_dayofweek', 'game_dayofweek', df),
                     ('park_x_home', ['home_team_id', 'pitcher_is_home'], dpark)]:
    for c in LBL:
        bv = between_var(d, key, c)
        vd.append(dict(axis=name, label=c.replace('lab_', ''),
                       bv_x1e4=round(bv * 1e4, 4),
                       score_potential=round(1e5 * bv / REF, 1)))
vd = pd.DataFrame(vd).sort_values(['label', 'bv_x1e4'], ascending=[True, False])
vd.to_csv(f'{OUT}/e4_C4_new_axes_variance.csv', index=False)
print('\n=== C4: 신규 축 분산 (비교: pitcher_id 22.8, count->ball 31.6) ===', flush=True)
print(vd.to_string(index=False), flush=True)
REPORT['C4_new_axes'] = vd.to_dict('records')

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
axes[0].bar([T('원정', 'away'), T('홈', 'home')], c1.loc[[0, 1], 'lab_success'],
            color=PAL[0])
axes[0].set_ylim(c1.lab_success.min() - 0.004, c1.lab_success.max() + 0.004)
axes[0].set_title(T('C1. 홈/원정 성공률', 'C1. home vs away success'), loc='left')
d2 = c2.lab_success.sort_values()
axes[1].barh([str(i) for i in d2.index], d2.values, color=PAL[1])
axes[1].axvline(0, color=INK2, lw=0.8)
axes[1].set_title(T('C2. 구장별 성공률 편차', 'C2. park deviation'), loc='left')
axes[2].bar([str(i) for i in c3.index], c3.lab_success, color=PAL[2])
axes[2].set_ylim(c3.lab_success.min() - 0.004, c3.lab_success.max() + 0.004)
axes[2].set_title(T('C3. 요일별 성공률 (0=월)', 'C3. by day of week (0=Mon)'),
                  loc='left')
save(fig, 'e4_C_new_axes.png')

REPORT['C1_home_away'] = c1.to_dict('index')
REPORT['C2_park_dev'] = c2[LBL].to_dict('index')
REPORT['C3_dayofweek'] = c3.to_dict('index')
REPORT['B_corr_table'] = piv.round(4).to_dict()
REPORT['stamp'] = STAMP
REPORT['runtime_sec'] = round(time.time() - t0, 1)
REPORT['limitation'] = ('trackman에 플레이트 통과 좌표가 없어 존 히트맵은 불가. '
                        '릴리스 포인트(궤적 시작)와 무브먼트(곡률)는 실측.')
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(REPORT, f, indent=1, ensure_ascii=False, default=str)
with open(f'{OUT}/findings.md', 'w') as f:
    f.write('# E4 findings (auto)\n\n'
            + json.dumps(REPORT, indent=1, ensure_ascii=False, default=str))
print(f'\ndecision=E4_COMPLETE runtime={time.time()-t0:.0f}s', flush=True)
print('OUT_E4=' + OUT, flush=True)
