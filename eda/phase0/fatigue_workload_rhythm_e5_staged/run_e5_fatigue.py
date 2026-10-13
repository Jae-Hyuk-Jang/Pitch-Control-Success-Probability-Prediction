# E5 - 피로 / 등판 리듬 / 워크로드
#
# 남은 마지막 EDA 빈칸: trackman의 pitch_no(경기 내 순번), pitch_of_pa(타석 내 순번),
# game_date(등판 간격). 이 세 컬럼은 test에 없다 -> 직접 피처 불가.
# 그러나 라벨 없이도 **구속 감소**라는 물리 신호로 피로를 직접 측정할 수 있고,
# 투수 단위 "피로 서명"은 동결 아티팩트로 쓸 수 있다.
#
# Part A  trackman 순수 물리 피로 (라벨 불필요)
#   A1 등판 내 투구수 -> 구속 감소 곡선 (선발/불펜 분리)
#   A2 등판 내 투구수 -> 릴리스/무브먼트 변화
#   A3 등판 간격(휴식일) -> 구속·릴리스
#   A4 타순 일순(TTO) -> 구속·무브먼트
#   A5 타석 내 순번(pitch_of_pa) -> 구속·무브먼트
# Part B  메인 데이터 등판 경계 복원 가능성 검증 (가정하지 않고 데이터로 판정)
#   B0 row_id 간격 분포가 이봉형인가 -> 임계값 결정 가능 여부
#   B1 (가능하면) 등판 내 투구수 -> 실패 유형
#   B2 이닝 -> 실패 유형 (항상 가능한 crude proxy)
# Part C  투수 단위 피로 서명 x 실패 프로파일 (TM0 매핑)
#
# 방법론 주의: 구속 감소를 볼 때 **fastball만** 사용한다. 후반에 변화구 비율이 늘면
#   전체 평균 구속이 떨어져 가짜 피로로 보이기 때문.
#
# CPU ~8-15분. 출력: runs/fatigue_workload_rhythm_e5_<stamp>/

import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
OUT = f'{ROOT}/runs/fatigue_workload_rhythm_e5_{STAMP}'
os.makedirs(OUT, exist_ok=True)
print('OUT_E5=' + OUT, flush=True)
t0 = time.time()


def setup_korean_font():
    for pat in ('/usr/share/fonts/**/*Nanum*.tt[fc]',
                '/usr/share/fonts/**/*Noto*CJK*.tt[fc]'):
        for p in glob.glob(pat, recursive=True):
            try:
                fm.fontManager.addfont(p)
            except Exception:
                pass
    have = {f.name for f in fm.fontManager.ttflist}
    for c in ('NanumGothic', 'NanumBarunGothic', 'Noto Sans CJK KR'):
        if c in have:
            plt.rcParams['font.family'] = c
            plt.rcParams['axes.unicode_minus'] = False
            return c
    return None


KFONT = setup_korean_font()
KO = KFONT is not None
print(f'korean font = {KFONT}', flush=True)


def T(ko, en):
    return ko if KO else en


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
# Part A — trackman 물리 피로
# ============================================================
tm = pd.read_csv(f'{ROOT}/data/trackman_history.csv', low_memory=False,
                 usecols=['season', 'game_date', 'trackman_game_id', 'pitch_no',
                          'inning', 'pitch_of_pa', 'pitcher_trackman_id',
                          'batter_trackman_id', 'pitcher_hand', 'pitch_type_group',
                          'balls_before', 'strikes_before',
                          'rel_speed', 'spin_rate', 'induced_vert_break',
                          'horz_break', 'rel_height', 'rel_side', 'extension'])
print(f'trackman {tm.shape} in {time.time()-t0:.0f}s', flush=True)
tm = tm[(tm.balls_before <= 3) & (tm.strikes_before <= 2) & (tm.inning >= 1)]

# 날짜 파싱: E0에서 두 포맷 혼재 확인됨 (2023-09-09 / 06/26/2021)
d1 = pd.to_datetime(tm.game_date, format='%Y-%m-%d', errors='coerce')
d2 = pd.to_datetime(tm.game_date, format='%m/%d/%Y', errors='coerce')
tm['gdate'] = d1.fillna(d2)
print(f'game_date 파싱 실패율: {tm.gdate.isna().mean():.6f}', flush=True)
REPORT['A_date_parse_fail_rate'] = round(float(tm.gdate.isna().mean()), 6)

# 등판 단위: (경기, 투수). 경기 내 순번으로 정렬해 등판 내 투구 인덱스 부여
tm = tm.sort_values(['trackman_game_id', 'pitcher_trackman_id', 'pitch_no'])
grp = tm.groupby(['trackman_game_id', 'pitcher_trackman_id'], sort=False)
tm['app_pitch'] = grp.cumcount() + 1
app = grp.agg(app_pitches=('pitch_no', 'size'), first_inning=('inning', 'min'),
              gdate=('gdate', 'first')).reset_index()
app['role'] = np.where((app.first_inning <= 1) & (app.app_pitches >= 40),
                       T('선발', 'starter'), T('불펜', 'reliever'))
tm = tm.merge(app[['trackman_game_id', 'pitcher_trackman_id', 'role', 'app_pitches']],
              on=['trackman_game_id', 'pitcher_trackman_id'], how='left')
print('\n등판 수:', len(app), ' 역할 분포:', flush=True)
print(app.role.value_counts().to_string(), flush=True)
print(app.groupby('role').app_pitches.describe()[['count', 'mean', '50%', 'max']]
      .round(1).to_string(), flush=True)
REPORT['A_appearances'] = int(len(app))
REPORT['A_role_counts'] = app.role.value_counts().to_dict()

# --- A1 등판 내 투구수 -> 구속 (fastball 한정!)
fb = tm[(tm.pitch_type_group == 'fastball') & tm.rel_speed.notna()]
BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130]
fb = fb.copy()
fb['pb'] = pd.cut(fb.app_pitch, BINS)
a1 = fb.groupby(['role', 'pb'], observed=True).agg(
    n=('rel_speed', 'size'), speed=('rel_speed', 'mean'),
    spin=('spin_rate', 'mean'), ivb=('induced_vert_break', 'mean'),
    rel_h=('rel_height', 'mean'), ext=('extension', 'mean')).reset_index()
a1 = a1[a1.n >= 500]
a1.to_csv(f'{OUT}/e5_A1_within_appearance.csv', index=False)
print('\n=== A1: 등판 내 투구수별 fastball 구속 ===', flush=True)
print(a1.to_string(index=False), flush=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for role, col in zip(sorted(a1.role.unique()), PAL):
    s = a1[a1.role == role]
    x = range(len(s))
    axes[0].plot(x, s.speed, color=col, lw=2, marker='o', ms=4, label=role)
    axes[1].plot(x, s.rel_h, color=col, lw=2, marker='o', ms=4, label=role)
    axes[2].plot(x, s.ivb, color=col, lw=2, marker='o', ms=4, label=role)
    for ax in axes:
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels([str(i) for i in s.pb], rotation=45, fontsize=6)
axes[0].set_ylabel(T('구속 (km/h)', 'velocity'))
axes[0].set_title(T('A1. 등판 내 투구수 → 구속 (fastball만)',
                    'A1. velocity vs pitches thrown (fastball only)'), loc='left')
axes[1].set_ylabel(T('릴리스 높이', 'release height'))
axes[1].set_title(T('릴리스 높이', 'release height'), loc='left')
axes[2].set_ylabel(T('수직 무브먼트', 'vert break'))
axes[2].set_title(T('수직 무브먼트', 'vert break'), loc='left')
axes[0].legend(frameon=False, fontsize=7)
save(fig, 'e5_A1_within_appearance.png')

# --- A3 등판 간격
pa = app.dropna(subset=['gdate']).sort_values(['pitcher_trackman_id', 'gdate'])
pa['rest'] = pa.groupby('pitcher_trackman_id').gdate.diff().dt.days
tm = tm.merge(pa[['trackman_game_id', 'pitcher_trackman_id', 'rest']],
              on=['trackman_game_id', 'pitcher_trackman_id'], how='left')
fb = tm[(tm.pitch_type_group == 'fastball') & tm.rel_speed.notna()].copy()
fb['rest_b'] = pd.cut(fb.rest, [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 14.5, 400],
                      labels=['0(연투)' if KO else '0', '1', '2', '3', '4', '5',
                              '6-7', '8-14', '15+'])
a3 = fb.groupby(['role', 'rest_b'], observed=True).agg(
    n=('rel_speed', 'size'), speed=('rel_speed', 'mean'),
    rel_h=('rel_height', 'mean'), rel_sd=('rel_side', 'std'),
    ivb=('induced_vert_break', 'mean')).reset_index()
a3 = a3[a3.n >= 500]
a3.to_csv(f'{OUT}/e5_A3_rest_days.csv', index=False)
print('\n=== A3: 등판 간격(휴식일)별 ===', flush=True)
print(a3.to_string(index=False), flush=True)

# --- A4 타순 일순 (TTO)
tm['tto'] = tm.groupby(['trackman_game_id', 'pitcher_trackman_id',
                        'batter_trackman_id'], sort=False).cumcount() + 1
fb2 = tm[(tm.pitch_type_group == 'fastball') & tm.rel_speed.notna() & (tm.tto <= 4)]
a4 = fb2.groupby(['role', 'tto'], observed=True).agg(
    n=('rel_speed', 'size'), speed=('rel_speed', 'mean'),
    ivb=('induced_vert_break', 'mean')).reset_index()
a4 = a4[a4.n >= 500]
a4.to_csv(f'{OUT}/e5_A4_times_through_order.csv', index=False)
print('\n=== A4: 타순 일순(TTO)별 ===', flush=True)
print(a4.to_string(index=False), flush=True)

# --- A5 타석 내 순번
a5 = tm[tm.pitch_of_pa <= 8].groupby('pitch_of_pa').agg(
    n=('rel_speed', 'size'), speed=('rel_speed', 'mean'),
    fastball=('pitch_type_group', lambda s: float((s == 'fastball').mean()))).reset_index()
a5.to_csv(f'{OUT}/e5_A5_pitch_of_pa.csv', index=False)
print('\n=== A5: 타석 내 순번별 ===', flush=True)
print(a5.to_string(index=False), flush=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
for role, col in zip(sorted(a3.role.unique()), PAL):
    s = a3[a3.role == role]
    axes[0].plot(range(len(s)), s.speed, color=col, lw=2, marker='o', ms=4, label=role)
    axes[0].set_xticks(range(len(s)))
    axes[0].set_xticklabels([str(i) for i in s.rest_b], rotation=45, fontsize=6)
axes[0].set_title(T('A3. 휴식일 → 구속', 'A3. rest days -> velocity'), loc='left')
axes[0].legend(frameon=False, fontsize=7)
for role, col in zip(sorted(a4.role.unique()), PAL):
    s = a4[a4.role == role]
    axes[1].plot(s.tto, s.speed, color=col, lw=2, marker='o', ms=5, label=role)
axes[1].set_xlabel('TTO')
axes[1].set_title(T('A4. 타순 일순 → 구속', 'A4. times through order'), loc='left')
axes[2].plot(a5.pitch_of_pa, a5.speed, color=PAL[0], lw=2, marker='o', ms=5)
ax2 = axes[2].twinx()
ax2.plot(a5.pitch_of_pa, a5.fastball, color=PAL[1], lw=2, ls='--', marker='s', ms=4)
ax2.set_ylabel(T('fastball 비율(주황)', 'fastball share (orange)'), color=PAL[1])
axes[2].set_xlabel(T('타석 내 순번', 'pitch of PA'))
axes[2].set_title(T('A5. 타석 내 순번 → 구속·구종', 'A5. within-PA'), loc='left')
save(fig, 'e5_A3A4A5_rhythm.png')

# ============================================================
# Part B — 메인 데이터 등판 경계 복원 검증
# ============================================================
tr = pd.read_csv(f'{ROOT}/data/train.csv',
                 usecols=['row_id', 'pitcher_id', 'season', 'game_month',
                          'game_dayofweek', 'inning', 'pitcher_team_id',
                          'batter_team_id', 'asof_pitcher_n', 'control_success'],
                 low_memory=False)
lab = pd.read_parquet(LR0, columns=['lab_success', 'lab_reverse', 'lab_middle',
                                    'lab_ball'])
df = pd.concat([tr.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
df = df[df.lab_success.notna()].copy()
df['lab_other_fail'] = ((df.lab_success == 0) & (df.lab_reverse == 0)
                        & (df.lab_middle == 0)).astype(float)
LBL = ['lab_success', 'lab_reverse', 'lab_middle', 'lab_ball', 'lab_other_fail']
df['rid'] = df.row_id.str.extract(r'(\d+)').astype(int)
df = df.sort_values('rid')

# B0: 같은 투수의 연속 투구 사이 row_id 간격 분포
df['gap'] = df.groupby('pitcher_id').rid.diff()
g = df.gap.dropna()
qs = g.quantile([.5, .6, .7, .75, .8, .85, .9, .95, .99]).round(1)
print('\n=== B0: 같은 투수 연속 투구의 row_id 간격 분위수 ===', flush=True)
print(qs.to_string(), flush=True)
small = float((g <= 60).mean())
big = float((g >= 300).mean())
mid = float(((g > 60) & (g < 300)).mean())
print(f'gap<=60: {small:.4f}   60<gap<300: {mid:.4f}   gap>=300: {big:.4f}', flush=True)
bimodal = mid < 0.10
print(f'-> 이봉형 판정(중간대 10% 미만): {bimodal}', flush=True)
REPORT['B0_gap_quantiles'] = qs.to_dict()
REPORT['B0_share_small'] = round(small, 4)
REPORT['B0_share_mid'] = round(mid, 4)
REPORT['B0_share_big'] = round(big, 4)
REPORT['B0_bimodal'] = bool(bimodal)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.hist(np.log10(g[g > 0].clip(1, 1e5)), bins=120, color=PAL[0])
ax.axvline(np.log10(60), color=PAL[1], lw=1.5, ls='--')
ax.axvline(np.log10(300), color=PAL[1], lw=1.5, ls='--')
ax.set_xlabel(T('log10(row_id 간격)', 'log10(row_id gap)'))
ax.set_title(T('B0. 같은 투수 연속 투구의 row_id 간격 — 이봉형이면 등판 경계 복원 가능',
               'B0. row_id gap between same-pitcher pitches'), loc='left')
save(fig, 'e5_B0_gap_distribution.png')

THRESH = 150
df['new_app'] = (df.gap.isna() | (df.gap > THRESH)).astype(int)
df['app_id'] = df.groupby('pitcher_id').new_app.cumsum()
df['app_pitch_main'] = df.groupby(['pitcher_id', 'app_id']).cumcount() + 1
appm = df.groupby(['pitcher_id', 'app_id']).size()
print(f'\n복원 등판 수={len(appm)}  등판당 투구수 중앙값={appm.median():.0f}  '
      f'평균={appm.mean():.1f}  95%={appm.quantile(.95):.0f}', flush=True)
REPORT['B1_reconstructed_appearances'] = int(len(appm))
REPORT['B1_pitches_per_app'] = dict(median=float(appm.median()),
                                    mean=round(float(appm.mean()), 1),
                                    p95=float(appm.quantile(.95)),
                                    max=int(appm.max()))

df['apb'] = pd.cut(df.app_pitch_main, BINS)
b1 = df.groupby('apb', observed=True).agg(
    n=('lab_success', 'size'), **{c.replace('lab_', ''): (c, 'mean') for c in LBL})
b1 = b1[b1.n >= 2000].round(5)
b1.to_csv(f'{OUT}/e5_B1_main_within_appearance.csv')
print('\n=== B1: (복원) 등판 내 투구수별 실패 유형 ===', flush=True)
print(b1.to_string(), flush=True)

b2 = df.groupby('inning').agg(
    n=('lab_success', 'size'), **{c.replace('lab_', ''): (c, 'mean') for c in LBL})
b2 = b2[b2.n >= 5000].round(5)
b2.to_csv(f'{OUT}/e5_B2_inning.csv')
print('\n=== B2: 이닝별 실패 유형 ===', flush=True)
print(b2.to_string(), flush=True)

vd = []
for name, key in [('app_pitch_bucket', 'apb'), ('inning', 'inning')]:
    for c in LBL:
        bv = between_var(df, key, c)
        vd.append(dict(axis=name, label=c.replace('lab_', ''),
                       bv_x1e4=round(bv * 1e4, 4),
                       score_potential=round(1e5 * bv / REF, 1)))
vd = pd.DataFrame(vd).sort_values(['label', 'bv_x1e4'], ascending=[True, False])
vd.to_csv(f'{OUT}/e5_B3_variance.csv', index=False)
print('\n=== B3: 분산 (비교: pitcher_id 22.8, count→ball 31.6) ===', flush=True)
print(vd.to_string(index=False), flush=True)
REPORT['B3_variance'] = vd.to_dict('records')

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
x = range(len(b1))
for i, c in enumerate(['success', 'reverse', 'middle', 'other_fail']):
    axes[0].plot(x, b1[c], color=PAL[i], lw=2, marker='o', ms=4, label=c)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels([str(i) for i in b1.index], rotation=45, fontsize=6)
axes[0].set_title(T('B1. 등판 내 투구수 → 실패 유형', 'B1. within-appearance'), loc='left')
axes[0].legend(frameon=False, fontsize=7)
for i, c in enumerate(['success', 'reverse', 'middle', 'other_fail']):
    axes[1].plot(b2.index, b2[c], color=PAL[i], lw=2, marker='o', ms=4, label=c)
axes[1].set_xlabel(T('이닝', 'inning'))
axes[1].set_title(T('B2. 이닝 → 실패 유형', 'B2. by inning'), loc='left')
save(fig, 'e5_B_main_fatigue.png')

# ============================================================
# Part C — 투수 단위 피로 서명 x 실패 프로파일
# ============================================================
long_app = fb[fb.app_pitches >= 50]
sig = []
for pid, s in long_app.groupby('pitcher_trackman_id'):
    early = s[s.app_pitch <= 15].rel_speed
    late = s[s.app_pitch >= 55].rel_speed
    if len(early) >= 100 and len(late) >= 100:
        sig.append(dict(pitcher_trackman_id=pid, n_early=len(early), n_late=len(late),
                        speed_early=float(early.mean()), speed_late=float(late.mean()),
                        decay=float(late.mean() - early.mean())))
sig = pd.DataFrame(sig)
print(f'\n피로 서명 산출 투수 = {len(sig)}명 '
      f'(구속 변화 평균 {sig.decay.mean():+.3f} km/h, sd {sig.decay.std():.3f})', flush=True)
sig.to_csv(f'{OUT}/e5_C_fatigue_signature.csv', index=False)
REPORT['C_signature_pitchers'] = int(len(sig))
REPORT['C_decay_mean'] = round(float(sig.decay.mean()), 4)
REPORT['C_decay_sd'] = round(float(sig.decay.std()), 4)

pmap = pd.read_csv(f'{TM0}/pitcher_id_map.csv')
if 'purity' in pmap.columns:
    pmap = pmap[pmap.purity >= 0.99]
pmap = pmap[['pitcher_id', 'pitcher_tm']].rename(
    columns={'pitcher_tm': 'pitcher_trackman_id'}).dropna()
prof = df.groupby('pitcher_id').agg(
    n_main=('lab_success', 'size'),
    **{c.replace('lab_', ''): (c, 'mean') for c in LBL}).reset_index()
prof = prof[prof.n_main >= 500]
JC = prof.merge(pmap, on='pitcher_id').merge(sig, on='pitcher_trackman_id')
print(f'C: 피로서명+라벨 결합 투수 = {len(JC)}명', flush=True)
REPORT['C_joined'] = int(len(JC))

crow = {}
if len(JC) >= 40:
    for c in ['success', 'reverse', 'middle', 'other_fail']:
        crow[c] = round(float(np.corrcoef(JC.decay, JC[c])[0, 1]), 4)
    print('\n=== C: 구속 감소량(피로 서명) vs 실패 유형 상관 ===', flush=True)
    print(json.dumps(crow, ensure_ascii=False), flush=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    for ax, c in zip(axes, ['success', 'reverse', 'middle', 'other_fail']):
        ax.scatter(JC.decay, JC[c], s=np.sqrt(JC.n_main) / 3, alpha=0.55,
                   color=PAL[0], edgecolors='none')
        b, a = np.polyfit(JC.decay, JC[c], 1)
        xs = np.linspace(JC.decay.min(), JC.decay.max(), 20)
        ax.plot(xs, b * xs + a, color=PAL[1], lw=2)
        ax.set_xlabel(T('구속 감소 (km/h)', 'velocity decay'))
        ax.set_ylabel(c)
        ax.set_title(f'r={crow[c]:+.3f}', loc='left')
    fig.suptitle(T('C. 피로에 약한 투수가 제구도 나쁜가',
                   'C. fatigue signature vs command'), x=0.01, ha='left', fontsize=11)
    save(fig, 'e5_C_fatigue_vs_command.png')
REPORT['C_corr'] = crow

REPORT['stamp'] = STAMP
REPORT['runtime_sec'] = round(time.time() - t0, 1)
REPORT['usability'] = ('pitch_no/pitch_of_pa/game_date는 test에 없다. 직접 피처 불가. '
                       '투수 단위 피로 서명(구속 감소량)만 동결 아티팩트로 사용 가능.')
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(REPORT, f, indent=1, ensure_ascii=False, default=str)
with open(f'{OUT}/findings.md', 'w') as f:
    f.write('# E5 findings (auto)\n\n'
            + json.dumps(REPORT, indent=1, ensure_ascii=False, default=str))
print(f'\ndecision=E5_COMPLETE runtime={time.time()-t0:.0f}s', flush=True)
print('OUT_E5=' + OUT, flush=True)
