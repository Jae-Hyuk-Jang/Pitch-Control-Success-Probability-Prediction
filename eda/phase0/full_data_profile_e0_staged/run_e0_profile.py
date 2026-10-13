# E0 - 기초 전수 프로파일링: train.csv 49컬럼 + trackman_history.csv 30컬럼
#
# 산출물 (runs/full_data_profile_e0_<stamp>/):
#   train_column_stats.csv / trackman_column_stats.csv  컬럼별 전수 통계
#   PNG 차트: 분포(수치/범주), 결측, 시즌 드리프트, 상관 히트맵, 타깃 연관,
#             trackman 물리량(구종군별), trackman 드리프트, 퓨처스 비중
#   findings.md + summary.json  자동 플래그 (중복컬럼, 준상수, 고결측, 고드리프트)
#
# CPU ~5-10분.

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = '.'
STAMP = time.strftime('%Y%m%d_%H%M%S')
OUT = f'{ROOT}/runs/full_data_profile_e0_{STAMP}'
os.makedirs(OUT, exist_ok=True)
print('OUT_E0=' + OUT, flush=True)
t0 = time.time()

# ---- plotting palette
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efeb", "#eb6834"])
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


def col_stats(df, target=None):
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        is_num = num.notna().sum() > 0.5 * len(s)
        r = dict(column=c, dtype=str(s.dtype), n_missing=int(s.isna().sum()),
                 miss_rate=round(float(s.isna().mean()), 5),
                 nunique=int(s.nunique(dropna=True)))
        if is_num:
            r.update(min=float(num.min()), max=float(num.max()),
                     mean=round(float(num.mean()), 5), std=round(float(num.std()), 5),
                     skew=round(float(num.skew()), 3))
            if target is not None and c != target.name:
                m = num.notna() & target.notna()
                r["corr_target"] = (round(float(np.corrcoef(num[m], target[m])[0, 1]), 5)
                                    if m.sum() > 1 and num[m].std() > 0 else np.nan)
        top = s.value_counts(dropna=True).head(5)
        r["top5"] = "; ".join(f"{k}:{v}" for k, v in top.items())
        rows.append(r)
    return pd.DataFrame(rows)


def hist_pages(df, cols, prefix, color):
    per = 16
    for pg in range(0, len(cols), per):
        chunk = cols[pg:pg + per]
        fig, axes = plt.subplots(4, 4, figsize=(13, 9))
        for ax, c in zip(axes.ravel(), chunk):
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if v.nunique() > 1:
                lo, hi = v.quantile([0.001, 0.999])
                ax.hist(v.clip(lo, hi), bins=60, color=color)
            ax.set_title(c, loc="left")
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        save(fig, f"{prefix}_hist_p{pg // per + 1}.png")


# ============================== TRAIN ==============================
tr = pd.read_csv(f'{ROOT}/data/train.csv', low_memory=False)
print(f'train {tr.shape} in {time.time()-t0:.0f}s', flush=True)
y = pd.to_numeric(tr['control_success'], errors="coerce")

st = col_stats(tr, target=y)
st.to_csv(f'{OUT}/train_column_stats.csv', index=False)
print(st.drop(columns=['top5']).to_string(index=False), flush=True)

num_cols = [c for c in tr.columns
            if pd.api.types.is_numeric_dtype(tr[c]) and c not in ('row_id',)]
cat_cols = [c for c in tr.columns if tr[c].nunique(dropna=True) <= 30
            and c not in ('control_success',)]

hist_pages(tr, num_cols, 'train', PAL[0])

# categorical bars
fig, axes = plt.subplots(3, 4, figsize=(13, 8))
for ax, c in zip(axes.ravel(), cat_cols[:12]):
    vc = tr[c].value_counts(dropna=False).head(12).sort_index()
    ax.bar([str(k)[:6] for k in vc.index], vc.values, color=PAL[0])
    ax.set_title(c, loc="left")
    ax.tick_params(axis='x', rotation=45)
for ax in axes.ravel()[len(cat_cols[:12]):]:
    ax.axis("off")
save(fig, 'train_cat_bars.png')

# missing
miss = st[st.n_missing > 0].sort_values('miss_rate', ascending=False)
if len(miss):
    fig, ax = plt.subplots(figsize=(8, max(2, 0.3 * len(miss))))
    ax.barh(miss.column[::-1], miss.miss_rate[::-1], color=PAL[0])
    ax.set_title('missing rate by column', loc='left')
    save(fig, 'train_missing.png')

# season drift: z-scored means for top-16 drifting numeric cols
g = tr.groupby('season')
means = g[num_cols].mean()
stds = tr[num_cols].std().replace(0, np.nan)
drift = ((means.max() - means.min()) / stds).sort_values(ascending=False)
drift.to_csv(f'{OUT}/train_season_drift_score.csv')
top16 = [c for c in drift.index if c != 'season'][:16]
fig, axes = plt.subplots(4, 4, figsize=(13, 9))
for ax, c in zip(axes.ravel(), top16):
    ax.plot(means.index, (means[c] - means[c].mean()) / stds[c],
            color=PAL[0], lw=2, marker='o', ms=3)
    ax.axvline(2022.5, color=INK2, lw=0.8, ls=':')
    ax.set_title(f'{c} (drift={drift[c]:.2f}sd)', loc='left')
save(fig, 'train_season_drift.png')

# correlation heatmap + near-duplicate pairs
cm = tr[num_cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm.values, cmap=DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=90, fontsize=6)
ax.set_yticks(range(len(num_cols)))
ax.set_yticklabels(num_cols, fontsize=6)
ax.grid(False)
fig.colorbar(im, shrink=0.7)
ax.set_title('train numeric correlation', loc='left')
save(fig, 'train_corr_heatmap.png')
dupes = [(a, b, round(float(cm.loc[a, b]), 6))
         for i, a in enumerate(num_cols) for b in num_cols[i + 1:]
         if abs(cm.loc[a, b]) > 0.999]
print('near-duplicate pairs:', dupes, flush=True)

# target association: |corr| for numeric
assoc = st.dropna(subset=['corr_target']) if 'corr_target' in st else pd.DataFrame()
if len(assoc):
    a = assoc.reindex(assoc.corr_target.abs().sort_values(ascending=False).index).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(a.column[::-1], a.corr_target[::-1],
            color=[PAL[0] if v >= 0 else PAL[1] for v in a.corr_target[::-1]])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title('corr with control_success (top 20 |r|)  blue=+ orange=-', loc='left')
    save(fig, 'train_target_assoc.png')

del tr
# ============================== TRACKMAN ==============================
tm = pd.read_csv(f'{ROOT}/data/trackman_history.csv', low_memory=False)
print(f'trackman {tm.shape} in {time.time()-t0:.0f}s', flush=True)
stm = col_stats(tm)
stm.to_csv(f'{OUT}/trackman_column_stats.csv', index=False)
print(stm.drop(columns=['top5']).to_string(index=False), flush=True)

phys = [c for c in ('rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
                    'extension', 'rel_height', 'rel_side', 'zone_speed')
        if c in tm.columns]
hist_pages(tm, [c for c in tm.columns if pd.api.types.is_numeric_dtype(tm[c])],
           'trackman', PAL[2])

# physics by pitch group (fixed palette order: fastball/breaking/offspeed)
groups = ['fastball', 'breaking', 'offspeed']
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for ax, c in zip(axes.ravel(), phys):
    for gname, col in zip(groups, PAL[:3]):
        v = pd.to_numeric(tm.loc[tm['pitch_type_group'] == gname, c],
                          errors='coerce').dropna()
        if len(v) > 100:
            lo, hi = v.quantile([0.005, 0.995])
            ax.hist(v.clip(lo, hi), bins=50, density=True, histtype='step',
                    lw=2, color=col, label=gname)
    ax.set_title(c, loc='left')
axes.ravel()[0].legend(frameon=False, fontsize=7)
for ax in axes.ravel()[len(phys):]:
    ax.axis("off")
save(fig, 'trackman_physics_by_group.png')

# physics drift by season
pm = tm.groupby('season')[phys].mean()
ps = tm[phys].std().replace(0, np.nan)
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for ax, c in zip(axes.ravel(), phys):
    ax.plot(pm.index, (pm[c] - pm[c].mean()) / ps[c], color=PAL[2], lw=2,
            marker='o', ms=3)
    ax.set_title(c, loc='left')
save(fig, 'trackman_physics_drift.png')

# futures share (MIN_*) by season
tm['is_futures'] = tm['pitcher_team'].astype(str).str.startswith('MIN')
fut = tm.groupby('season')['is_futures'].mean()
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(fut.index.astype(str), fut.values, color=PAL[0])
for x_, v_ in zip(range(len(fut)), fut.values):
    ax.text(x_, v_, f'{v_:.0%}', ha='center', va='bottom', fontsize=7, color=INK2)
ax.set_title('trackman futures(MIN_*) share by season', loc='left')
save(fig, 'trackman_futures_share.png')

# pitch type tagged vs auto agreement
if {'tagged_pitch_type', 'auto_pitch_type'} <= set(tm.columns):
    both = tm.dropna(subset=['tagged_pitch_type', 'auto_pitch_type'])
    agree = float((both['tagged_pitch_type'].astype(str)
                   == both['auto_pitch_type'].astype(str)).mean())
else:
    agree = None

# ---- findings
flags = dict(
    stamp=STAMP,
    train_shape=[1475092, 49],
    near_duplicate_numeric_pairs=dupes,
    near_constant=[c for c in num_cols
                   if st.set_index('column').loc[c, 'nunique'] <= 3],
    high_missing=miss[miss.miss_rate > 0.01].column.tolist() if len(miss) else [],
    top_drift={c: round(float(drift[c]), 3) for c in top16[:8]},
    top_target_corr=(a.set_index('column').corr_target.round(4).to_dict()
                     if len(assoc) else {}),
    trackman_tagged_auto_agreement=agree,
    trackman_futures_share=round(float(tm['is_futures'].mean()), 4),
    runtime_sec=round(time.time() - t0, 1))
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(flags, f, indent=1, ensure_ascii=False)
with open(f'{OUT}/findings.md', 'w') as f:
    f.write('# E0 findings (auto)\n\n' + json.dumps(flags, indent=1, ensure_ascii=False))

print(f'\ndecision=E0_COMPLETE runtime={time.time()-t0:.0f}s', flush=True)
print('OUT_E0=' + OUT, flush=True)
