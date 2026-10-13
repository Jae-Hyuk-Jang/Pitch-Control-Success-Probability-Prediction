# PROBE0 - reconstructed_labels.parquet 정면 파악 (EDA 전 단계)
#
# 목적: 라벨 파일의 실제 스키마·값 분포·정합성을 확정한다.
#  1) parquet 스키마, 행 수, dtype
#  2) 컬럼별 결측/고유값/값 분포
#  3) train.csv와의 정렬 방식 확인 (row_id 유무, 행 수, 순서)
#  4) 앵커: 복원 success == control_success 일치율
#  5) 라벨 간 배타성/동시발생 구조 (success vs reverse vs middle, 구종군 3분할)
#
# CPU, 약 1-2분. 결과: runs/reconstructed_label_probe_<stamp>/probe.md + stdout

import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = '.'
def _latest(pattern):
    """Newest run folder (or file) matching the glob; the unit that writes it must have been run first."""
    import glob
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else pattern


LR0 = _latest(ROOT + '/runs/latent_label_reconstruction_audit_v1_staged_*'
             '/stage1_reconstruction_integrity/reconstructed_labels.parquet')

STAMP = time.strftime('%Y%m%d_%H%M%S')
OUT = f'{ROOT}/runs/reconstructed_label_probe_{STAMP}'
os.makedirs(OUT, exist_ok=True)
print('OUT_PROBE0=' + OUT, flush=True)
t0 = time.time()
L = []


def emit(s=''):
    print(s, flush=True)
    L.append(str(s))


# ---- 1) schema without loading
pf = pq.ParquetFile(LR0)
emit(f'## parquet meta')
emit(f'rows={pf.metadata.num_rows} row_groups={pf.metadata.num_row_groups} '
     f'cols={pf.metadata.num_columns}')
emit(f'schema:')
for f_ in pf.schema_arrow:
    emit(f'  {f_.name}: {f_.type}')

# ---- 2) full load + per-column stats
lab = pf.read().to_pandas()
emit(f'\n## loaded {lab.shape} in {time.time()-t0:.0f}s  mem='
     f'{lab.memory_usage(deep=True).sum()/1e6:.0f}MB')
emit('\n## per-column stats')
for c in lab.columns:
    s = lab[c]
    nn = int(s.isna().sum())
    nu = int(s.nunique(dropna=True))
    line = f'  {c}: dtype={s.dtype} null={nn} ({nn/len(s):.5f}) nunique={nu}'
    if nu <= 12:
        vc = s.value_counts(dropna=False).head(12)
        line += ' | ' + ', '.join(f'{k}={v}' for k, v in vc.items())
    else:
        line += (f' | min={s.min()} max={s.max()} mean={pd.to_numeric(s, errors="coerce").mean():.6f}')
    emit(line)

emit('\n## head(10)')
emit(lab.head(10).to_string())

# ---- 3) alignment with train
tr = pd.read_csv(ROOT + '/data/train.csv',
                 usecols=['row_id', 'control_success'])
emit(f'\n## train rows={len(tr)}  label rows={len(lab)}  equal={len(tr)==len(lab)}')
if 'row_id' in lab.columns:
    same_order = bool((lab['row_id'].values == tr['row_id'].values).all())
    same_set = set(lab['row_id']) == set(tr['row_id'])
    emit(f'row_id present: same_order={same_order} same_set={same_set}')
else:
    emit('row_id ABSENT -> positional alignment assumed; verifying via anchor')

# ---- 4) anchor
cand = [c for c in lab.columns if 'success' in c and 'bsuccess' not in c
        and not c.startswith('asof')]
emit(f'\n## anchor candidates: {cand}')
for c in cand:
    m = lab[c].notna().values
    match = float((lab[c].values[m] == tr['control_success'].values[m]).mean())
    emit(f'  {c}: labelled={int(m.sum())} match_vs_control_success={match:.8f}')

# ---- 5) label structure
bin_cols = [c for c in lab.columns
            if lab[c].dropna().isin([0, 1]).all() and lab[c].notna().any()
            and c != 'row_id']
emit(f'\n## binary label columns: {bin_cols}')
sub = lab[bin_cols].dropna()
emit(f'complete-case rows={len(sub)}')
emit('\nmeans:')
emit(sub.mean().round(6).to_string())
emit('\npairwise co-occurrence P(a=1 & b=1):')
co = pd.DataFrame(index=bin_cols, columns=bin_cols, dtype=float)
v = sub.values.astype(float)
for i, a in enumerate(bin_cols):
    for j, b in enumerate(bin_cols):
        co.loc[a, b] = float((v[:, i] * v[:, j]).mean())
emit(co.round(4).to_string())

grp = [c for c in bin_cols if any(k in c for k in
       ('fastball', 'breaking', 'offspeed'))]
if len(grp) == 3:
    tot = sub[grp].sum(axis=1)
    emit(f'\npitch-group partition sum: '
         + json.dumps(tot.value_counts().to_dict()))

axA = [c for c in bin_cols if any(k in c for k in
       ('success', 'reverse', 'middle')) and 'b' + 'success' not in c
       and 'bmiddle' not in c]
emit(f'\naxis-A cols {axA}')
if axA:
    emit('axis-A row-sum distribution (배타성 확인):')
    emit(sub[axA].sum(axis=1).value_counts().sort_index().to_string())

with open(f'{OUT}/probe.md', 'w') as f:
    f.write('\n'.join(L))
print(f'\ndecision=PROBE0_COMPLETE runtime={time.time()-t0:.0f}s', flush=True)
print('OUT_PROBE0=' + OUT, flush=True)
