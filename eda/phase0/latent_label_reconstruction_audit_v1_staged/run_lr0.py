"""LR0 stage1 - can the per-pitch latent labels be recovered exactly?

Run:  python run_lr0.py

No model is trained. Three questions, in order, each able to stop the stage:

1. Precision. Is `rate * n` close enough to an integer that rounding
   recovers the true cumulative count? Measured per counter and per n
   decile, because drift grows with n.
2. Recoverability. For consecutive rows of the same entity, does the
   counter advance by exactly one pitch?
3. Correctness. Does the reconstructed success increment reproduce the
   official `control_success` column? This is the anchor: if it does, the
   other increments are trustworthy new labels; if it does not, nothing
   downstream is safe.

Then the taxonomy itself is measured rather than assumed. The profile means
(success .535, reverse .215, middle .142, ball .372, strike .442) sum to
1.71, so the five control counters are NOT a partition. The per-row
increment sums reported here say what the actual structure is.

Outputs the reconstructed labels to disk so later steps can consume them without
recomputation.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lr0_config as C  # noqa: E402


def _log(m):
    print(m, flush=True)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def precision_report(df, n_col, rate_cols):
    """How far is rate * n from an integer, overall and by n decile."""
    n = _num(df[n_col]).astype(float)
    dec = pd.qcut(n, 10, labels=False, duplicates="drop")
    out = {}
    for label, col in rate_cols.items():
        c = _num(df[col]).astype(float) * n
        resid = (c - np.round(c)).abs()
        ok = resid < C.ROUND_TOLERANCE
        by = pd.DataFrame({"d": dec, "r": resid, "ok": ok}).groupby("d").agg(
            resid_p50=("r", "median"), resid_p99=("r", lambda x: x.quantile(.99)),
            ok_rate=("ok", "mean"))
        out[label] = dict(
            resid_p50=float(np.nanmedian(resid)),
            resid_p99=float(np.nanquantile(resid, .99)),
            resid_max=float(np.nanmax(resid)),
            within_tolerance=float(np.nanmean(ok)),
            by_n_decile={int(k): dict(resid_p50=float(v.resid_p50),
                                      resid_p99=float(v.resid_p99),
                                      ok_rate=float(v.ok_rate))
                         for k, v in by.iterrows()})
    return out


def reconstruct(df, key, n_col, rate_cols):
    """Per-pitch increments from consecutive rows of the same entity."""
    n = _num(df[n_col]).astype(float)
    g = df.groupby(key, sort=False)
    n_next = g[n_col].shift(-1).astype(float)
    step_ok = (n_next - n).eq(1.0)

    inc = {}
    for label, col in rate_cols.items():
        c = np.round(_num(df[col]).astype(float) * n)
        tmp = pd.Series(c, index=df.index, name="_c")
        c_next = tmp.groupby(df[key].to_numpy()).shift(-1)
        d = (c_next - tmp)
        inc[label] = d.where(step_ok)
    return inc, step_ok


def main():
    t0 = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(C.RUNS_DIR) / f"{C.RUN_PREFIX}_{stamp}"
    out_dir = run_dir / C.STAGE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"OUT_LR0={run_dir}")

    usecols = [C.SEASON, C.TARGET, C.PITCHER, C.BATTER]
    for n_col, rates in C.COUNTERS.values():
        usecols += [n_col] + list(rates.values())
    df = pd.read_csv(C.TRAIN_PATH, usecols=sorted(set(usecols)), low_memory=False)
    _log(f"rows={len(df)} (file order preserved; row_id is chronological)")

    # ---- 1. precision ---------------------------------------------------
    prec = {}
    for fam, (n_col, rates) in C.COUNTERS.items():
        prec[fam] = precision_report(df, n_col, rates)
        worst = min(v["within_tolerance"] for v in prec[fam].values())
        _log(f"precision {fam}: min within_tolerance={worst:.6f}  "
             + " ".join(f"{k}={v['resid_p99']:.2e}" for k, v in prec[fam].items()))
    with open(out_dir / "precision_report.json", "w", encoding="utf-8") as f:
        json.dump(prec, f, indent=2)

    worst_prec = min(v["within_tolerance"]
                     for fam in prec.values() for v in fam.values())
    if worst_prec < 0.5:
        _log("STOP: rate*n does not round to integers; CSV precision is "
             "insufficient and reconstruction is impossible.")
        with open(out_dir / "gate.json", "w", encoding="utf-8") as f:
            json.dump(dict(stage="LR0", passed=False,
                           decision="LR0_INSUFFICIENT_CSV_PRECISION",
                           worst_within_tolerance=worst_prec), f, indent=2)
        return

    # ---- 2 & 3. reconstruct and validate --------------------------------
    labels, recov = {}, {}
    for fam, (n_col, rates) in C.COUNTERS.items():
        key = C.ENTITY_KEY[fam]
        inc, step_ok = reconstruct(df, key, n_col, rates)
        recov[fam] = float(step_ok.mean())
        labels.update({f"lab_{k}": v for k, v in inc.items()})
        _log(f"reconstruct {fam}: consecutive-step rate={recov[fam]:.6f}")

    lab = pd.DataFrame(labels)
    binary_ok = {c: float(lab[c].isin([0.0, 1.0]).mean()) for c in lab.columns}
    _log("increment in {0,1} rate: "
         + " ".join(f"{k.replace('lab_','')}={v:.5f}" for k, v in binary_ok.items()))

    y = _num(df[C.TARGET]).astype(float)
    m = lab["lab_success"].notna()
    agreement = float((lab.loc[m, "lab_success"] == y[m]).mean())
    _log(f"ANCHOR success agreement with control_success = {agreement:.8f} "
         f"on {int(m.sum())} rows")

    # ---- taxonomy: what actually partitions? ----------------------------
    ctrl = ["lab_success", "lab_reverse", "lab_middle", "lab_ball", "lab_strike"]
    mix = ["lab_fastball", "lab_breaking", "lab_offspeed"]
    tax = {}
    sub = lab.loc[m, ctrl]
    tax["control_sum_distribution"] = {
        str(k): int(v) for k, v in sub.sum(axis=1).value_counts().head(10).items()}
    tax["control_marginals"] = {c: float(sub[c].mean()) for c in ctrl}
    mm = lab[mix].notna().all(axis=1)
    tax["pitchmix_sum_distribution"] = {
        str(k): int(v) for k, v in lab.loc[mm, mix].sum(axis=1)
        .value_counts().head(10).items()}
    tax["pitchmix_marginals"] = {c: float(lab.loc[mm, c].mean()) for c in mix}
    ct = pd.crosstab(lab.loc[m, "lab_success"],
                     [lab.loc[m, "lab_ball"], lab.loc[m, "lab_strike"]])
    tax["success_by_ball_strike"] = json.loads(ct.to_json())
    for c in ctrl:
        if c == "lab_success":
            continue
        tax[f"success_vs_{c.replace('lab_','')}"] = json.loads(
            pd.crosstab(lab.loc[m, "lab_success"], lab.loc[m, c]).to_json())
    _log("control increment sum: " + json.dumps(tax["control_sum_distribution"]))
    _log("pitchmix increment sum: " + json.dumps(tax["pitchmix_sum_distribution"]))
    _log("control marginals: "
         + " ".join(f"{k.replace('lab_','')}={v:.4f}"
                    for k, v in tax["control_marginals"].items()))
    _log("pitchmix marginals: "
         + " ".join(f"{k.replace('lab_','')}={v:.4f}"
                    for k, v in tax["pitchmix_marginals"].items()))

    # coverage by season, to show which folds are usable
    lab[C.SEASON] = df[C.SEASON].to_numpy()
    cov = lab.groupby(C.SEASON)[ctrl + mix].apply(lambda g: g.notna().mean())
    cov.to_csv(out_dir / "label_coverage_by_season.csv")
    _log("label coverage by season:\n" + cov.round(4).to_string())

    lab.drop(columns=[C.SEASON]).astype("float32").to_parquet(
        out_dir / "reconstructed_labels.parquet", index=False)

    with open(out_dir / "reconstruction_report.json", "w", encoding="utf-8") as f:
        json.dump(dict(consecutive_step_rate=recov, increment_binary_rate=binary_ok,
                       success_agreement=agreement,
                       n_validated_rows=int(m.sum()), taxonomy=tax), f, indent=2)

    passed = bool(agreement >= C.GATE_MIN_SUCCESS_AGREEMENT
                  and recov["pitcher"] >= C.GATE_MIN_RECOVERABLE)
    decision = ("LR0_LABEL_RECONSTRUCTION_CONFIRMED" if passed
                else "LR0_RECONSTRUCTION_UNRELIABLE")
    gate = dict(
        stage="LR0", passed=passed, decision=decision,
        success_agreement=agreement, gate_min_agreement=C.GATE_MIN_SUCCESS_AGREEMENT,
        consecutive_step_rate=recov, gate_min_recoverable=C.GATE_MIN_RECOVERABLE,
        worst_precision_within_tolerance=worst_prec,
        increment_binary_rate=binary_ok, taxonomy=tax,
        n_new_label_columns=int(lab.shape[1] - 1),
        labels_path=str(out_dir / "reconstructed_labels.parquet"),
        rule_basis=("future-row reference inside train.csv is allowed for training "
                    "targets only; model inputs remain row-local"),
        trains_model=False, uses_other_test_rows=False, reads_trackman=False,
    )
    with open(out_dir / "gate.json", "w", encoding="utf-8") as f:
        json.dump(gate, f, indent=2)

    (out_dir / "findings.md").write_text("\n".join([
        "# LR0 - latent label reconstruction integrity", "",
        f"- decision: **{decision}**",
        f"- anchor: reconstructed success vs control_success = {agreement:.8f} "
        f"on {int(m.sum())} rows",
        f"- consecutive-step rate: {json.dumps(recov)}",
        f"- worst precision within tolerance: {worst_prec:.6f}", "",
        "## taxonomy", "",
        f"- control increment sum: {json.dumps(tax['control_sum_distribution'])}",
        f"- pitchmix increment sum: {json.dumps(tax['pitchmix_sum_distribution'])}",
        f"- control marginals: {json.dumps(tax['control_marginals'])}",
        f"- pitchmix marginals: {json.dumps(tax['pitchmix_marginals'])}", "",
        "The five control counters are not a partition (profile means sum to",
        "1.71). The increment sums above say what the real structure is, which",
        "determines whether to train a single multiclass head, several binary",
        "heads, or a two-factor target.",
    ]), encoding="utf-8")

    with open(run_dir / "stageLR0_runtime.json", "w", encoding="utf-8") as f:
        json.dump(dict(stage_requested="LR0", status="completed", root=C.ROOT,
                       output=str(run_dir), started_unix=t0,
                       elapsed_seconds=time.time() - t0, result=gate), f, indent=2)

    _log(f"decision={decision}")
    _log(f"OUT_LR0={run_dir}")


if __name__ == "__main__":
    main()
