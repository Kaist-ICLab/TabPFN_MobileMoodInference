"""Repeat-level paired significance tests (complement to the geographic-unit tests).

STATISTICAL UNIT = the Monte Carlo repeat (the repeated user-level random holdout,
N = 10). The geographic-unit tests in ../geographic_tests/ ask whether the TabPFN
advantage generalizes ACROSS GEOGRAPHIC POPULATIONS; this analysis asks a DIFFERENT
question: is the aggregate advantage ROBUST TO THE RANDOMNESS OF THE REPEATED
USER-LEVEL HOLDOUT SPLIT? Neither test replaces the other.

Input (../data/repeat_level/rq{1,2}_repeat_values.csv): one row per
setting x task x budget x repeat, holding each model's aggregate AUROC for that
repeat, formed with the SAME aggregation the paper's Table 2 / Table 3 rows use
(unweighted mean over the setting's geographic units; pooled multi-country values
used directly). Within a repeat only sub-units valid (non-NaN) for BOTH models
enter the aggregate — identical unit sets on both sides, count recorded in
n_geographic_units_used; invalid repeats are dropped (valid_pair=False), nothing
is imputed. Repeat r of TabPFN is paired ONLY with repeat r of the baseline
(identical seed-0 split streams for both models).

Tests per cell: two-sided paired Wilcoxon signed-rank (PRIMARY) across the
repeat-level deltas; paired t-test + Cohen's dz as parametric robustness checks;
95% CI for the mean paired delta via the t-distribution. Holm-Bonferroni
corrections are reported in SEPARATE columns (raw p is never replaced):
(A) per-budget family = all setting x task cells of the same RQ at that budget;
(B) full-table family = all cells of the RQ's table. NOTE: with n = 10 the
smallest attainable Wilcoxon p is 2^-9 ~= 0.00195, so the full-table Holm
correction (40-60 cells) cannot fall below ~0.08-0.12 and is reported as a
sensitivity analysis only.

Run from this directory:  python3 run_repeat_level_tests.py
Outputs: results/rq{1,2}_repeat_level_tests.csv,
         results/rq{1,2}_geographic_vs_repeat_comparison.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VALS = os.path.join(REPO, "data", "repeat_level")
GEO = os.path.join(REPO, "geographic_tests", "results")
OUT = os.path.join(HERE, "results")

BUDGETS = [10, 20, 30, 40, 50]
ALPHA = 0.05


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (NaN-safe)."""
    p = np.asarray(pvals, float)
    adj = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    pv = p[mask]
    m = len(pv)
    order = np.argsort(pv)
    out = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pv[idx])
        out[idx] = min(1.0, running)
    adj[mask] = out
    return adj


def test_cells(vals, rq):
    rows = []
    for (setting, task, budget), g in vals.groupby(["setting", "task", "budget"]):
        d = g.loc[g["valid_pair"], "delta"].to_numpy()
        n = len(d)
        rec = {"rq": rq, "setting": setting, "task": task, "budget": budget,
               "baseline": "hm_latest RF (hybrid)" if rq == "RQ1" else "target-only RF",
               "n_valid": n,
               "tabpfn_mean": g.loc[g["valid_pair"], "tabpfn_auroc"].mean(),
               "baseline_mean": g.loc[g["valid_pair"], "baseline_auroc"].mean(),
               "mean_delta": np.mean(d) if n else np.nan,
               "median_delta": np.median(d) if n else np.nan,
               "delta_sd": np.std(d, ddof=1) if n > 1 else np.nan,
               "paired_se": np.std(d, ddof=1) / np.sqrt(n) if n > 1 else np.nan,
               "positive_repeats": int((d > 0).sum()),
               "negative_repeats": int((d < 0).sum()),
               "zero_repeats": int((d == 0).sum()),
               "wilcoxon_stat": np.nan, "wilcoxon_p_raw": np.nan,
               "paired_t_stat": np.nan, "paired_t_p": np.nan, "cohen_dz": np.nan,
               "ci95_low": np.nan, "ci95_high": np.nan}
        if n >= 5 and np.any(d != 0):          # signed-rank needs nonzero deltas & sane n
            w = stats.wilcoxon(d, alternative="two-sided")
            rec["wilcoxon_stat"], rec["wilcoxon_p_raw"] = float(w.statistic), float(w.pvalue)
        if n >= 2 and np.std(d, ddof=1) > 0:
            t = stats.ttest_rel(g.loc[g["valid_pair"], "tabpfn_auroc"],
                                g.loc[g["valid_pair"], "baseline_auroc"])
            rec["paired_t_stat"], rec["paired_t_p"] = float(t.statistic), float(t.pvalue)
            sd = np.std(d, ddof=1)
            rec["cohen_dz"] = float(np.mean(d) / sd)
            tcrit = stats.t.ppf(0.975, n - 1)
            rec["ci95_low"] = float(np.mean(d) - tcrit * sd / np.sqrt(n))
            rec["ci95_high"] = float(np.mean(d) + tcrit * sd / np.sqrt(n))
        rows.append(rec)
    out = pd.DataFrame(rows)
    # Holm A: per-budget family (all setting x task cells of this RQ at that budget)
    out["wilcoxon_p_holm_per_budget"] = np.nan
    for b in BUDGETS:
        m = out["budget"] == b
        out.loc[m, "wilcoxon_p_holm_per_budget"] = holm(out.loc[m, "wilcoxon_p_raw"])
    # Holm B: full-table family (all cells of this RQ)
    out["wilcoxon_p_holm_full_table"] = holm(out["wilcoxon_p_raw"])
    cols = ["rq", "setting", "task", "budget", "baseline", "n_valid", "tabpfn_mean",
            "baseline_mean", "mean_delta", "median_delta", "delta_sd", "paired_se",
            "positive_repeats", "negative_repeats", "zero_repeats", "wilcoxon_stat",
            "wilcoxon_p_raw", "wilcoxon_p_holm_per_budget", "wilcoxon_p_holm_full_table",
            "paired_t_stat", "paired_t_p", "cohen_dz", "ci95_low", "ci95_high"]
    return out[cols].sort_values(["setting", "task", "budget"]).reset_index(drop=True)


# Map repeat-level setting labels to the geographic-tests setting labels.
GEO_MAP_RQ1 = {"Country-specific": "Country-specific",
               "Continent-specific": "Continent-specific",
               "Country-Agnostic I": "Country-Agnostic I",
               "Country-Agnostic II": "Country-Agnostic II",
               "Multi-country natural": "Multi-country",
               "Multi-country balanced": "Multi-country"}
GEO_MAP_RQ2 = {"Country-specific": "Target-only (country-specific)"}


def compare_with_geo(tests, geo_csv, geo_map):
    geo = pd.read_csv(geo_csv)
    rows = []
    for _, r in tests.iterrows():
        gset = geo_map.get(r["setting"])
        sub = geo[(geo["setting"] == gset) & (geo["task"] == r["task"]) &
                  (geo["budget"] == r["budget"])] if gset else geo.iloc[0:0]
        geo_p = sub["wilcoxon_p"].iloc[0] if len(sub) else np.nan
        geo_testable = bool(sub["testable"].iloc[0]) if len(sub) else False
        rep_p = r["wilcoxon_p_raw"]
        geo_sig = geo_testable and not pd.isna(geo_p) and geo_p < ALPHA
        rep_sig = not pd.isna(rep_p) and rep_p < ALPHA
        if not geo_testable or pd.isna(geo_p):
            interp = "geographic-unit test unavailable; repeat-level available"
        elif geo_sig and rep_sig:
            interp = "significant under both"
        elif geo_sig:
            interp = "geographic-unit significant only"
        elif rep_sig:
            interp = "repeat-level significant only"
        else:
            interp = "neither significant"
        rows.append({"rq": r["rq"], "setting": r["setting"], "task": r["task"],
                     "budget": r["budget"], "mean_delta": r["mean_delta"],
                     "geo_unit": sub["unit"].iloc[0] if len(sub) else "n/a",
                     "geo_wilcoxon_p": geo_p, "geo_testable": geo_testable,
                     "repeat_wilcoxon_p": rep_p, "repeat_n_valid": r["n_valid"],
                     "interpretation": interp})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for rq, geo_csv, geo_map in (
            ("RQ1", os.path.join(GEO, "rq1_paired_tests.csv"), GEO_MAP_RQ1),
            ("RQ2", os.path.join(GEO, "rq2_paired_tests.csv"), GEO_MAP_RQ2)):
        tag = rq.lower()
        vals = pd.read_csv(os.path.join(VALS, f"{tag}_repeat_values.csv"))
        tests = test_cells(vals, rq)
        comp = compare_with_geo(tests, geo_csv, geo_map)
        tests.to_csv(os.path.join(OUT, f"{tag}_repeat_level_tests.csv"), index=False)
        comp.to_csv(os.path.join(OUT, f"{tag}_geographic_vs_repeat_comparison.csv"), index=False)
        n_sig = int((tests["wilcoxon_p_raw"] < ALPHA).sum())
        print(f"[{rq}] {len(tests)} cells -> {n_sig} raw-significant; wrote 2 CSVs")
