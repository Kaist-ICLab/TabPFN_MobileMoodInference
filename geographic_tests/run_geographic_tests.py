"""Paired significance tests over geographic evaluation units (paper Tables 2 & 3).

For every (setting, task, budget) cell this script computes, from the per-unit
aggregate AUROCs in ../data/:
  - the mean paired difference (TabPFN - RF/HM) with its paired SE and 95% CI
    (t-distribution),
  - a two-sided Wilcoxon signed-rank test (the primary, non-parametric paired test),
  - a paired t-test and Cohen's dz as parametric complements,
  - Holm-Bonferroni adjustment within the 50%-budget operating-point family.

Unit of analysis = the natural geographic evaluation unit of each setting:

  Country-specific            country            (n = 8)
  Country-Agnostic I          ordered source->target country pair (n = 56)
  Country-Agnostic II         held-out country   (n = 8)
  Continent-specific          continent          (n = 2  -> not testable)
  Multi-country (nat + bal)   pooled model       (n = 2  -> not testable)
  RQ2 target-only (country)   country            (n = 8)

Settings with fewer than MIN_UNITS (=3) units cannot support a signed-rank test
from the stored aggregates and are reported descriptively in the paper (marked
with a dagger in Tables 2 and 3). The paper reports mean AUROC with a MARGINAL
standard error (each model's dispersion across units); for a claim about the
DIFFERENCE between two models evaluated on identical splits the relevant
quantity is the PAIRED SE, which both are reported here for contrast.

Bold cells in paper Table 2 (RQ1) and Table 3 (RQ2) are cells with raw
two-sided Wilcoxon p < 0.05; at the 50% budget the text additionally notes
which cells survive the Holm correction (`wilcoxon_p_holm50` / `verdict`).

Run from this directory:  python3 run_geographic_tests.py
Outputs: results/rq1_paired_tests.csv, results/rq2_paired_tests.csv
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
OUT_DIR = os.path.join(HERE, "results")

RQ1_SOURCES = {
    "Country-specific": (f"{DATA}/rq1/v3_vs_hmlatest_country.csv", "country"),
    "Continent-specific": (f"{DATA}/rq1/v3_vs_hmlatest_continent.csv", "continent"),
    "Country-Agnostic I": (f"{DATA}/rq1/v3_vs_hmlatest_agnostic1.csv", "source-target pair"),
    "Country-Agnostic II": (f"{DATA}/rq1/v3_vs_hmlatest_agnostic2.csv", "held-out country"),
    "Multi-country": (f"{DATA}/rq1/v3_vs_hmlatest_multi.csv", "pooled model"),
}
RQ2_TARGET_ONLY = f"{DATA}/rq2/tabpfn_vs_hm_target_only.csv"

MIN_UNITS = 3     # a signed-rank test below this cannot reach any useful resolution
ALPHA = 0.05
BUDGETS = [10, 20, 30, 40, 50]


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def paired_row(setting: str, unit: str, task: str, budget: int,
               a: np.ndarray, b: np.ndarray) -> dict:
    """One paired comparison of TabPFN (a) against the RF baseline (b) over matched units."""
    diff = a - b
    n = len(diff)
    row = dict(setting=setting, unit=unit, task=task, budget=budget, n_units=n,
               mean_delta=diff.mean(), paired_se=np.nan, n_positive=int((diff > 0).sum()),
               wilcoxon_p=np.nan, ttest_p=np.nan, cohens_dz=np.nan,
               ci_lo=np.nan, ci_hi=np.nan, testable=n >= MIN_UNITS)
    if n < MIN_UNITS:
        return row
    se = diff.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=se)
    row.update(paired_se=se,
               wilcoxon_p=stats.wilcoxon(a, b).pvalue,
               ttest_p=stats.ttest_rel(a, b).pvalue,
               cohens_dz=diff.mean() / diff.std(ddof=1),
               ci_lo=ci[0], ci_hi=ci[1])
    return row


def budget_pct(df: pd.DataFrame) -> pd.Series:
    """Budget as an integer percentage, whichever way the file stores it."""
    col = next(c for c in df.columns if "pct" in c or "ratio" in c)
    s = df[col]
    return s.astype(int) if s.max() > 1 else (s * 100).round().astype(int)


def marginal_se(df: pd.DataFrame) -> tuple[float, float]:
    """The SEs the paper reports, for contrast with the paired SE."""
    return df.tabpfn_se.mean(), df.hm_se.mean()


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    rq1 = []
    for setting, (path, unit) in RQ1_SOURCES.items():
        df = pd.read_csv(path)
        df["budget"] = budget_pct(df)
        for budget in BUDGETS:
            for task in ("2c", "3c"):
                s = df[(df.budget == budget) & (df.task == task)]
                row = paired_row(setting, unit, task, budget,
                                 s.tabpfn_auroc.values, s.hm_auroc.values)
                row["marginal_se_tabpfn"], row["marginal_se_hm"] = marginal_se(s)
                rq1.append(row)

    t = pd.read_csv(RQ2_TARGET_ONLY)
    t["budget"] = budget_pct(t)
    rq2 = []
    for budget in BUDGETS:
        for task in ("2c", "3c"):
            s = t[(t.budget == budget) & (t.task == task)]
            row = paired_row("Target-only (country-specific)", "country", task, budget,
                             s.tabpfn_auroc.values, s.hm_auroc.values)
            row["marginal_se_tabpfn"], row["marginal_se_hm"] = marginal_se(s)
            rq2.append(row)
    return pd.DataFrame(rq1), pd.DataFrame(rq2)


def add_holm(df: pd.DataFrame) -> pd.DataFrame:
    """Holm correction within the 50% operating-point family (the headline comparisons)."""
    df["wilcoxon_p_holm50"] = np.nan
    mask = (df.budget == 50) & df.testable
    if mask.any():
        df.loc[mask, "wilcoxon_p_holm50"] = holm(df.loc[mask, "wilcoxon_p"].values)
    df["verdict"] = np.where(~df.testable, "not testable",
                             np.where(df.budget != 50, "",
                                      np.where(df.wilcoxon_p_holm50 < ALPHA,
                                               "significant", "not significant")))
    return df


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rq1, rq2 = collect()
    rq1, rq2 = add_holm(rq1), add_holm(rq2)
    rq1.to_csv(os.path.join(OUT_DIR, "rq1_paired_tests.csv"), index=False)
    rq2.to_csv(os.path.join(OUT_DIR, "rq2_paired_tests.csv"), index=False)

    pd.set_option("display.width", 250)
    show = ["setting", "task", "budget", "n_units", "mean_delta", "paired_se",
            "marginal_se_tabpfn", "n_positive", "wilcoxon_p", "wilcoxon_p_holm50", "verdict"]
    print("=== RQ1 (paper Table 2): all budgets, raw Wilcoxon p ===")
    print(rq1[rq1.testable][show[:-2] + ["cohens_dz"]].round(5).to_string(index=False))
    print("\n=== RQ1 (paper Table 2): 50% operating point, Holm-corrected ===")
    print(rq1[rq1.budget == 50][show].round(5).to_string(index=False))
    print("\n=== RQ2 (paper Table 3): target-only, country-specific ===")
    print(rq2[show].round(5).to_string(index=False))
    print(f"\nWrote {OUT_DIR}/rq1_paired_tests.csv and rq2_paired_tests.csv")


if __name__ == "__main__":
    main()
