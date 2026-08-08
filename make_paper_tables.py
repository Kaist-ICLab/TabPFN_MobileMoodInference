"""Regenerate the paper's Table 2 (RQ1) and Table 3 (RQ2) delta matrices.

Each cell is Delta AUROC = TabPFN - RF baseline, aggregated over the setting's
geographic units (unweighted mean; pooled multi-country settings are a single
model). Values match the paper at 3 decimals. Bolding in the paper follows the
raw two-sided Wilcoxon p < 0.05 from geographic_tests/results/*.csv; daggered
settings (< 3 independent geographic units) are reported descriptively.

Run from this directory:  python3 make_paper_tables.py
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BUDGETS = [10, 20, 30, 40, 50]


def budget_pct(df: pd.DataFrame) -> pd.Series:
    col = next(c for c in df.columns if "pct" in c or "ratio" in c)
    s = df[col]
    return s.astype(int) if s.max() > 1 else (s * 100).round().astype(int)


def deltas(path: str, variant_col: str | None = None, variant: str | None = None) -> dict:
    df = pd.read_csv(path)
    if variant_col is not None:
        df = df[df[variant_col] == variant]
    df["budget"] = budget_pct(df)
    out = {}
    for task in ("2c", "3c"):
        for b in BUDGETS:
            s = df[(df.task == task) & (df.budget == b)]
            out[(task, b)] = s["delta_tabpfn_minus_hm"].mean()
    return out


def print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{title}")
    header = "| Geographic setting | " + " | ".join(
        f"{t} {b}%" for t in ("2c", "3c") for b in BUDGETS) + " |"
    print(header)
    print("|---" + "|---:" * 10 + "|")
    for name, d in rows:
        cells = " | ".join(f"{d[(t, b)]:+.3f}" for t in ("2c", "3c") for b in BUDGETS)
        print(f"| {name} | {cells} |")


def main() -> None:
    rq1 = [
        ("Country-specific", deltas(f"{DATA}/rq1/v3_vs_hmlatest_country.csv")),
        ("Continent-specific", deltas(f"{DATA}/rq1/v3_vs_hmlatest_continent.csv")),
        ("Country-Agnostic I", deltas(f"{DATA}/rq1/v3_vs_hmlatest_agnostic1.csv")),
        ("Country-Agnostic II", deltas(f"{DATA}/rq1/v3_vs_hmlatest_agnostic2.csv")),
        ("Multi-country natural",
         deltas(f"{DATA}/rq1/v3_vs_hmlatest_multi.csv", "model", "Multi-Country")),
        ("Multi-country balanced",
         deltas(f"{DATA}/rq1/v3_vs_hmlatest_multi.csv", "model", "Multi-Country (Bal)")),
    ]
    rq2 = [
        ("Country-specific", deltas(f"{DATA}/rq2/tabpfn_vs_hm_target_only.csv")),
        ("Continent-specific", deltas(f"{DATA}/rq2/tabpfn_vs_hm_target_only_continent.csv")),
        ("Multi-country natural",
         deltas(f"{DATA}/rq2/tabpfn_vs_hm_target_only_multi.csv", "variant", "Multi-Country")),
        ("Multi-country balanced",
         deltas(f"{DATA}/rq2/tabpfn_vs_hm_target_only_multi.csv", "variant", "Multi-Country (Bal)")),
    ]
    print_table("Table 2 - RQ1: Delta AUROC = TabPFN - Hybrid Model", rq1)
    print_table("Table 3 - RQ2 target-only: Delta AUROC = TabPFN - RF", rq2)


if __name__ == "__main__":
    main()
