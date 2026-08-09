# TabPFN_MobileMoodInference

Result data and statistical analysis for the paper **"Training-Free Few-Shot
Personalization for Mobile Mood Inference with TabPFN"** (UbiComp workshop
submission). The paper compares TabPFN-v3 in-context learning against the
DiversityOne Random Forest Hybrid Model (RQ1) and a target-only Random Forest
(RQ2) for mood-valence prediction from smartphone sensing, across five
personalization budgets (10–50%), six geographic configurations, and two
valence tasks (two-class, three-class).

This repository contains the **aggregate evaluation results** (macro AUROC per
geographic evaluation unit and per Monte Carlo repeat) and the **paired
statistical tests** behind the significance markers in the paper's Tables 2
and 3 — both the geographic-unit tests and the complementary repeat-level
tests. It contains **no raw or
per-participant DiversityOne data** — only derived model-performance metrics.

## Repository layout

```
data/
  rq1/                       per-unit AUROC comparisons, TabPFN vs Hybrid Model
    v3_vs_hmlatest_country.csv        8 countries      x task x budget
    v3_vs_hmlatest_continent.csv      2 continents     x task x budget
    v3_vs_hmlatest_agnostic1.csv      56 ordered pairs x task x budget
    v3_vs_hmlatest_agnostic2.csv      8 held-out ctys  x task x budget
    v3_vs_hmlatest_multi.csv          2 pooled models  x task x budget
  rq2/                       per-unit AUROC comparisons, target-only TabPFN vs RF
    tabpfn_vs_hm_target_only.csv            8 countries  x task x budget
    tabpfn_vs_hm_target_only_continent.csv  2 continents x task x budget
    tabpfn_vs_hm_target_only_multi.csv      2 pooled models x task x budget
  repeat_level/              per-repeat aggregate AUROCs (both models)
    rq1_repeat_values.csv    one row per setting x task x budget x repeat
    rq2_repeat_values.csv    one row per setting x task x budget x repeat
geographic_tests/
  run_geographic_tests.py    paired Wilcoxon / t-test / Holm over geographic units
  results/
    rq1_paired_tests.csv     one row per setting x task x budget (Table 2 markers)
    rq2_paired_tests.csv     one row per task x budget (Table 3 markers)
repeat_tests/
  run_repeat_level_tests.py  paired Wilcoxon / t-test / Holm over Monte Carlo repeats
  results/
    rq{1,2}_repeat_level_tests.csv             one row per setting x task x budget
    rq{1,2}_geographic_vs_repeat_comparison.csv  both tests side by side
make_paper_tables.py         regenerates the Table 2 / Table 3 delta matrices
```

Each `data/` CSV row holds one geographic evaluation unit at one task and
budget: both models' mean macro AUROC over 10 repeated user-level random
holdouts (identical splits for both models), their standard errors, and the
paired delta.

## Statistical testing (paper Tables 2 & 3)

The unit of analysis is the natural geographic evaluation unit of each setting:
country (n=8), ordered source→target country pair (n=56), held-out country
(n=8), continent (n=2), or pooled model (n=1 per variant). For every
setting × task × budget cell, `run_geographic_tests.py` computes the mean
paired delta with paired SE and 95% CI, a **two-sided Wilcoxon signed-rank
test** (primary), a paired t-test and Cohen's dz, and a Holm–Bonferroni
correction within the 50%-budget family.

- **Bold** cells in Tables 2 and 3 = raw Wilcoxon p < 0.05.
- **†** settings (continent-specific, multi-country) have fewer than three
  independent geographic units, cannot support a signed-rank test, and are
  reported descriptively.
- At the 50% budget, the paper additionally notes which cells remain
  significant after the Holm correction (`wilcoxon_p_holm50`, `verdict`).

## Repeat-level statistical testing (complementary analysis)

The geographic-unit tests above ask whether the advantage **generalizes across
geographic populations**. A second, complementary analysis in `repeat_tests/`
uses the **Monte Carlo repeat** (the repeated user-level random holdout, n=10)
as the statistical unit and asks whether the aggregate advantage is **robust to
which users happen to be selected** in the random holdout. Neither test
replaces the other.

For every setting × task × budget cell, one aggregate AUROC per model per
repeat is formed with the same aggregation as the paper's table rows
(unweighted mean over the setting's geographic units, restricted within each
repeat to units valid for both models; pooled multi-country values used
directly). Repeat *r* of TabPFN is paired only with repeat *r* of the baseline
(identical splits). The per-repeat values are shipped in `data/repeat_level/`;
`repeat_tests/run_repeat_level_tests.py` recomputes the two-sided paired
Wilcoxon signed-rank test (primary), paired t-test, Cohen's dz, 95% CI, and
Holm corrections (per-budget family and full-table family, in separate
columns — raw p is never replaced).

Headline outcomes:

- RQ1: 40/60 cells significant at raw p < .05 (22 survive the per-budget
  Holm); RQ2: 33/40 (all 33 survive the per-budget Holm).
- The two tests never disagree in the confirming direction: every
  geographic-significant cell is also repeat-significant. The repeat-level
  test additionally reaches significance in 6 RQ1 cells and covers the 30
  cells per RQ that are untestable at the geographic level (continent-specific
  n=2 units, multi-country n=1).
- With n=10 the smallest attainable Wilcoxon p is 2⁻⁹ ≈ 0.00195, so the
  full-table Holm correction can never fall below ≈ 0.08–0.12; it is reported
  as a sensitivity analysis only.
- Caveat: TabPFN inference is run-to-run nondeterministic for contexts above
  its 10k-row pretraining limit. This mainly affects the Country-Agnostic II
  setting (every context pools 7 countries), where run noise is of the same
  order as the deltas — its repeat-level p-values should be read as indicative
  (they are largely non-significant, consistent with the geographic-unit
  test). All other settings' repeat noise (~1e-3) is negligible relative to
  the tested deltas (~1e-2 to 4e-2).

## Reproduce

```bash
pip install -r requirements.txt
python3 geographic_tests/run_geographic_tests.py   # rewrites geographic_tests/results/
python3 repeat_tests/run_repeat_level_tests.py     # rewrites repeat_tests/results/
python3 make_paper_tables.py                       # prints Table 2 / Table 3 deltas
```

## Experimental protocol (summary)

Full details are in the paper. In brief: 10 repeated user-level random
holdouts; per repeat, ~20% of eligible users are held out as unseen target
users and a fixed stratified 50% of each target user's records is reserved as
the test set (constant across budgets); target-support records are drawn at
10–50% of each user's eligible records; the Hybrid Model drops one random
source record per added support record so context size stays fixed; TabPFN-v3
(2 ensemble members, fixed seed, pretraining row limit disabled) receives the
identical context in a single forward pass. Both models use identical source
users, target users, support records, test records, features (~102 reconstructed
DiversityOne features), and median imputation fit on context rows only. Metric:
macro AUROC (one-vs-rest for three-class).

## Data availability

The DiversityOne dataset is available from its authors under approved
research-use conditions (Busso et al., 2025, IMWUT). This repository
redistributes no sensing or self-report data — only aggregate model
evaluation metrics derived from it.

## License

Code and derived result tables are released under the MIT License (see
`LICENSE`).
