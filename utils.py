import warnings
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy.stats import levene, shapiro, kruskal, chi2_contingency, fisher_exact
import statsmodels.api as sm

import pingouin as pg


plt.style.use("dark_background")


def draw_bar_graph(data: pd.DataFrame, feature: str) -> None:
    counts = data[feature].value_counts()

    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=45)
    plt.ylabel("count")
    plt.xlabel(feature)
    plt.title(f"{feature} distribution")
    plt.show()


def draw_box_plot(data: pd.DataFrame, feature: str, target: str, order=None) -> None:
    plt.figure(figsize=(15, 12))
    # plt.boxplot(
    #     [data[data[feature] == c][target] for c in data[feature].unique()],
    #     labels=data[feature].unique()
    # )
    sns.boxplot(x=feature, y=target, data=data, order=order, hue=feature)
    plt.xticks(rotation=45)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"{target} distribution by {feature}")
    plt.show()


def anova_full_report(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    order_by_frequency: bool = True,
    show_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    All-in-one ANOVA + diagnostics + alternatives.

    Parameters
    ----------
    df : pd.DataFrame
        Full data. If split_train_frac is not None, we split and run tests on training part.
    cat_col : str
        Name of categorical column (can be object/str, or numeric that we treat as categorical).
    target_col : str
        Name of numeric target column.
    random_state : int
        Random state for split.
    order_by_frequency : bool
        Whether to order plots by frequency.
    show_plots : bool
        If True, display plots inline (not returned). Set False to only get numeric outputs.
    verbose : bool
        Print intermediate results as they are computed.

    Returns
    -------
    results : dict
        A dictionary containing ANOVA table, levene p, shapiro p, per-group normality summary,
        tukey object, kruskal result, welch result (if available), eta2, grouped dataframe used, model, etc.
    """
    # 1) basic checks
    if cat_col not in df.columns or target_col not in df.columns:
        raise ValueError("cat_col and target_col must be columns in df")

    df_train = df.copy()

    df_train["__cat_grp__"] = df_train[cat_col]

    # optionally order categories by freq for plotting
    if order_by_frequency:
        freq_order = df_train["__cat_grp__"].value_counts().index.tolist()
    else:
        freq_order = sorted(df_train["__cat_grp__"].cat.categories.tolist())

    # 1) plots: boxplot + violin
    if show_plots:
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            x="__cat_grp__",
            y=target_col,
            data=df_train,
            order=freq_order,
            hue="__cat_grp__",
        )
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Boxplot: {target_col} by {cat_col} (train)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.violinplot(
            x="__cat_grp__",
            y=target_col,
            data=df_train,
            order=freq_order,
            inner="quartile",
            hue="__cat_grp__",
        )
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Violin: {target_col} by {cat_col} (train)")
        plt.tight_layout()
        plt.show()

    # 3) fit OLS and ANOVA table
    formula = f"{target_col} ~ C(__cat_grp__)"
    model = smf.ols(formula, data=df_train).fit()
    anova_results = anova_lm(model)
    if verbose:
        print("\nANOVA table (OLS):")
        print(anova_results)

    # 4) compute residuals as used by ANOVA (within-group residuals)
    # Residuals from model are equivalent; but show both group-mean residuals and model residuals
    df_train["_group_mean_"] = df_train.groupby("__cat_grp__")[target_col].transform(
        "mean"
    )
    df_train["_resid_group_mean_"] = df_train[target_col] - df_train["_group_mean_"]
    df_train["_resid_model_"] = model.resid

    # 5) Levene test (homoscedasticity)
    groups = [g[target_col].values for _, g in df_train.groupby("__cat_grp__")]
    # filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    lev_stat, lev_p = levene(*groups)
    if verbose:
        print(f"\nLevene test for equal variances: stat={lev_stat:.4f}, p={lev_p:.4g}")

    # 6) Shapiro-Wilk on residuals (global), and per-group small-sample checks
    # Note: Shapiro is sensitive to large n; interpret with QQ-plot
    try:
        sh_stat, sh_p = shapiro(df_train["_resid_group_mean_"].dropna())
    except Exception as e:
        sh_stat, sh_p = np.nan, np.nan
        warnings.warn(f"Shapiro test failed: {e}")

    if verbose:
        print(
            f"\nShapiro-Wilk (residuals vs group means): stat={sh_stat:.4f}, p={sh_p:.4g}"
        )
        print(
            "  (Note: for large samples Shapiro almost always rejects small deviations. Inspect QQ plot.)"
        )

    # per-group shapiro summary (only for groups with n between 3 and 5000)
    shapiro_per_group = []
    for name, grp in df_train.groupby("__cat_grp__"):
        vals = grp["_resid_group_mean_"].dropna()
        n = len(vals)
        if n >= 3 and n <= 5000:
            try:
                s_stat, s_p = shapiro(vals)
                shapiro_per_group.append((name, n, s_stat, s_p))
            except Exception:
                shapiro_per_group.append((name, n, np.nan, np.nan))
        else:
            shapiro_per_group.append((name, n, np.nan, np.nan))

    shapiro_df = pd.DataFrame(
        shapiro_per_group, columns=["group", "n", "shapiro_stat", "shapiro_p"]
    ).sort_values("n", ascending=False)

    if verbose:
        print(
            "\nPer-group Shapiro summary (NaN p means group too small/large to test):"
        )
        print(shapiro_df.head(10).to_string(index=False))

    # 7) Q-Q plot + residual histogram + residuals vs fitted
    if show_plots:
        plt.figure(figsize=(6, 6))
        sm.qqplot(df_train["_resid_group_mean_"].dropna(), line="45", fit=True)
        plt.title("Q-Q plot: residuals (obs - group mean)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.hist(df_train["_resid_group_mean_"].dropna(), bins=50)
        plt.title("Histogram of residuals (obs - group mean)")
        plt.show()

        # residuals vs fitted (using model residuals)
        plt.figure(figsize=(8, 4))
        plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
        plt.axhline(0, color="k", linewidth=0.7)
        plt.xlabel("Fitted values")
        plt.ylabel("Model residuals")
        plt.title("Residuals vs Fitted (model.resid)")
        plt.show()

    # 8) Tukey HSD post-hoc (only if >1 level)
    tukey_result = None
    tukey_summary = None
    try:
        if df_train["__cat_grp__"].nunique() > 1:
            tukey_result = pairwise_tukeyhsd(
                endog=df_train[target_col], groups=df_train["__cat_grp__"], alpha=0.05
            )
            tukey_summary = tukey_result.summary()
            if verbose:
                print("\nTukey HSD pairwise comparisons (summary):")
                print(tukey_summary)
    except Exception as e:
        warnings.warn(f"Tukey HSD failed: {e}")

    # 9) Welch ANOVA (heteroscedasticity-safe) via pingouin if available
    welch_res = None
    try:
        welch_res = pg.welch_anova(dv=target_col, between="__cat_grp__", data=df_train)
        if verbose:
            print("\nWelch ANOVA (pingouin):")
            print(welch_res)
    except Exception as e:
        warnings.warn(f"pingouin welch_anova failed: {e}")
        welch_res = None
    # 10) Kruskal-Wallis (nonparametric)
    try:
        kw_stat, kw_p = kruskal(
            *[g[target_col].values for _, g in df_train.groupby("__cat_grp__")]
        )
        if verbose:
            print(f"\nKruskal-Wallis H-test: H={kw_stat:.4f}, p={kw_p:.4g}")
    except Exception as e:
        kw_stat, kw_p = np.nan, np.nan
        warnings.warn(f"Kruskal-Wallis failed: {e}")

    # 11) Effect size: eta-squared (SS_between / SS_total)
    try:
        ss_between = anova_results.loc["C(__cat_grp__)", "sum_sq"]
        ss_resid = anova_results.loc["Residual", "sum_sq"]
        eta2 = ss_between / (ss_between + ss_resid)
    except Exception:
        eta2 = np.nan

    if verbose:
        print(
            f"\nEffect size (eta-squared): {eta2:.4f}  (rough: 0.01 small, 0.06 medium, 0.14 large)"
        )

    # 12) Important statistics summary to return
    results = {
        "df_train": df_train,
        "anova_table": anova_results,
        "ols_model": model,
        "levene_stat": lev_stat,
        "levene_p": lev_p,
        "shapiro_stat": sh_stat,
        "shapiro_p": sh_p,
        "shapiro_per_group": shapiro_df,
        "tukey": tukey_result,
        "tukey_summary": tukey_summary,
        "welch_anova": welch_res,
        "kruskal_stat": kw_stat,
        "kruskal_p": kw_p,
        "eta2": eta2,
    }

    return results



def chi2_association_report(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    fisher_threshold: int = 5,
    permutation: int = 0,
    random_state: Optional[int] = None,
    show_plot: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Check association between two categorical columns using chi-square and diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the two categorical columns.
    col_a, col_b : str
        Column names (categorical features).
    fisher_threshold : int
        If smallest expected cell < fisher_threshold for a 2x2 table, run Fisher exact test.
    permutation : int
        Number of permutation repetitions to compute empirical p-value (0 to disable).
    random_state : Optional[int]
        Random seed for permutation test / reproducibility.
    show_plot : bool
        Show standardized residual heatmap.
    verbose : bool
        Print interpretive messages.

    Returns
    -------
    results : dict
        Keys include: contingency, observed, expected, chi2, p_value, dof, cramers_v,
        std_residuals (DataFrame), fisher (if used), permutation_p (if computed), recommendation.
    """
    # 0) basic checks
    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError("col_a and col_b must exist in df")

    # 1) Prepare data
    df_work = df[[col_a, col_b]].copy() # 3) Build contingency table
    contingency = pd.crosstab(df_work[col_a], df_work[col_b])
    observed = contingency.values
    r, c = contingency.shape

    # 4) Run χ² test (SciPy). Use Yates correction for 2x2 by default (chi2_contingency handles it).
    try:
        chi2, p, dof, expected = chi2_contingency(observed, correction=True)
    except Exception as e:
        raise RuntimeError(f"chi2_contingency failed: {e}")

    # 5) For 2x2 with small expected counts, compute Fisher exact (if applicable)
    fisher_res = None
    smallest_expected = expected.min() if expected.size > 0 else np.nan
    if r == 2 and c == 2 and smallest_expected < fisher_threshold:
        # fisher_exact expects a 2x2 table as [[a,b],[c,d]]
        try:
            oddsratio, fisher_p = fisher_exact(observed)
            fisher_res = dict(oddsratio=oddsratio, p_value=fisher_p)
        except Exception:
            fisher_res = None

    # 6) Permutation test (empirical p-value) if requested
    permutation_p = None
    if permutation and permutation > 0:
        rng = np.random.default_rng(random_state)
        observed_chi2 = chi2
        greater = 0
        flat_a = df_work[col_a].values
        flat_b = df_work[col_b].values
        for _ in range(permutation):
            # shuffle B and recompute chi2_on_perm
            rng.shuffle(flat_b)
            perm_tab = pd.crosstab(flat_a, flat_b).values
            try:
                chi2_perm, _, _, _ = chi2_contingency(perm_tab, correction=True)
            except Exception:
                chi2_perm = 0.0
            if chi2_perm >= observed_chi2:
                greater += 1
        # +1 for observed; use (greater+1)/(permutation+1)
        permutation_p = (greater + 1) / (permutation + 1)

    # 7) Compute effect size: Cramér's V
    n = observed.sum()
    denom = n * (min(r - 1, c - 1))
    cramers_v = np.sqrt(chi2 / denom) if denom > 0 else np.nan

    # 8) Standardized residuals: (O - E)/sqrt(E)
    std_resid = (observed - expected) / np.sqrt(expected)
    std_resid_df = pd.DataFrame(std_resid, index=contingency.index, columns=contingency.columns)

    # 9) Warning rules for small expected counts
    small_cells = (expected < 5).sum()
    pct_small = small_cells / expected.size if expected.size > 0 else 0.0
    small_expected_warning = pct_small > 0.2 or (expected.min() < 1)

    # 10) Basic recommendation heuristic (guideline, not absolute)
    # thresholds are heuristic: use domain & CV to confirm.
    if p < 0.05:
        if cramers_v >= 0.5:
            rec = "strong_association -> likely redundant; consider merging or dropping one column (verify with model/CV)."
        elif cramers_v >= 0.3:
            rec = "moderate_association -> treat as related; consider merging, creating joint feature, or dropping one if redundant."
        elif cramers_v >= 0.1:
            rec = "weak_association -> small association; probably keep both and validate with CV; consider encoding that preserves identity."
        else:
            rec = "statistically_significant_but_tiny_effect -> likely negligible in practice; deprioritize unless domain says otherwise."
    else:
        rec = "no_significant_association -> features likely independent; you can keep both but consider dropping one if redundant with other logic."

    if small_expected_warning:
        rec += " WARNING: many small expected counts — chi-square p-value may be unreliable. Consider merging sparse categories or using permutation test / Fisher exact (2x2)."

    # 11) Plot standardized residuals heatmap
    if show_plot:
        plt.figure(figsize=(max(6, c * 0.7), max(4, r * 0.5)))
        sns.heatmap(std_resid_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title(f"Standardized residuals ( (O-E)/sqrt(E) )\nchi2={chi2:.3f}, p={p:.3g}, cramers_v={cramers_v:.3f}")
        plt.xlabel(col_b)
        plt.ylabel(col_a)
        plt.tight_layout()
        plt.show()

    # 12) Verbose printed summary
    if verbose:
        print(f"chi2 = {chi2:.4f}, p = {p:.4g}, dof = {dof}")
        if fisher_res is not None:
            print(f"Fisher exact (2x2) oddsratio = {fisher_res['oddsratio']:.4f}, p = {fisher_res['p_value']:.4g}")
        if permutation_p is not None:
            print(f"Permutation empirical p-value (permutations={permutation}) = {permutation_p:.4g}")
        print(f"Cramér's V = {cramers_v:.4f}")
        if small_expected_warning:
            print("WARNING: small expected counts detected — interpret p-values with caution (consider merging categories).")
        print("Recommendation (heuristic):", rec)

    # 13) Build results dict
    results: Dict[str, Any] = {
        "contingency": contingency,
        "observed": observed,
        "expected": expected,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": float(cramers_v) if not np.isnan(cramers_v) else np.nan,
        "std_residuals": std_resid_df,
        "fisher": fisher_res,
        "permutation_p": float(permutation_p) if permutation_p is not None else None,
        "small_expected_warning": bool(small_expected_warning),
        "pct_small_expected": float(pct_small),
        "recommendation": rec,
    }

    return results
