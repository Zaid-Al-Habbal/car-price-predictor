import warnings
import inspect
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy.stats import levene, shapiro, kruskal
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


import inspect
from typing import Any, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_tukey(
    tukey: Any, 
    ax: Optional[Axes] = None, 
    title: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot Tukey HSD test results.
    
    Args:
        tukey: Tukey test results object with plot() or plot_simultaneous() method
        ax: Optional matplotlib axes to plot on
        title: Optional title for the plot
        
    Returns:
        Tuple of (figure, axes)
        
    Raises:
        AttributeError: If tukey object doesn't have required plotting methods
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Try using the simple plot() method first
    if hasattr(tukey, "plot") and callable(tukey.plot):
        result = tukey.plot()
        if isinstance(result, Figure):
            fig = result
            ax = fig.axes[0] if fig.axes else ax
    
    # Fall back to plot_simultaneous() method
    elif hasattr(tukey, "plot_simultaneous") and callable(tukey.plot_simultaneous):
        sig = inspect.signature(tukey.plot_simultaneous)
        params = sig.parameters
        
        kwargs = {}
        
        # Add ax parameter if supported
        if "ax" in params:
            kwargs["ax"] = ax
        
        # Handle required comparison_name parameter
        if "comparison_name" in params:
            param = params["comparison_name"]
            # Check if parameter has no default (is required)
            if param.default is inspect.Parameter.empty:
                # Try to get a valid comparison name
                if hasattr(tukey, "groupsunique") and len(tukey.groupsunique) > 0:
                    kwargs["comparison_name"] = tukey.groupsunique[0]
                else:
                    # Fallback to first group or 0
                    kwargs["comparison_name"] = 0
        
        result = tukey.plot_simultaneous(**kwargs)
        if isinstance(result, Figure):
            fig = result
            ax = fig.axes[0] if fig.axes else ax
    
    else:
        raise AttributeError(
            "Unsupported Tukey results object: expected 'plot()' or "
            "'plot_simultaneous()' method."
        )
    
    # Set title if provided
    if title is not None:
        ax.set_title(title)
    
    return fig, ax