"""
132_regression_runner.py

One-shot regression runner for deal-level analysis.
Runs all standard models (bivariate, full, robustness, heterogeneity)
and outputs results in table format.

Usage:
  # Run all models on the default dataset
  python -m src.scripts.132_regression_runner

  # Custom dataset
  python -m src.scripts.132_regression_runner --file data/cb_insights/deal_dataset.csv

  # Specific DVs only
  python -m src.scripts.132_regression_runner --dvs follow_on,exit

  # Skip heterogeneity analysis
  python -m src.scripts.132_regression_runner --skip-heterogeneity
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cb_insights", "deal_dataset.csv"
)


def prep_data(df):
    """Prepare dataset for regression."""
    # Exclude GVC_solo
    df = df[df["investment_pattern"] != "GVC_solo"].copy()

    # Variables
    df["log_deal_size"] = np.log(df["deal_size_m"].clip(lower=0.01))
    df.loc[df["deal_size_m"].isna(), "log_deal_size"] = np.nan

    df["deal_year"] = pd.to_numeric(df["deal_year"], errors="coerce")
    df = df[(df["deal_year"] >= 2010) & (df["deal_year"] <= 2025)].copy()
    df["deal_year"] = df["deal_year"].astype(int)

    df["log_syndicate"] = np.log(df["syndicate_size"].clip(lower=1))

    for col in ["exit", "follow_on", "round_progression", "has_cvc"]:
        df[col] = df[col].astype(int)

    df["gvc_co_invest"] = df["treatment"].astype(int)

    # Industry grouping
    top_ind = df["industry"].fillna("Unknown").value_counts().head(10).index.tolist()
    df["industry_group"] = df["industry"].fillna("Unknown").apply(
        lambda x: x if x in top_ind else "Other"
    )

    return df


def run_model(df, formula, cluster_col="company_id"):
    """Run OLS with clustered SE, return key results."""
    try:
        model = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
        )
        return model
    except Exception as e:
        return None


def format_coef(model, var, decimals=4):
    """Format coefficient with significance stars."""
    if model is None:
        return "—", "—", "—"
    coef = model.params.get(var)
    pval = model.pvalues.get(var)
    se = model.bse.get(var)
    if coef is None:
        return "—", "—", "—"
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    return f"{coef:+.{decimals}f}{stars}", f"({se:.{decimals}f})", f"[p={pval:.3f}]"


def print_table_row(label, values, widths=None):
    """Print a formatted table row."""
    if widths is None:
        widths = [25] + [20] * len(values)
    parts = [f"{label:<{widths[0]}}"]
    for i, v in enumerate(values):
        w = widths[i + 1] if i + 1 < len(widths) else 20
        parts.append(f"{v:>{w}}")
    print("".join(parts))


def run_main_results(df, dvs):
    """Table 1: Main results with full controls."""
    print(f"\n{'='*80}")
    print("TABLE 1: MAIN RESULTS (Full model, clustered SE)")
    print(f"{'='*80}")

    df_full = df.dropna(subset=["log_deal_size"]).copy()
    widths = [25] + [22] * len(dvs)

    print_table_row("", dvs, widths)
    print("-" * (25 + 22 * len(dvs)))

    formula_template = "{dv} ~ gvc_co_invest + log_deal_size + log_syndicate + C(stage) + C(deal_year) + C(country) + C(industry_group)"

    models = {}
    for dv in dvs:
        formula = formula_template.format(dv=dv)
        models[dv] = run_model(df_full, formula)

    # GVC coefficient
    coefs = [format_coef(models[dv], "gvc_co_invest")[0] for dv in dvs]
    print_table_row("GVC_CoInvest", coefs, widths)

    ses = [format_coef(models[dv], "gvc_co_invest")[1] for dv in dvs]
    print_table_row("", ses, widths)

    pvals = [format_coef(models[dv], "gvc_co_invest")[2] for dv in dvs]
    print_table_row("", pvals, widths)

    print("-" * (25 + 22 * len(dvs)))
    print_table_row("Controls", ["Yes"] * len(dvs), widths)
    print_table_row("Stage FE", ["Yes"] * len(dvs), widths)
    print_table_row("Year FE", ["Yes"] * len(dvs), widths)
    print_table_row("Country FE", ["Yes"] * len(dvs), widths)
    print_table_row("Industry FE", ["Yes"] * len(dvs), widths)
    ns = [f"{int(models[dv].nobs)}" if models[dv] else "—" for dv in dvs]
    print_table_row("N", ns, widths)
    r2s = [f"{models[dv].rsquared:.3f}" if models[dv] else "—" for dv in dvs]
    print_table_row("R²", r2s, widths)

    return models


def run_robustness(df, main_dv="follow_on"):
    """Table 2: Robustness checks for the main DV."""
    print(f"\n{'='*80}")
    print(f"TABLE 2: ROBUSTNESS ({main_dv})")
    print(f"{'='*80}")

    df_full = df.dropna(subset=["log_deal_size"]).copy()

    specs = {
        "(1) Full model": f"{main_dv} ~ gvc_co_invest + log_deal_size + log_syndicate + C(stage) + C(deal_year) + C(country) + C(industry_group)",
        "(2) No country FE": f"{main_dv} ~ gvc_co_invest + log_deal_size + log_syndicate + C(stage) + C(deal_year) + C(industry_group)",
        "(3) No industry FE": f"{main_dv} ~ gvc_co_invest + log_deal_size + log_syndicate + C(stage) + C(deal_year) + C(country)",
        "(4) No deal controls": f"{main_dv} ~ gvc_co_invest + C(stage) + C(deal_year) + C(country) + C(industry_group)",
    }

    # Minimal controls uses full sample (no deal_size requirement)
    specs_full_sample = {
        "(5) Minimal (full sample)": f"{main_dv} ~ gvc_co_invest + C(stage) + C(deal_year) + C(country)",
    }

    print(f"\n{'Specification':<30} {'GVC coef':>12} {'p-value':>10} {'N':>8}")
    print("-" * 60)

    for label, formula in specs.items():
        m = run_model(df_full, formula)
        if m:
            coef = m.params["gvc_co_invest"]
            pval = m.pvalues["gvc_co_invest"]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"{label:<30} {coef:>+.4f}{stars:3s} {pval:>10.3f} {int(m.nobs):>8}")

    for label, formula in specs_full_sample.items():
        m = run_model(df, formula)
        if m:
            coef = m.params["gvc_co_invest"]
            pval = m.pvalues["gvc_co_invest"]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"{label:<30} {coef:>+.4f}{stars:3s} {pval:>10.3f} {int(m.nobs):>8}")


def run_heterogeneity(df, dvs):
    """Table 3-4: Stage and country subsamples."""
    df_full = df.dropna(subset=["log_deal_size"]).copy()

    for dv in dvs:
        print(f"\n{'='*80}")
        print(f"TABLE 3: STAGE HETEROGENEITY ({dv})")
        print(f"{'='*80}")

        print(f"\n{'Stage':<15} {'GVC coef':>12} {'p-value':>10} {'N':>8} {'T':>6}")
        print("-" * 55)

        for stage in ["Seed", "Series A", "Series B"]:
            sub = df_full[df_full["stage"] == stage]
            t_count = sub["gvc_co_invest"].sum()
            if t_count < 10:
                print(f"{stage:<15} {'skip':>12} {'(T<10)':>10} {len(sub):>8} {t_count:>6}")
                continue
            formula = f"{dv} ~ gvc_co_invest + log_deal_size + log_syndicate + C(deal_year) + C(country) + C(industry_group)"
            m = run_model(sub, formula)
            if m:
                coef = m.params["gvc_co_invest"]
                pval = m.pvalues["gvc_co_invest"]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"{stage:<15} {coef:>+.4f}{stars:3s} {pval:>10.3f} {int(m.nobs):>8} {t_count:>6}")

        print(f"\n{'='*80}")
        print(f"TABLE 4: COUNTRY HETEROGENEITY ({dv})")
        print(f"{'='*80}")

        print(f"\n{'Country':<15} {'GVC coef':>12} {'p-value':>10} {'N':>8} {'T':>6}")
        print("-" * 55)

        for country in ["JP", "KR", "SG"]:
            sub = df_full[df_full["country"] == country]
            t_count = sub["gvc_co_invest"].sum()
            if t_count < 10:
                print(f"{country:<15} {'skip':>12} {'(T<10)':>10} {len(sub):>8} {t_count:>6}")
                continue
            formula = f"{dv} ~ gvc_co_invest + log_deal_size + log_syndicate + C(stage) + C(deal_year) + C(industry_group)"
            m = run_model(sub, formula)
            if m:
                coef = m.params["gvc_co_invest"]
                pval = m.pvalues["gvc_co_invest"]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"{country:<15} {coef:>+.4f}{stars:3s} {pval:>10.3f} {int(m.nobs):>8} {t_count:>6}")


def run_interaction(df, dvs):
    """Table 5: GVC × CVC interaction."""
    print(f"\n{'='*80}")
    print("TABLE 5: GVC × CVC INTERACTION")
    print(f"{'='*80}")

    df_full = df.dropna(subset=["log_deal_size"]).copy()
    df_full["gvc_x_cvc"] = df_full["gvc_co_invest"] * df_full["has_cvc"]

    for dv in dvs:
        formula = f"{dv} ~ gvc_co_invest + has_cvc + gvc_x_cvc + log_deal_size + log_syndicate + C(stage) + C(deal_year) + C(country) + C(industry_group)"
        m = run_model(df_full, formula)
        if m:
            print(f"\n  {dv}:")
            for var in ["gvc_co_invest", "has_cvc", "gvc_x_cvc"]:
                coef, se, pval = format_coef(m, var)
                print(f"    {var:<20} {coef:>15}  {se}  {pval}")


def main():
    parser = argparse.ArgumentParser(description="Regression runner for deal-level analysis")
    parser.add_argument("--file", type=str, default=DEFAULT_DATASET, help="Path to deal dataset CSV")
    parser.add_argument("--dvs", type=str, default="follow_on,exit,round_progression",
                        help="Comma-separated DV names")
    parser.add_argument("--skip-heterogeneity", action="store_true", help="Skip heterogeneity analysis")
    parser.add_argument("--skip-interaction", action="store_true", help="Skip GVC×CVC interaction")
    args = parser.parse_args()

    dvs = [x.strip() for x in args.dvs.split(",")]

    print(f"Loading: {args.file}")
    df = pd.read_csv(args.file)
    df = prep_data(df)

    print(f"Sample: {len(df)} deals (Treatment={df['gvc_co_invest'].sum()}, Control={(df['gvc_co_invest']==0).sum()})")

    # Table 1: Main results
    run_main_results(df, dvs)

    # Table 2: Robustness (for first DV)
    run_robustness(df, main_dv=dvs[0])

    # Table 3-4: Heterogeneity
    if not args.skip_heterogeneity:
        run_heterogeneity(df, dvs)

    # Table 5: Interaction
    if not args.skip_interaction:
        run_interaction(df, dvs)

    print(f"\n{'='*80}")
    print("REGRESSION COMPLETE")
    print(f"{'='*80}")


@dataclass
class RegressionResult:
    """Structured regression results for 108 orchestrator."""
    dataset_path: str = ""
    sample_size: int = 0
    treatment_count: int = 0
    control_count: int = 0
    main_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # dv -> {coef, pval, se, n, r2}
    all_failed: bool = False

    @property
    def passed(self) -> bool:
        return not self.all_failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "sample_size": self.sample_size,
            "treatment_count": self.treatment_count,
            "control_count": self.control_count,
            "main_results": self.main_results,
            "all_failed": self.all_failed,
            "passed": self.passed,
        }


def run_regression(dataset_path: str, dvs: Optional[List[str]] = None,
                   skip_heterogeneity: bool = False,
                   skip_interaction: bool = False) -> RegressionResult:
    """Library interface: run full regression suite and return structured results.

    Called by 108 orchestrator for Phase 3b.
    """
    if dvs is None:
        dvs = ["follow_on", "exit", "round_progression"]

    result = RegressionResult(dataset_path=dataset_path)

    df = pd.read_csv(dataset_path)
    df = prep_data(df)

    result.sample_size = len(df)
    result.treatment_count = int(df["gvc_co_invest"].sum())
    result.control_count = int((df["gvc_co_invest"] == 0).sum())

    print(f"Sample: {len(df)} deals (Treatment={result.treatment_count}, Control={result.control_count})")

    # Main results
    models = run_main_results(df, dvs)

    succeeded = 0
    for dv in dvs:
        m = models.get(dv)
        if m is not None:
            succeeded += 1
            result.main_results[dv] = {
                "coef": float(m.params.get("gvc_co_invest", 0)),
                "pval": float(m.pvalues.get("gvc_co_invest", 1)),
                "se": float(m.bse.get("gvc_co_invest", 0)),
                "n": int(m.nobs),
                "r2": float(m.rsquared),
            }
        else:
            result.main_results[dv] = {"coef": None, "pval": None, "se": None, "n": 0, "r2": None}

    result.all_failed = succeeded == 0

    # Robustness
    run_robustness(df, main_dv=dvs[0])

    # Heterogeneity
    if not skip_heterogeneity:
        run_heterogeneity(df, dvs)

    # Interaction
    if not skip_interaction:
        run_interaction(df, dvs)

    print(f"\n{'='*80}")
    print("REGRESSION COMPLETE")
    print(f"{'='*80}")

    return result


if __name__ == "__main__":
    main()
