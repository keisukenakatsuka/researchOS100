"""
133_psm.py

Propensity Score Matching (PSM) for GVC co-investment analysis.

Addresses selection bias: GVC may systematically select different deals than PVC.
PSM matches GVC-backed deals to observably similar PVC-only deals, then re-estimates
treatment effects on the matched sample.

Usage:
  # Default: nearest-neighbor matching with caliper
  python -m src.scripts.133_psm

  # Custom dataset / DVs
  python -m src.scripts.133_psm --file data/cb_insights/deal_dataset.csv \
    --dvs follow_on,exit,round_progression

  # Specify number of matches
  python -m src.scripts.133_psm --n-matches 3
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cb_insights", "deal_dataset.csv"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cb_insights"
)


def prep_data(df):
    """Prepare dataset for PSM (same as 132 prep)."""
    df = df[df["investment_pattern"] != "GVC_solo"].copy()

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


def estimate_propensity_scores(df):
    """Estimate propensity scores using logistic regression.

    Covariates: stage, country, deal_year, industry_group, log_syndicate, has_cvc.
    log_deal_size excluded to maximize sample size (26% missing).
    """
    import statsmodels.api as sm

    # Encode categoricals — sanitize column names for formula compatibility
    df_model = df.copy()
    df_model = pd.get_dummies(df_model, columns=["stage", "country", "industry_group"],
                               drop_first=True, dtype=int)
    # Sanitize column names (replace spaces/special chars with underscores)
    df_model.columns = [c.replace(" ", "_").replace("&", "and").replace("-", "_")
                        for c in df_model.columns]

    # Build covariate list
    covariates = [c for c in df_model.columns
                  if c.startswith(("stage_", "country_", "industry_group_"))]
    covariates += ["log_syndicate", "has_cvc"]

    # Add deal_year dummies
    df_model = pd.get_dummies(df_model, columns=["deal_year"], drop_first=True, dtype=int)
    year_cols = [c for c in df_model.columns if c.startswith("deal_year_")]
    covariates += year_cols

    # Drop rows with missing covariates
    all_cols = covariates + ["gvc_co_invest"]
    df_clean = df_model.dropna(subset=all_cols).copy()

    # Logistic regression (matrix-based, not formula, to avoid column name issues)
    X = sm.add_constant(df_clean[covariates].values)
    y = df_clean["gvc_co_invest"].values
    model = sm.Logit(y, X).fit(disp=0)

    df_clean["pscore"] = model.predict(X)

    print(f"\n  Propensity Score Model:")
    print(f"    N = {len(df_clean)} (Treatment = {df_clean['gvc_co_invest'].sum()}, "
          f"Control = {(df_clean['gvc_co_invest'] == 0).sum()})")
    print(f"    Pseudo R2 = {model.prsquared:.4f}")
    print(f"    Covariates: {len(covariates)}")

    # Propensity score distribution
    treated = df_clean[df_clean["gvc_co_invest"] == 1]["pscore"]
    control = df_clean[df_clean["gvc_co_invest"] == 0]["pscore"]
    print(f"\n  Propensity Score Distribution:")
    print(f"    Treatment: mean={treated.mean():.4f}, sd={treated.std():.4f}, "
          f"range=[{treated.min():.4f}, {treated.max():.4f}]")
    print(f"    Control:   mean={control.mean():.4f}, sd={control.std():.4f}, "
          f"range=[{control.min():.4f}, {control.max():.4f}]")

    # Common support
    overlap_min = max(treated.min(), control.min())
    overlap_max = min(treated.max(), control.max())
    in_support = df_clean[(df_clean["pscore"] >= overlap_min) &
                           (df_clean["pscore"] <= overlap_max)]
    print(f"\n  Common Support:")
    print(f"    Range: [{overlap_min:.4f}, {overlap_max:.4f}]")
    print(f"    In support: {len(in_support)}/{len(df_clean)} "
          f"({len(in_support)/len(df_clean)*100:.1f}%)")

    return df_clean, model, covariates


def nearest_neighbor_match(df, n_matches=1, caliper=0.05):
    """Perform nearest-neighbor matching on propensity scores.

    Args:
        df: DataFrame with 'pscore' and 'gvc_co_invest' columns
        n_matches: number of control matches per treated unit
        caliper: maximum allowed distance in propensity score
    """
    treated = df[df["gvc_co_invest"] == 1].copy()
    control = df[df["gvc_co_invest"] == 0].copy()

    matched_pairs = []
    used_controls = set()

    # Sort treated by pscore to reduce search space
    treated_sorted = treated.sort_values("pscore")

    for idx, t_row in treated_sorted.iterrows():
        t_pscore = t_row["pscore"]

        # Find closest controls within caliper
        available = control[~control.index.isin(used_controls)]
        if available.empty:
            continue

        distances = (available["pscore"] - t_pscore).abs()
        within_caliper = distances[distances <= caliper]

        if within_caliper.empty:
            continue

        # Select n closest
        closest = within_caliper.nsmallest(n_matches)

        for c_idx in closest.index:
            matched_pairs.append({
                "treated_idx": idx,
                "control_idx": c_idx,
                "distance": abs(t_pscore - control.loc[c_idx, "pscore"]),
            })
            used_controls.add(c_idx)

    # Build matched dataset
    treated_matched_idx = list(set(p["treated_idx"] for p in matched_pairs))
    control_matched_idx = list(set(p["control_idx"] for p in matched_pairs))

    matched_df = pd.concat([
        df.loc[treated_matched_idx],
        df.loc[control_matched_idx],
    ])

    n_treated = len(treated_matched_idx)
    n_control = len(control_matched_idx)
    match_rate = n_treated / len(treated) * 100

    print(f"\n  Matching Results (NN, k={n_matches}, caliper={caliper}):")
    print(f"    Treated matched: {n_treated}/{len(treated)} ({match_rate:.1f}%)")
    print(f"    Control matched: {n_control}")
    print(f"    Matched sample:  {len(matched_df)}")

    if matched_pairs:
        avg_dist = np.mean([p["distance"] for p in matched_pairs])
        max_dist = np.max([p["distance"] for p in matched_pairs])
        print(f"    Avg distance:    {avg_dist:.4f}")
        print(f"    Max distance:    {max_dist:.4f}")

    return matched_df, matched_pairs


def check_balance(df_full, df_matched, covariates_to_check):
    """Check covariate balance before and after matching.

    Reports standardized mean differences (SMD).
    """
    print(f"\n{'='*80}")
    print("COVARIATE BALANCE CHECK")
    print(f"{'='*80}")
    print(f"\n{'Variable':<30} {'SMD Before':>12} {'SMD After':>12} {'Improved':>10}")
    print("-" * 64)

    for var in covariates_to_check:
        if var not in df_full.columns or var not in df_matched.columns:
            continue

        # Before matching
        t_before = df_full[df_full["gvc_co_invest"] == 1][var].dropna()
        c_before = df_full[df_full["gvc_co_invest"] == 0][var].dropna()
        if len(t_before) == 0 or len(c_before) == 0:
            continue
        pooled_sd = np.sqrt((t_before.var() + c_before.var()) / 2)
        smd_before = (t_before.mean() - c_before.mean()) / pooled_sd if pooled_sd > 0 else 0

        # After matching
        t_after = df_matched[df_matched["gvc_co_invest"] == 1][var].dropna()
        c_after = df_matched[df_matched["gvc_co_invest"] == 0][var].dropna()
        if len(t_after) == 0 or len(c_after) == 0:
            continue
        pooled_sd_after = np.sqrt((t_after.var() + c_after.var()) / 2)
        smd_after = (t_after.mean() - c_after.mean()) / pooled_sd_after if pooled_sd_after > 0 else 0

        improved = "yes" if abs(smd_after) < abs(smd_before) else "no"
        flag = " *" if abs(smd_after) > 0.1 else ""

        print(f"  {var:<28} {smd_before:>+.4f}      {smd_after:>+.4f}      {improved}{flag}")

    print("\n  * SMD > 0.1 after matching (residual imbalance)")


def run_att_estimation(df_matched, dvs):
    """Estimate ATT on matched sample."""
    print(f"\n{'='*80}")
    print("ATT ESTIMATION ON MATCHED SAMPLE")
    print(f"{'='*80}")

    n_t = df_matched["gvc_co_invest"].sum()
    n_c = (df_matched["gvc_co_invest"] == 0).sum()
    print(f"\n  Matched sample: N={len(df_matched)} (T={n_t}, C={n_c})")

    print(f"\n{'DV':<25} {'ATT':>10} {'SE':>10} {'p-value':>10} {'95% CI':>22}")
    print("-" * 80)

    results = {}
    for dv in dvs:
        treated = df_matched[df_matched["gvc_co_invest"] == 1][dv].dropna()
        control = df_matched[df_matched["gvc_co_invest"] == 0][dv].dropna()

        if len(treated) == 0 or len(control) == 0:
            print(f"  {dv:<25} {'—':>10}")
            continue

        att = treated.mean() - control.mean()
        # Welch's t-test
        t_stat, pval = stats.ttest_ind(treated, control, equal_var=False)
        se = att / t_stat if abs(t_stat) > 1e-10 else 0
        ci_low = att - 1.96 * abs(se)
        ci_high = att + 1.96 * abs(se)

        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""

        print(f"  {dv:<25} {att:>+.4f}{stars:3s} {abs(se):>10.4f} {pval:>10.3f} "
              f"[{ci_low:>+.4f}, {ci_high:>+.4f}]")

        results[dv] = {
            "att": float(att), "se": float(abs(se)), "pval": float(pval),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "n_treated": int(len(treated)), "n_control": int(len(control)),
        }

    return results


def run_regression_on_matched(df_matched, dvs):
    """Run OLS regression on matched sample for comparison."""
    import statsmodels.api as sm

    print(f"\n{'='*80}")
    print("OLS ON MATCHED SAMPLE (with residual controls)")
    print(f"{'='*80}")

    df_m = df_matched.copy()

    # Build dummy columns from the dummified data
    stage_cols = [c for c in df_m.columns if c.startswith("stage_")]
    country_cols = [c for c in df_m.columns if c.startswith("country_")]
    year_cols = [c for c in df_m.columns if c.startswith("deal_year_")]
    control_cols = ["gvc_co_invest", "log_syndicate", "has_cvc"] + stage_cols + country_cols + year_cols

    available_controls = [c for c in control_cols if c in df_m.columns]
    df_reg = df_m.dropna(subset=available_controls).copy()

    print(f"\n{'DV':<25} {'GVC coef':>12} {'p-value':>10} {'N':>8} {'R2':>8}")
    print("-" * 65)

    for dv in dvs:
        if dv not in df_reg.columns:
            print(f"  {dv:<25} SKIP (column not found)")
            continue
        try:
            X = sm.add_constant(df_reg[available_controls].values)
            y = df_reg[dv].values
            model = sm.OLS(y, X).fit()
            # gvc_co_invest is index 1 (after constant)
            coef = model.params[1]
            pval = model.pvalues[1]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {dv:<25} {coef:>+.4f}{stars:3s} {pval:>10.3f} {int(model.nobs):>8} "
                  f"{model.rsquared:>8.3f}")
        except Exception as e:
            print(f"  {dv:<25} FAILED: {e}")


def sensitivity_analysis(df_full, dvs, caliper_values=None, n_matches_values=None):
    """Test sensitivity to matching parameters."""
    if caliper_values is None:
        caliper_values = [0.01, 0.025, 0.05, 0.1]
    if n_matches_values is None:
        n_matches_values = [1, 3, 5]

    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*80}")

    for dv in dvs[:1]:  # Only for primary DV
        print(f"\n  DV: {dv}")
        print(f"\n  {'Caliper':>10} {'k':>5} {'N_matched':>12} {'ATT':>10} {'p-value':>10}")
        print("  " + "-" * 50)

        for caliper in caliper_values:
            for n_m in n_matches_values:
                matched_df, pairs = nearest_neighbor_match(df_full, n_matches=n_m, caliper=caliper)
                if len(matched_df) < 20:
                    print(f"  {caliper:>10.3f} {n_m:>5} {len(matched_df):>12} {'skip':>10}")
                    continue

                treated = matched_df[matched_df["gvc_co_invest"] == 1][dv].dropna()
                control = matched_df[matched_df["gvc_co_invest"] == 0][dv].dropna()

                if len(treated) == 0 or len(control) == 0:
                    continue

                att = treated.mean() - control.mean()
                _, pval = stats.ttest_ind(treated, control, equal_var=False)
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"  {caliper:>10.3f} {n_m:>5} {len(matched_df):>12} "
                      f"{att:>+.4f}{stars:3s} {pval:>10.3f}")


@dataclass
class PSMResult:
    """Structured PSM results."""
    sample_size: int = 0
    matched_sample_size: int = 0
    match_rate: float = 0.0
    att_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "matched_sample_size": self.matched_sample_size,
            "match_rate": self.match_rate,
            "att_results": self.att_results,
        }


def run_psm(dataset_path: str, dvs: Optional[List[str]] = None,
            n_matches: int = 1, caliper: float = 0.05,
            run_sensitivity: bool = False) -> PSMResult:
    """Library interface for PSM analysis."""
    if dvs is None:
        dvs = ["follow_on", "exit", "round_progression"]

    result = PSMResult()

    df = pd.read_csv(dataset_path)
    df = prep_data(df)
    result.sample_size = len(df)

    # Step 1: Estimate propensity scores
    df_scored, ps_model, covariates = estimate_propensity_scores(df)

    # Step 2: Nearest-neighbor matching
    df_matched, pairs = nearest_neighbor_match(df_scored, n_matches=n_matches, caliper=caliper)
    result.matched_sample_size = len(df_matched)
    n_treated_original = (df_scored["gvc_co_invest"] == 1).sum()
    n_treated_matched = (df_matched["gvc_co_invest"] == 1).sum()
    result.match_rate = n_treated_matched / n_treated_original if n_treated_original > 0 else 0

    # Step 3: Balance check
    balance_vars = ["log_syndicate", "has_cvc"]
    # Add stage/country as numeric for balance check
    for col in ["stage", "country"]:
        if col in df_scored.columns:
            dummies = pd.get_dummies(df_scored[col], prefix=col, drop_first=True, dtype=int)
            for d in dummies.columns:
                df_scored[d] = dummies[d]
                if d in df_matched.columns or True:
                    matched_dummies = pd.get_dummies(df_matched[col], prefix=col, drop_first=True, dtype=int)
                    for md in matched_dummies.columns:
                        df_matched[md] = matched_dummies[md]
                    balance_vars.extend(matched_dummies.columns.tolist())
                    break

    check_balance(df_scored, df_matched, balance_vars)

    # Step 4: ATT estimation
    result.att_results = run_att_estimation(df_matched, dvs)

    # Step 5: OLS on matched sample
    run_regression_on_matched(df_matched, dvs)

    # Step 6: Sensitivity (optional)
    if run_sensitivity:
        # Suppress matching output during sensitivity
        import io
        from contextlib import redirect_stdout
        sensitivity_analysis(df_scored, dvs)

    print(f"\n{'='*80}")
    print("PSM ANALYSIS COMPLETE")
    print(f"{'='*80}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Propensity Score Matching for GVC analysis")
    parser.add_argument("--file", type=str, default=DEFAULT_DATASET, help="Path to deal dataset CSV")
    parser.add_argument("--dvs", type=str, default="follow_on,exit,round_progression",
                        help="Comma-separated DV names")
    parser.add_argument("--n-matches", type=int, default=1, help="Number of NN matches (default: 1)")
    parser.add_argument("--caliper", type=float, default=0.05, help="Caliper width (default: 0.05)")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    args = parser.parse_args()

    dvs = [x.strip() for x in args.dvs.split(",")]

    print(f"Loading: {args.file}")
    result = run_psm(args.file, dvs=dvs, n_matches=args.n_matches,
                     caliper=args.caliper, run_sensitivity=args.sensitivity)

    # Save results
    import json
    output_path = os.path.join(OUTPUT_DIR, "psm_results.json")
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
