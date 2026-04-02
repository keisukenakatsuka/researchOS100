"""
125_data_source_audit.py

Phase 0: Data Source Audit.

Given a sample CSV export from CB Insights (or other data source),
audit which fields are available, measure coverage, and assess
DV feasibility BEFORE committing to variable design.

This script must run BEFORE hypothesis design and full data collection.

Usage:
  # Audit a sample export
  python -m src.scripts.125_data_source_audit --file data/cb_insights/sample.csv

  # Audit with specific DV candidates
  python -m src.scripts.125_data_source_audit --file data/cb_insights/sample.csv \
    --dv-candidates "exit,follow_on,employee_growth,revenue,patent"

  # Audit all 6 files at once
  python -m src.scripts.125_data_source_audit --dir data/cb_insights/
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- DV Feasibility Rules ---
# Maps DV candidate names to the fields required and how to assess feasibility.
DV_REGISTRY = {
    "exit": {
        "description": "IPO or Acquired = 1",
        "required_fields": ["Company Status"],
        "valid_values_field": "Company Status",
        "valid_check": lambda vals: any(
            "ipo" in v.lower() or "acquired" in v.lower() or "went public" in v.lower()
            for v in vals if v
        ),
        "proxy": None,
    },
    "follow_on": {
        "description": "Subsequent funding round exists = 1",
        "required_fields": ["Latest Funding Date", "Deal Date"],
        "valid_values_field": None,
        "valid_check": lambda vals: True,  # Constructible from date comparison
        "proxy": "Latest Funding Round (presence of later round)",
    },
    "round_progression": {
        "description": "Company reached a later investment stage = 1",
        "required_fields": ["Investment Stage", "Latest Funding Round"],
        "valid_values_field": None,
        "valid_check": lambda vals: True,
        "proxy": None,
    },
    "employee_growth": {
        "description": "Employee count CAGR over 3 years",
        "required_fields": ["Employee Count", "Employee Count History"],
        "valid_values_field": None,
        "valid_check": lambda vals: False,  # Typically not in CB Insights
        "proxy": "Follow-on funding as growth proxy",
    },
    "revenue": {
        "description": "Revenue or revenue growth",
        "required_fields": ["Revenue", "Revenue Growth"],
        "valid_values_field": None,
        "valid_check": lambda vals: False,
        "proxy": "Total Funding as scale proxy",
    },
    "patent": {
        "description": "Patent filings (5yr cumulative)",
        "required_fields": ["Patent Count"],
        "valid_values_field": None,
        "valid_check": lambda vals: False,  # Requires external data
        "proxy": "Requires Lens.org or patent office DB linkage",
    },
    "survival": {
        "description": "Company still active = 1",
        "required_fields": ["Company Status"],
        "valid_values_field": "Company Status",
        "valid_check": lambda vals: any(
            "alive" in v.lower() or "active" in v.lower() for v in vals if v
        ),
        "proxy": None,
    },
    "valuation": {
        "description": "Latest valuation (USD)",
        "required_fields": ["Latest Valuation (M)"],
        "valid_values_field": "Latest Valuation (M)",
        "valid_check": lambda vals: sum(1 for v in vals if v and v.strip()) > 0,
        "proxy": None,
    },
}

# --- Control Variable Feasibility ---
CONTROL_REGISTRY = {
    "deal_size": {"field": "Deal Size (M)", "type": "continuous"},
    "firm_age": {"field": "Founded Year", "type": "derived (year - founded)"},
    "total_funding": {"field": "Total Funding (M)", "type": "continuous"},
    "stage": {"field": "Investment Stage", "type": "categorical"},
    "country": {"field": "Country", "type": "categorical"},
    "industry": {"field": "Industry", "type": "categorical"},
    "sub_industry": {"field": "Sub-Industry", "type": "categorical"},
    "syndicate_size": {"field": "Round Investors", "type": "derived (count)"},
    "deal_date": {"field": "Deal Date", "type": "date"},
}


def load_csv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def compute_coverage(rows, field):
    """Compute non-null, non-empty coverage for a field."""
    total = len(rows)
    if total == 0:
        return 0, 0, 0.0
    non_null = sum(1 for r in rows if r.get(field, "").strip())
    return non_null, total, non_null / total


def audit_fields(rows, fieldnames):
    """Audit all fields for coverage."""
    print(f"\n{'='*70}")
    print(f"FIELD COVERAGE AUDIT ({len(rows)} rows)")
    print(f"{'='*70}")
    print(f"{'Field':<45} {'Coverage':>10} {'Rate':>8}")
    print(f"{'-'*45} {'-'*10} {'-'*8}")

    for field in fieldnames:
        non_null, total, rate = compute_coverage(rows, field)
        indicator = "✅" if rate >= 0.7 else "⚠️" if rate >= 0.3 else "❌"
        print(f"{indicator} {field:<43} {non_null:>5}/{total:<5} {rate:>6.0%}")


def audit_dv_candidates(rows, fieldnames, candidates):
    """Assess DV feasibility."""
    print(f"\n{'='*70}")
    print(f"DV FEASIBILITY ASSESSMENT")
    print(f"{'='*70}")

    for dv_name in candidates:
        if dv_name not in DV_REGISTRY:
            print(f"\n  ❓ {dv_name}: Unknown DV candidate (not in registry)")
            continue

        dv = DV_REGISTRY[dv_name]
        print(f"\n  📊 {dv_name}: {dv['description']}")

        # Check required fields
        fields_present = []
        fields_missing = []
        for req_field in dv["required_fields"]:
            # Fuzzy match: check if any fieldname contains the required field
            matched = None
            for fn in fieldnames:
                if req_field.lower() in fn.lower():
                    matched = fn
                    break
            if matched:
                non_null, total, rate = compute_coverage(rows, matched)
                fields_present.append((req_field, matched, rate))
            else:
                fields_missing.append(req_field)

        for req, actual, rate in fields_present:
            indicator = "✅" if rate >= 0.7 else "⚠️" if rate >= 0.3 else "❌"
            print(f"     {indicator} {req} → '{actual}' (coverage: {rate:.0%})")

        for req in fields_missing:
            print(f"     ❌ {req} → NOT FOUND in data")

        # Verdict
        if fields_missing:
            verdict = "❌ NOT FEASIBLE"
            if dv.get("proxy"):
                verdict += f" → Proxy: {dv['proxy']}"
        elif all(rate >= 0.5 for _, _, rate in fields_present):
            verdict = "✅ FEASIBLE"
        else:
            low_coverage = [(req, rate) for req, _, rate in fields_present if rate < 0.5]
            verdict = f"⚠️ PARTIAL (low coverage: {', '.join(f'{r}={rt:.0%}' for r, rt in low_coverage)})"
            if dv.get("proxy"):
                verdict += f" → Proxy: {dv['proxy']}"

        print(f"     → {verdict}")


def audit_controls(rows, fieldnames):
    """Assess control variable feasibility."""
    print(f"\n{'='*70}")
    print(f"CONTROL VARIABLE FEASIBILITY")
    print(f"{'='*70}")

    for ctrl_name, ctrl in CONTROL_REGISTRY.items():
        matched = None
        for fn in fieldnames:
            if ctrl["field"].lower() in fn.lower():
                matched = fn
                break

        if matched:
            non_null, total, rate = compute_coverage(rows, matched)
            indicator = "✅" if rate >= 0.7 else "⚠️" if rate >= 0.3 else "❌"
            print(f"  {indicator} {ctrl_name:<20} → '{matched}' ({rate:.0%}) [{ctrl['type']}]")
        else:
            print(f"  ❌ {ctrl_name:<20} → '{ctrl['field']}' NOT FOUND")


def audit_treatment_structure(rows, fieldnames, gvc_names=None):
    """Check co-investment structure in investor fields."""
    print(f"\n{'='*70}")
    print(f"TREATMENT STRUCTURE CHECK")
    print(f"{'='*70}")

    # Find investor fields
    investor_fields = [f for f in fieldnames if "investor" in f.lower()]
    if not investor_fields:
        print("  ⚠️ No investor fields found. Cannot check treatment structure.")
        return

    print(f"  Investor fields found: {investor_fields}")

    if gvc_names:
        # Check co-investment rate
        gvc_deals = 0
        coinvest_deals = 0
        for r in rows:
            investors_combined = ", ".join(r.get(f, "") for f in investor_fields)
            inv_lower = investors_combined.lower()
            has_gvc = any(g.lower() in inv_lower for g in gvc_names)
            if has_gvc:
                gvc_deals += 1
                # Count non-GVC investors
                non_gvc = 0
                for inv in investors_combined.split(","):
                    inv_clean = inv.strip()
                    if inv_clean and not any(g.lower() in inv_clean.lower() for g in gvc_names):
                        if inv_clean.lower() not in ["undisclosed investors", "undisclosed angel investors"]:
                            non_gvc += 1
                if non_gvc > 0:
                    coinvest_deals += 1

        print(f"\n  GVC names checked: {gvc_names}")
        print(f"  Deals with GVC: {gvc_deals}/{len(rows)} ({gvc_deals/len(rows)*100:.1f}%)")
        if gvc_deals > 0:
            coinvest_rate = coinvest_deals / gvc_deals
            print(f"  Co-investment rate: {coinvest_deals}/{gvc_deals} ({coinvest_rate:.0%})")
            if coinvest_rate > 0.8:
                print(f"\n  ⚠️ Co-investment rate > 80%.")
                print(f"     → Treatment should be defined as 'GVC syndicate participation'")
                print(f"     → NOT 'GVC vs PVC' binary comparison")
            elif coinvest_rate > 0.5:
                print(f"\n  ℹ️ Co-investment rate > 50%. Consider treatment definition carefully.")
    else:
        # Just report syndicate size distribution
        syndicate_sizes = []
        for r in rows:
            for f in investor_fields:
                investors = r.get(f, "")
                if investors:
                    count = len([x.strip() for x in investors.split(",") if x.strip()])
                    syndicate_sizes.append(count)
                    break
        if syndicate_sizes:
            avg = sum(syndicate_sizes) / len(syndicate_sizes)
            solo = sum(1 for s in syndicate_sizes if s <= 1)
            print(f"\n  Average syndicate size: {avg:.1f}")
            print(f"  Solo investor deals: {solo}/{len(syndicate_sizes)} ({solo/len(syndicate_sizes)*100:.1f}%)")


@dataclass
class AuditResult:
    """Structured result from data source audit (for library use)."""
    files_audited: int = 0
    total_rows: int = 0
    dv_status: Dict[str, str] = field(default_factory=dict)   # dv_name -> FEASIBLE/PARTIAL/NOT_FEASIBLE
    feasible_dvs: List[str] = field(default_factory=list)
    coverage: Dict[str, float] = field(default_factory=dict)   # field -> rate
    co_investment_rate: Optional[float] = None
    treatment_definition: str = "GVC vs PVC binary"
    issues: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.feasible_dvs) >= 1 and not any("STOP" in i for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_audited": self.files_audited,
            "total_rows": self.total_rows,
            "dv_status": self.dv_status,
            "feasible_dvs": self.feasible_dvs,
            "coverage": self.coverage,
            "co_investment_rate": self.co_investment_rate,
            "treatment_definition": self.treatment_definition,
            "issues": self.issues,
            "passed": self.passed,
        }


def _assess_dv_status(rows, fieldnames, dv_name):
    """Assess a single DV's feasibility, return status string."""
    if dv_name not in DV_REGISTRY:
        return "UNKNOWN"
    dv = DV_REGISTRY[dv_name]
    fields_present = []
    fields_missing = []
    for req_field in dv["required_fields"]:
        matched = None
        for fn in fieldnames:
            if req_field.lower() in fn.lower():
                matched = fn
                break
        if matched:
            _, _, rate = compute_coverage(rows, matched)
            fields_present.append((req_field, matched, rate))
        else:
            fields_missing.append(req_field)

    if fields_missing:
        return "NOT_FEASIBLE"
    elif all(rate >= 0.5 for _, _, rate in fields_present):
        return "FEASIBLE"
    else:
        return "PARTIAL"


def audit_file(filepath, dv_candidates, gvc_names=None):
    """Run full audit on a single file."""
    print(f"\n{'#'*70}")
    print(f"# DATA SOURCE AUDIT: {os.path.basename(filepath)}")
    print(f"{'#'*70}")

    rows, fieldnames = load_csv(filepath)
    print(f"\nFile: {filepath}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(fieldnames)}")

    audit_fields(rows, fieldnames)
    audit_dv_candidates(rows, fieldnames, dv_candidates)
    audit_controls(rows, fieldnames)
    audit_treatment_structure(rows, fieldnames, gvc_names)

    # Country distribution
    countries = defaultdict(int)
    for r in rows:
        c = r.get("Country", "").strip()
        if c:
            countries[c] += 1
    if countries:
        print(f"\n  Country distribution:")
        for c, n in sorted(countries.items(), key=lambda x: -x[1])[:10]:
            print(f"    {c}: {n}")

    return rows, fieldnames


def run_audit(data_dir: str, dv_candidates: Optional[List[str]] = None,
              gvc_names: Optional[List[str]] = None) -> AuditResult:
    """Library interface: run audit on a directory and return structured result.

    Called by 108 orchestrator for Phase 0.
    """
    if dv_candidates is None:
        dv_candidates = list(DV_REGISTRY.keys())

    result = AuditResult()
    data_path = Path(data_dir)
    csv_files = sorted(f for f in data_path.glob("*.csv") if f.name != "deal_dataset.csv")

    if not csv_files:
        result.issues.append(f"STOP: No CSV files found in {data_dir}")
        return result

    all_rows = []
    all_fieldnames = set()

    for f in csv_files:
        rows, fieldnames = audit_file(str(f), dv_candidates, gvc_names)
        all_rows.extend(rows)
        all_fieldnames.update(fieldnames)
        result.files_audited += 1

    result.total_rows = len(all_rows)
    fieldnames_list = list(all_fieldnames)

    # DV feasibility
    for dv_name in dv_candidates:
        status = _assess_dv_status(all_rows, fieldnames_list, dv_name)
        result.dv_status[dv_name] = status
        if status == "FEASIBLE":
            result.feasible_dvs.append(dv_name)

    if not result.feasible_dvs:
        result.issues.append("STOP: No feasible dependent variables found in data source.")

    # Key field coverage
    for field_name in ["Company Status", "Deal Date", "Deal Size (M)", "Total Funding (M)"]:
        for fn in fieldnames_list:
            if field_name.lower() in fn.lower():
                _, _, rate = compute_coverage(all_rows, fn)
                result.coverage[field_name] = rate
                if field_name in ("Company Status", "Deal Date") and rate < 0.9:
                    result.issues.append(f"STOP: {field_name} coverage too low ({rate:.0%}).")
                break

    # Co-investment rate (from GVC files)
    if gvc_names:
        gvc_rows = [r for r in all_rows
                     if any(g.lower() in ", ".join(
                         r.get(f, "") for f in r.keys() if "investor" in f.lower()
                     ).lower() for g in gvc_names)]
        if gvc_rows:
            investor_fields = [f for f in all_rows[0].keys() if "investor" in f.lower()]
            coinvest = 0
            for r in gvc_rows:
                inv_combined = ", ".join(r.get(f, "") for f in investor_fields)
                non_gvc = [x.strip() for x in inv_combined.split(",")
                           if x.strip()
                           and not any(g.lower() in x.strip().lower() for g in gvc_names)
                           and x.strip().lower() not in ("undisclosed investors", "undisclosed angel investors")]
                if non_gvc:
                    coinvest += 1
            rate = coinvest / len(gvc_rows) if gvc_rows else 0
            result.co_investment_rate = rate
            if rate > 0.8:
                result.treatment_definition = "GVC syndicate participation"

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Data Source Audit")
    parser.add_argument("--file", type=str, help="Path to sample CSV file")
    parser.add_argument("--dir", type=str, help="Path to directory with CSV files")
    parser.add_argument(
        "--dv-candidates",
        type=str,
        default="exit,follow_on,round_progression,survival,valuation,employee_growth,revenue,patent",
        help="Comma-separated DV candidate names",
    )
    parser.add_argument(
        "--gvc-names",
        type=str,
        default=None,
        help="Comma-separated GVC names for treatment structure check",
    )
    args = parser.parse_args()

    dv_candidates = [x.strip() for x in args.dv_candidates.split(",")]
    gvc_names = [x.strip() for x in args.gvc_names.split(",")] if args.gvc_names else None

    if args.file:
        audit_file(args.file, dv_candidates, gvc_names)
    elif args.dir:
        csv_files = sorted(Path(args.dir).glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {args.dir}")
            sys.exit(1)
        for f in csv_files:
            if f.name == "deal_dataset.csv":
                continue  # Skip generated dataset
            audit_file(str(f), dv_candidates, gvc_names)
    else:
        print("Please specify --file or --dir")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("AUDIT COMPLETE")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("  1. Review DV feasibility → confirm which DVs to use")
    print("  2. Review control coverage → decide on imputation strategy")
    print("  3. Review treatment structure → confirm treatment definition")
    print("  4. Proceed to hypothesis design (Phase 1)")


if __name__ == "__main__":
    main()
