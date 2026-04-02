"""
128_export_validator.py

Validate CB Insights export CSVs immediately after download.
Checks data integrity, detects co-investment, and flags issues
BEFORE dataset construction.

Usage:
  # Validate a single export
  python -m src.scripts.128_export_validator --file data/cb_insights/sg_pvc.csv \
    --expected-country Singapore --vc-type PVC

  # Validate with GVC co-investment detection
  python -m src.scripts.128_export_validator --file data/cb_insights/sg_pvc.csv \
    --expected-country Singapore --vc-type PVC \
    --gvc-names "Temasek,EDBI,Vertex Ventures,Seeds Capital,SGInnovate,Pavilion Capital"

  # Validate all files in directory
  python -m src.scripts.128_export_validator --dir data/cb_insights/ \
    --gvc-names "INCJ,JIC Venture Growth,DBJ Capital,KVIC,Temasek,EDBI,Vertex Ventures,Seeds Capital,SGInnovate,Pavilion Capital"
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


def load_csv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


class ValidationResult:
    def __init__(self, filename):
        self.filename = filename
        self.errors = []
        self.warnings = []
        self.info = []
        self.stats = {}

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def add_info(self, msg):
        self.info.append(msg)

    @property
    def passed(self):
        return len(self.errors) == 0

    def print_report(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        print(f"\n{'='*70}")
        print(f"VALIDATION: {self.filename} — {status}")
        print(f"{'='*70}")

        if self.errors:
            print(f"\n  ERRORS ({len(self.errors)}):")
            for e in self.errors:
                print(f"    ❌ {e}")

        if self.warnings:
            print(f"\n  WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"    ⚠️ {w}")

        if self.info:
            print(f"\n  INFO:")
            for i in self.info:
                print(f"    ℹ️ {i}")


def validate_country(rows, expected_country, result):
    """Check that all rows match expected country."""
    countries = defaultdict(int)
    for r in rows:
        c = r.get("Country", "").strip()
        countries[c] += 1

    result.stats["countries"] = dict(countries)

    if expected_country:
        # Fuzzy match (e.g., "Korea" matches "South Korea")
        matching = sum(
            n for c, n in countries.items()
            if expected_country.lower() in c.lower() or c.lower() in expected_country.lower()
        )
        non_matching = len(rows) - matching

        if non_matching > 0:
            result.error(
                f"Country filter issue: {non_matching}/{len(rows)} rows don't match '{expected_country}'. "
                f"Found: {dict(countries)}"
            )
        else:
            result.add_info(f"Country: all {len(rows)} rows = {expected_country}")
    else:
        result.add_info(f"Country distribution: {dict(countries)}")


def validate_stage(rows, result):
    """Check investment stage distribution."""
    stages = defaultdict(int)
    for r in rows:
        s = r.get("Investment Stage", "").strip()
        stages[s] += 1

    result.stats["stages"] = dict(stages)

    # Check for unexpected stages
    valid_prefixes = ["Seed", "Series A", "Series B"]
    unexpected = {s: n for s, n in stages.items()
                  if not any(s.startswith(p) for p in valid_prefixes)}

    if unexpected:
        result.warn(
            f"Unexpected stages found: {unexpected}. "
            f"Expected only Seed/Series A/Series B variants."
        )

    # Check for Pre-Seed
    preseed = sum(n for s, n in stages.items() if "pre-seed" in s.lower() or "pre seed" in s.lower())
    if preseed > 0:
        result.error(f"Pre-Seed deals found: {preseed}. These should be excluded.")

    # Stage summary
    seed = sum(n for s, n in stages.items() if "seed" in s.lower() and "pre" not in s.lower())
    series_a = sum(n for s, n in stages.items() if "series a" in s.lower())
    series_b = sum(n for s, n in stages.items() if "series b" in s.lower())
    result.add_info(f"Stage: Seed={seed}, Series A={series_a}, Series B={series_b}")


def validate_dates(rows, result):
    """Check deal date range."""
    dates = [r.get("Deal Date", "").strip() for r in rows if r.get("Deal Date", "").strip()]
    if dates:
        dates.sort()
        result.stats["date_range"] = (dates[0], dates[-1])
        result.add_info(f"Date range: {dates[0]} to {dates[-1]}")

        # Check for very old deals
        old = sum(1 for d in dates if d < "2005-01-01")
        if old > 0:
            result.warn(f"{old} deals before 2005. These will be filtered in analysis.")
    else:
        result.error("No deal dates found.")


def validate_coverage(rows, result):
    """Check field coverage for key variables."""
    key_fields = {
        "Deal Size (M)": 0.60,     # Threshold for warning
        "Total Funding (M)": 0.70,
        "Founded Year": 0.80,
        "Company Status": 0.95,
        "Round Investors": 0.90,
    }

    coverage = {}
    for field, threshold in key_fields.items():
        # Fuzzy match
        matched = None
        for r in rows[0].keys():
            if field.lower() in r.lower():
                matched = r
                break
        if not matched:
            continue

        non_null = sum(1 for r in rows if r.get(matched, "").strip())
        rate = non_null / len(rows) if rows else 0
        coverage[field] = rate

        if rate < threshold:
            result.warn(f"Low coverage: {field} = {rate:.0%} (threshold: {threshold:.0%})")

    result.stats["coverage"] = coverage


def detect_co_investment(rows, gvc_names, result):
    """
    Core function: detect GVC names in PVC deal investor lists.
    This is the key insight generator.
    """
    if not gvc_names:
        return

    investor_fields = []
    for key in rows[0].keys():
        if "investor" in key.lower():
            investor_fields.append(key)

    if not investor_fields:
        result.warn("No investor fields found for co-investment check.")
        return

    gvc_deals = []
    for i, r in enumerate(rows):
        investors_combined = ", ".join(r.get(f, "") for f in investor_fields)
        inv_lower = investors_combined.lower()

        matched_gvcs = [g for g in gvc_names if g.lower() in inv_lower]
        if matched_gvcs:
            gvc_deals.append({
                "row": i,
                "company": r.get("Companies", ""),
                "stage": r.get("Investment Stage", ""),
                "date": r.get("Deal Date", ""),
                "gvcs": matched_gvcs,
            })

    co_invest_rate = len(gvc_deals) / len(rows) if rows else 0
    result.stats["co_investment"] = {
        "gvc_deals": len(gvc_deals),
        "total_deals": len(rows),
        "rate": co_invest_rate,
    }

    result.add_info(f"Co-investment detection: {len(gvc_deals)}/{len(rows)} deals ({co_invest_rate:.1%}) contain GVC investors")

    if len(gvc_deals) > 0:
        # GVC breakdown
        gvc_counts = defaultdict(int)
        for d in gvc_deals:
            for g in d["gvcs"]:
                gvc_counts[g] += 1

        result.add_info(f"GVC breakdown in PVC deals:")
        for g, n in sorted(gvc_counts.items(), key=lambda x: -x[1]):
            result.add_info(f"  {g}: {n} deals")

        # Treatment reclassification notice
        if co_invest_rate > 0.01:
            result.warn(
                f"{len(gvc_deals)} PVC deals contain GVC investors → "
                f"these should be reclassified as Treatment (GVC+PVC co-investment) "
                f"in the analysis dataset."
            )

        # Show sample
        result.add_info(f"Sample co-investment deals (first 5):")
        for d in gvc_deals[:5]:
            result.add_info(f"  {d['company'][:30]:30s} | {d['stage']:15s} | {d['date']} | GVC: {', '.join(d['gvcs'])}")

    return gvc_deals


def detect_gvc_solo(rows, gvc_names, result):
    """For GVC files: detect solo vs co-investment."""
    if not gvc_names:
        return

    investor_fields = []
    for key in rows[0].keys():
        if "round investor" in key.lower():
            investor_fields.append(key)

    if not investor_fields:
        return

    solo = 0
    coinvest = 0

    for r in rows:
        investors_combined = ", ".join(r.get(f, "") for f in investor_fields)
        investors = [x.strip() for x in investors_combined.split(",") if x.strip()]

        # Remove GVC names and undisclosed
        non_gvc = []
        for inv in investors:
            if any(g.lower() in inv.lower() for g in gvc_names):
                continue
            if inv.lower() in ["undisclosed investors", "undisclosed angel investors"]:
                continue
            non_gvc.append(inv)

        if non_gvc:
            coinvest += 1
        else:
            solo += 1

    total = solo + coinvest
    if total > 0:
        coinvest_rate = coinvest / total
        result.stats["gvc_coinvest_rate"] = coinvest_rate
        result.add_info(f"GVC investment pattern: solo={solo}, co-invest={coinvest} ({coinvest_rate:.0%})")

        if coinvest_rate > 0.8:
            result.warn(
                f"Co-investment rate = {coinvest_rate:.0%} (>{80}%). "
                f"Treatment should be 'GVC syndicate participation', not 'GVC vs PVC'."
            )


def validate_file(filepath, expected_country=None, vc_type=None, gvc_names=None):
    """Run all validations on a single file."""
    filename = os.path.basename(filepath)
    result = ValidationResult(filename)

    rows, fieldnames = load_csv(filepath)
    result.add_info(f"Rows: {len(rows)}, Columns: {len(fieldnames)}")

    if len(rows) == 0:
        result.error("File is empty.")
        result.print_report()
        return result

    validate_country(rows, expected_country, result)
    validate_stage(rows, result)
    validate_dates(rows, result)
    validate_coverage(rows, result)

    if vc_type == "PVC" and gvc_names:
        detect_co_investment(rows, gvc_names, result)
    elif vc_type == "GVC" and gvc_names:
        detect_gvc_solo(rows, gvc_names, result)

    result.print_report()
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate CB Insights CSV exports")
    parser.add_argument("--file", type=str, help="Path to CSV file")
    parser.add_argument("--dir", type=str, help="Path to directory with CSV files")
    parser.add_argument("--expected-country", type=str, help="Expected country for all rows")
    parser.add_argument("--vc-type", type=str, choices=["GVC", "PVC"], help="GVC or PVC file")
    parser.add_argument("--gvc-names", type=str, help="Comma-separated GVC names for co-investment detection")
    args = parser.parse_args()

    gvc_names = [x.strip() for x in args.gvc_names.split(",")] if args.gvc_names else None

    if args.file:
        validate_file(args.file, args.expected_country, args.vc_type, gvc_names)
    elif args.dir:
        csv_files = sorted(Path(args.dir).glob("*.csv"))
        results = []
        for f in csv_files:
            if f.name == "deal_dataset.csv":
                continue
            # Infer vc_type and country from filename
            fname = f.stem.lower()
            vc_type = "GVC" if "gvc" in fname else "PVC"
            country = None
            if "jp" in fname:
                country = "Japan"
            elif "kr" in fname:
                country = "South Korea"
            elif "sg" in fname:
                country = "Singapore"

            r = validate_file(str(f), country, vc_type, gvc_names)
            results.append(r)

        # Summary
        print(f"\n{'='*70}")
        print(f"VALIDATION SUMMARY ({len(results)} files)")
        print(f"{'='*70}")
        for r in results:
            status = "✅" if r.passed else "❌"
            errs = f", {len(r.errors)} errors" if r.errors else ""
            warns = f", {len(r.warnings)} warnings" if r.warnings else ""
            print(f"  {status} {r.filename}{errs}{warns}")
    else:
        print("Please specify --file or --dir")
        sys.exit(1)


@dataclass
class DirectoryValidationResult:
    """Aggregated validation result for 108 orchestrator."""
    files_validated: int = 0
    total_rows: int = 0
    has_errors: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.has_errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_validated": self.files_validated,
            "total_rows": self.total_rows,
            "has_errors": self.has_errors,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
            "passed": self.passed,
        }


def run_validation(data_dir: str, gvc_names: Optional[List[str]] = None) -> DirectoryValidationResult:
    """Library interface: validate all CSVs in a directory.

    Called by 108 orchestrator for Phase 0 (pre-check) and Phase 2b (full check).
    """
    agg = DirectoryValidationResult()
    data_path = Path(data_dir)
    csv_files = sorted(f for f in data_path.glob("*.csv") if f.name != "deal_dataset.csv")

    if not csv_files:
        agg.has_errors = True
        agg.errors.append(f"No CSV files found in {data_dir}")
        return agg

    co_invest_total = {"gvc_deals": 0, "total_deals": 0}

    for f in csv_files:
        fname = f.stem.lower()
        vc_type = "GVC" if "gvc" in fname else "PVC"
        country = None
        if "jp" in fname:
            country = "Japan"
        elif "kr" in fname:
            country = "South Korea"
        elif "sg" in fname:
            country = "Singapore"

        r = validate_file(str(f), country, vc_type, gvc_names)
        agg.files_validated += 1

        rows, _ = load_csv(str(f))
        agg.total_rows += len(rows)

        if not r.passed:
            agg.has_errors = True
            for e in r.errors:
                agg.errors.append(f"{f.name}: {e}")

        for w in r.warnings:
            agg.warnings.append(f"{f.name}: {w}")

        # Aggregate co-investment stats
        if "co_investment" in r.stats:
            co_invest_total["gvc_deals"] += r.stats["co_investment"]["gvc_deals"]
            co_invest_total["total_deals"] += r.stats["co_investment"]["total_deals"]

    if co_invest_total["total_deals"] > 0:
        co_invest_total["rate"] = co_invest_total["gvc_deals"] / co_invest_total["total_deals"]
    agg.stats["co_investment"] = co_invest_total

    return agg


if __name__ == "__main__":
    main()
