"""
130_build_deal_dataset.py

Build unified deal-level dataset from CB Insights exports.

Steps:
  1. Load 6 CSV files (3 GVC + 3 PVC)
  2. Vertical concat with country / vc_type labels
  3. Identify GVC-PVC co-investment deals
  4. Generate treatment, outcome, and control variables
  5. Export analysis-ready dataset

Usage:
  python -m src.scripts.130_build_deal_dataset
"""

import csv
import os
import re
from collections import defaultdict
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cb_insights")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cb_insights")

# --- GVC name lists for co-investment detection ---
GVC_NAMES_JP = ["INCJ", "JIC Venture Growth", "JIC-VGI", "DBJ Capital"]
GVC_NAMES_KR = ["KVIC", "Korea Venture Investment Corp"]
GVC_NAMES_SG = [
    "Temasek", "EDBI", "Vertex Ventures SE Asia", "Vertex Ventures Southeast Asia",
    "Seeds Capital", "SGInnovate", "Pavilion Capital",
]
ALL_GVC_NAMES = GVC_NAMES_JP + GVC_NAMES_KR + GVC_NAMES_SG

# --- CVC detection (corporate venture capitals in PVC list) ---
CVC_NAMES = [
    "CyberAgent Capital", "Mitsui Sumitomo Insurance Venture Capital",
    "MS&AD Ventures", "Nissay Capital",
    "Samsung Ventures", "Kakao Ventures", "Smilegate Investment",
    "SoftBank Ventures Asia", "SBVA", "Lotte Ventures",
    "SingTel Innov8",
]


def load_csv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def normalize_stage(raw_stage):
    """Normalize Investment Stage to Seed / Series A / Series B."""
    s = raw_stage.strip()
    if "Series B" in s:
        return "Series B"
    elif "Series A" in s:
        return "Series A"
    elif "Seed" in s:
        return "Seed"
    return s


def parse_date(date_str):
    """Parse date string to datetime."""
    if not date_str or not date_str.strip():
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        return None


def parse_float(val):
    """Parse float, return None if invalid."""
    if not val or not val.strip():
        return None
    try:
        return float(val.strip())
    except ValueError:
        return None


def investor_contains_gvc(investor_str):
    """Check if investor string contains any GVC name."""
    if not investor_str:
        return False
    inv_lower = investor_str.lower()
    for name in ALL_GVC_NAMES:
        if name.lower() in inv_lower:
            return True
    return False


def investor_contains_cvc(investor_str):
    """Check if investor string contains any known CVC."""
    if not investor_str:
        return False
    inv_lower = investor_str.lower()
    for name in CVC_NAMES:
        if name.lower() in inv_lower:
            return True
    return False


def count_investors(investor_str):
    """Count number of investors in comma-separated string."""
    if not investor_str or not investor_str.strip():
        return 0
    return len([x.strip() for x in investor_str.split(",") if x.strip()])


def extract_year(date_str):
    """Extract year from date string."""
    d = parse_date(date_str)
    return d.year if d else None


def compute_firm_age(founded_year, deal_date):
    """Compute firm age at deal time."""
    if not founded_year or not deal_date:
        return None
    try:
        fy = int(founded_year)
        dy = extract_year(deal_date)
        if dy:
            return dy - fy
    except ValueError:
        pass
    return None


def build_exit_dummy(status):
    """Exit = 1 if IPO or Acquired."""
    s = (status or "").strip().lower()
    if "ipo" in s or "went public" in s or "acquired" in s:
        return 1
    return 0


def build_survival_dummy(status):
    """Survival = 1 if not dead/bankrupt."""
    s = (status or "").strip().lower()
    if "dead" in s or "inactive" in s or "bankrupt" in s:
        return 0
    return 1


def process_file(filepath, country, vc_type):
    """Load and tag rows from a single CSV."""
    rows = load_csv(filepath)
    processed = []
    for r in rows:
        row = {}

        # --- Identifiers ---
        row["company_name"] = r.get("Companies", "").strip()
        row["company_id"] = r.get("Company ID", "").strip()
        row["url"] = r.get("URL", "").strip()

        # --- Source labels ---
        row["country"] = country
        row["source_vc_type"] = vc_type  # which file it came from

        # --- Deal info ---
        row["investment_stage_raw"] = r.get("Investment Stage", "").strip()
        row["stage"] = normalize_stage(row["investment_stage_raw"])
        row["deal_size_m"] = parse_float(r.get("Deal Size (M)", ""))
        row["deal_date"] = r.get("Deal Date", "").strip()
        row["deal_year"] = extract_year(row["deal_date"])
        row["deal_status"] = r.get("Deal Status", "").strip()

        # --- Investors ---
        row["lead_investors"] = r.get("Lead Round Investors", "").strip()
        row["round_investors"] = r.get("Round Investors", "").strip()
        row["all_investors"] = r.get("All Investors", "").strip()
        row["syndicate_size"] = count_investors(row["round_investors"])

        # --- Company info ---
        row["founded_year"] = r.get("Founded Year", "").strip()
        row["firm_age"] = compute_firm_age(row["founded_year"], row["deal_date"])
        row["sector"] = r.get("Sector", "").strip()
        row["industry"] = r.get("Industry", "").strip()
        row["sub_industry"] = r.get("Sub-Industry", "").strip()
        row["business_model"] = r.get("Business Model", "").strip()
        row["technologies"] = r.get("Technologies", "").strip()
        row["commercial_maturity"] = r.get("Commercial Maturity", "").strip()
        row["description"] = r.get("Description", "").strip()

        # --- Funding ---
        row["total_funding_m"] = parse_float(r.get("Total Funding (M)", ""))
        row["latest_funding_date"] = r.get("Latest Funding Date", "").strip()
        row["latest_funding_amount_m"] = parse_float(r.get("Latest Funding Amount (M)", ""))
        row["latest_funding_round"] = r.get("Latest Funding Round", "").strip()
        row["latest_valuation_m"] = parse_float(r.get("Latest Valuation (M)", ""))

        # --- Outcome proxies ---
        row["company_status"] = r.get("Company Status", "").strip()
        row["exit"] = build_exit_dummy(row["company_status"])
        row["survival"] = build_survival_dummy(row["company_status"])

        processed.append(row)

    return processed


def detect_co_investment(rows):
    """
    Step 3: Detect GVC-PVC co-investment patterns.
    For each deal, check if GVC names appear in the investor lists.
    """
    for row in rows:
        investors_combined = ", ".join([
            row.get("round_investors", ""),
            row.get("all_investors", ""),
            row.get("lead_investors", ""),
        ])

        if row["source_vc_type"] == "GVC":
            row["gvc_backed"] = 1
            # Check if PVC also present (co-investment)
            # GVC deals always have GVC; co_investment = 1 if non-GVC investors present
            non_gvc_count = 0
            for inv in investors_combined.split(","):
                inv_clean = inv.strip()
                if inv_clean and not any(g.lower() in inv_clean.lower() for g in ALL_GVC_NAMES):
                    if inv_clean.lower() not in ["undisclosed investors", "undisclosed angel investors", ""]:
                        non_gvc_count += 1
            row["co_investment"] = 1 if non_gvc_count > 0 else 0
        else:
            # PVC deal — check if any GVC is also in the investor list
            has_gvc = investor_contains_gvc(investors_combined)
            row["gvc_backed"] = 1 if has_gvc else 0
            row["co_investment"] = 1 if has_gvc else 0

        # CVC detection
        row["has_cvc"] = 1 if investor_contains_cvc(investors_combined) else 0

    return rows


def build_follow_on_proxy(rows):
    """
    Step 5: Build follow-on funding proxy.
    For each deal, check if the company has a later funding round.
    Uses latest_funding_date vs deal_date comparison.
    Also: round_progression = 1 if company reached a later stage.
    """
    # Group deals by company_id
    company_deals = defaultdict(list)
    for row in rows:
        company_deals[row["company_id"]].append(row)

    stage_order = {"Seed": 0, "Series A": 1, "Series B": 2}

    for row in rows:
        cid = row["company_id"]
        deal_date = parse_date(row["deal_date"])
        deal_stage = row["stage"]

        # follow_on: does this company have ANY later deal in the dataset?
        has_follow_on = 0
        follow_on_amount = None
        for other in company_deals[cid]:
            other_date = parse_date(other["deal_date"])
            if deal_date and other_date and other_date > deal_date:
                has_follow_on = 1
                if other["deal_size_m"] is not None:
                    if follow_on_amount is None or other["deal_size_m"] > follow_on_amount:
                        follow_on_amount = other["deal_size_m"]

        # Also check latest_funding_date as a broader signal
        latest_date = parse_date(row["latest_funding_date"])
        if deal_date and latest_date and latest_date > deal_date:
            has_follow_on = 1
            if row["latest_funding_amount_m"] is not None:
                if follow_on_amount is None or row["latest_funding_amount_m"] > follow_on_amount:
                    follow_on_amount = row["latest_funding_amount_m"]

        row["follow_on"] = has_follow_on
        row["follow_on_amount_m"] = follow_on_amount

        # round_progression: did the company reach a LATER stage?
        current_order = stage_order.get(deal_stage, -1)
        progressed = 0
        for other in company_deals[cid]:
            other_order = stage_order.get(other["stage"], -1)
            other_date = parse_date(other["deal_date"])
            if other_order > current_order:
                progressed = 1
                break
        # Also check latest_funding_round for progression beyond B
        latest_round = row.get("latest_funding_round", "")
        if latest_round:
            if current_order == 0 and ("Series A" in latest_round or "Series B" in latest_round or "Series C" in latest_round):
                progressed = 1
            elif current_order == 1 and ("Series B" in latest_round or "Series C" in latest_round or "Series D" in latest_round):
                progressed = 1
            elif current_order == 2 and ("Series C" in latest_round or "Series D" in latest_round or "Series E" in latest_round):
                progressed = 1

        row["round_progression"] = progressed

    return rows


def assign_treatment(rows):
    """
    Final treatment assignment:
    - treatment = 1 if gvc_backed = 1 (GVC participated in this deal)
    - treatment = 0 if gvc_backed = 0 (pure PVC deal)
    """
    for row in rows:
        row["treatment"] = row["gvc_backed"]

        # Investment pattern classification
        if row["gvc_backed"] == 1 and row["co_investment"] == 1:
            row["investment_pattern"] = "GVC+PVC"
        elif row["gvc_backed"] == 1 and row["co_investment"] == 0:
            row["investment_pattern"] = "GVC_solo"
        else:
            row["investment_pattern"] = "PVC_only"

    return rows


def write_output(rows, filepath):
    """Write final dataset to CSV."""
    if not rows:
        return

    # Define output column order
    columns = [
        # Identifiers
        "company_name", "company_id", "url",
        # Source
        "country", "source_vc_type",
        # Treatment
        "treatment", "gvc_backed", "co_investment", "has_cvc", "investment_pattern",
        # Deal
        "deal_date", "deal_year", "stage", "investment_stage_raw",
        "deal_size_m", "deal_status",
        # Investors
        "lead_investors", "round_investors", "syndicate_size",
        # Company
        "founded_year", "firm_age", "sector", "industry", "sub_industry",
        "business_model", "technologies", "commercial_maturity",
        # Funding
        "total_funding_m", "latest_funding_date", "latest_funding_amount_m",
        "latest_funding_round", "latest_valuation_m",
        # Outcomes
        "company_status", "exit", "survival",
        "follow_on", "follow_on_amount_m", "round_progression",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Convert None to empty string
            out = {}
            for c in columns:
                v = row.get(c)
                out[c] = "" if v is None else v
            writer.writerow(out)


def print_summary(rows):
    """Print dataset summary statistics."""
    total = len(rows)
    treatment = sum(1 for r in rows if r["treatment"] == 1)
    control = total - treatment

    print(f"\n{'='*60}")
    print(f"DEAL-LEVEL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total deals:     {total}")
    print(f"Treatment (GVC):  {treatment}")
    print(f"Control (PVC):    {control}")
    print(f"Ratio:            1:{control/treatment:.1f}" if treatment > 0 else "")

    # By country
    print(f"\n--- By Country ---")
    for c in ["JP", "KR", "SG"]:
        t = sum(1 for r in rows if r["country"] == c and r["treatment"] == 1)
        ctrl = sum(1 for r in rows if r["country"] == c and r["treatment"] == 0)
        print(f"  {c}: Treatment={t}, Control={ctrl}, Total={t+ctrl}")

    # Investment patterns
    print(f"\n--- Investment Patterns ---")
    patterns = defaultdict(int)
    for r in rows:
        patterns[r["investment_pattern"]] += 1
    for p, c in sorted(patterns.items()):
        print(f"  {p}: {c}")

    # Stage distribution
    print(f"\n--- Stage Distribution ---")
    for s in ["Seed", "Series A", "Series B"]:
        t = sum(1 for r in rows if r["stage"] == s and r["treatment"] == 1)
        ctrl = sum(1 for r in rows if r["stage"] == s and r["treatment"] == 0)
        print(f"  {s}: Treatment={t}, Control={ctrl}")

    # Outcome variables
    print(f"\n--- Outcome Variables ---")
    exit_t = sum(1 for r in rows if r["treatment"] == 1 and r["exit"] == 1)
    exit_c = sum(1 for r in rows if r["treatment"] == 0 and r["exit"] == 1)
    fo_t = sum(1 for r in rows if r["treatment"] == 1 and r["follow_on"] == 1)
    fo_c = sum(1 for r in rows if r["treatment"] == 0 and r["follow_on"] == 1)
    rp_t = sum(1 for r in rows if r["treatment"] == 1 and r["round_progression"] == 1)
    rp_c = sum(1 for r in rows if r["treatment"] == 0 and r["round_progression"] == 1)
    print(f"  Exit:             Treatment={exit_t}/{treatment} ({exit_t/treatment*100:.1f}%), Control={exit_c}/{control} ({exit_c/control*100:.1f}%)")
    print(f"  Follow-on:        Treatment={fo_t}/{treatment} ({fo_t/treatment*100:.1f}%), Control={fo_c}/{control} ({fo_c/control*100:.1f}%)")
    print(f"  RoundProgression: Treatment={rp_t}/{treatment} ({rp_t/treatment*100:.1f}%), Control={rp_c}/{control} ({rp_c/control*100:.1f}%)")

    # CVC
    cvc_count = sum(1 for r in rows if r["has_cvc"] == 1)
    print(f"\n--- CVC Participation ---")
    print(f"  Deals with CVC: {cvc_count} ({cvc_count/total*100:.1f}%)")

    # Unique companies
    companies = set(r["company_id"] for r in rows)
    print(f"\n--- Companies ---")
    print(f"  Unique companies: {len(companies)}")

    # Data completeness
    print(f"\n--- Data Completeness ---")
    for field in ["deal_size_m", "total_funding_m", "firm_age", "follow_on_amount_m"]:
        non_null = sum(1 for r in rows if r.get(field) is not None and r.get(field) != "")
        print(f"  {field}: {non_null}/{total} ({non_null/total*100:.0f}%)")


def main():
    files = [
        ("jp_gvc.csv", "JP", "GVC"),
        ("kr_gvc.csv", "KR", "GVC"),
        ("sg_gvc.csv", "SG", "GVC"),
        ("jp_pvc.csv", "JP", "PVC"),
        ("kr_pvc.csv", "KR", "PVC"),
        ("sg_pvc.csv", "SG", "PVC"),
    ]

    # Step 2: Load and concat
    all_rows = []
    for filename, country, vc_type in files:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping")
            continue
        rows = process_file(filepath, country, vc_type)
        print(f"Loaded {filename}: {len(rows)} deals")
        all_rows.extend(rows)

    print(f"\nTotal after concat: {len(all_rows)} deals")

    # Step 3: Co-investment detection
    all_rows = detect_co_investment(all_rows)

    # Step 4 + 5: Treatment assignment + outcome variables
    all_rows = assign_treatment(all_rows)
    all_rows = build_follow_on_proxy(all_rows)

    # Print summary
    print_summary(all_rows)

    # Write output
    output_path = os.path.join(OUTPUT_DIR, "deal_dataset.csv")
    write_output(all_rows, output_path)
    print(f"\nDataset written to: {output_path}")
    print(f"Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
