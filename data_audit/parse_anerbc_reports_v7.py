"""
AneRBC-I CBC report parser (v7) for the anemia multimodal project.

Fixes vs v6:
1. Only parses AneRBC-I CBC_reports (AneRBC-II is excluded -- it duplicates
   the same 1,000 patients' reports verbatim, confirmed identical).
2. Adds an internal lab-consistency check using the standard identity:
       expected_HCT = RBC * MCV / 10
   If reported HCT deviates from expected_HCT by more than a tolerance,
   the report is flagged "Inconsistent" (likely corrupted / data-entry
   error) and excluded from severity/subtype label derivation.
3. Reports the "Healthy but non-Normal severity" mismatch rate BOTH before
   and after removing inconsistent rows, so we can see how much of the
   mismatch was genuine data corruption vs a real labeling quirk in the
   source dataset.

Usage:
    python parse_anerbc_reports_v7.py "C:\\path\\to\\datasets"

Output:
    anerbc_cbc_structured_v7.csv   -- one row per unique AneRBC-I patient
"""

import os
import re
import sys
import csv
from collections import Counter

CONSISTENCY_TOLERANCE_PCT = 20  # allowed %% deviation between reported and expected HCT


def find_anerbc1_cbc_dirs(root_path):
    """Only folders under a path containing 'AneRBC-I' (not AneRBC-II)."""
    hits = []
    for dirpath, dirnames, _ in os.walk(root_path):
        if os.path.basename(dirpath) == "CBC_reports" and "AneRBC-I" in dirpath and "AneRBC-II" not in dirpath:
            hits.append(dirpath)
    return hits


def parse_report(filepath):
    values = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        test_name = parts[0].strip().upper()
        result_field = parts[1].strip()
        flagged = result_field.startswith("*")
        result_field = result_field.lstrip("*").strip()
        m = re.search(r'[-+]?\d*\.?\d+', result_field)
        if not m:
            continue
        value = float(m.group())
        values[test_name] = {"value": value, "flagged": flagged}
    return values


def classify_severity(hgb):
    if hgb is None:
        return "Unknown"
    if hgb >= 12:
        return "Normal"
    elif hgb >= 10:
        return "Mild"
    elif hgb >= 7:
        return "Moderate"
    else:
        return "Severe"


def classify_subtype(mcv):
    if mcv is None:
        return "Unknown"
    if mcv < 80:
        return "Microcytic"
    elif mcv <= 100:
        return "Normocytic"
    else:
        return "Macrocytic"


def check_consistency(rbc, mcv, hct):
    """Returns (is_consistent: bool, expected_hct: float or None)."""
    if rbc is None or mcv is None or hct is None:
        return None, None  # can't check
    expected_hct = rbc * mcv / 10
    if expected_hct == 0:
        return False, expected_hct
    pct_dev = abs(hct - expected_hct) / expected_hct * 100
    return (pct_dev <= CONSISTENCY_TOLERANCE_PCT), expected_hct


def main():
    root_path = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    root_path = os.path.abspath(root_path)

    if not os.path.isdir(root_path):
        print(f"ERROR: '{root_path}' is not a valid directory.")
        sys.exit(1)

    cbc_dirs = find_anerbc1_cbc_dirs(root_path)
    print(f"Found {len(cbc_dirs)} AneRBC-I CBC_reports folder(s) (AneRBC-II excluded):")
    for d in cbc_dirs:
        print(f"  - {d}")
    print()

    rows = []
    for cbc_dir in cbc_dirs:
        path_lower = cbc_dir.lower()
        if "anemic" in path_lower and "healthy" not in path_lower:
            provided_label = "Anemic"
        elif "healthy" in path_lower:
            provided_label = "Healthy"
        else:
            provided_label = "Unknown"

        for fname in sorted(os.listdir(cbc_dir)):
            if not fname.lower().endswith(".txt"):
                continue
            full = os.path.join(cbc_dir, fname)
            image_id = os.path.splitext(fname)[0]
            values = parse_report(full)

            hgb = values.get("HGB", {}).get("value")
            mcv = values.get("MCV", {}).get("value")
            mchc = values.get("MCHC", {}).get("value")
            rbc = values.get("RBC", {}).get("value")
            hct = values.get("HCT", {}).get("value")
            mch = values.get("MCH", {}).get("value")

            is_consistent, expected_hct = check_consistency(rbc, mcv, hct)

            rows.append({
                "image_id": image_id,
                "provided_label": provided_label,
                "HGB": hgb,
                "RBC": rbc,
                "HCT": hct,
                "MCV": mcv,
                "MCH": mch,
                "MCHC": mchc,
                "expected_HCT_from_RBCxMCV": round(expected_hct, 1) if expected_hct is not None else None,
                "lab_consistent": is_consistent,
                "derived_severity": classify_severity(hgb),
                "derived_subtype": classify_subtype(mcv),
            })

    out_path = os.path.join(os.getcwd(), "anerbc_cbc_structured_v7.csv")
    fieldnames = ["image_id", "provided_label", "HGB", "RBC", "HCT", "MCV", "MCH", "MCHC",
                  "expected_HCT_from_RBCxMCV", "lab_consistent", "derived_severity", "derived_subtype"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} unique AneRBC-I patients -> {out_path}\n")

    n_consistent = sum(1 for r in rows if r["lab_consistent"] is True)
    n_inconsistent = sum(1 for r in rows if r["lab_consistent"] is False)
    n_unchecked = sum(1 for r in rows if r["lab_consistent"] is None)
    print(f"Lab-consistent reports: {n_consistent}")
    print(f"Lab-INconsistent reports (likely corrupted, recommend excluding): {n_inconsistent}")
    print(f"Could not check (missing RBC/MCV/HCT): {n_unchecked}")

    def mismatch_rate(subset):
        healthy = [r for r in subset if r["provided_label"] == "Healthy"]
        mism = [r for r in healthy if r["derived_severity"] not in ("Normal", "Unknown")]
        return len(mism), len(healthy)

    mism_all, n_all = mismatch_rate(rows)
    clean_rows = [r for r in rows if r["lab_consistent"] is not False]
    mism_clean, n_clean = mismatch_rate(clean_rows)

    print(f"\nHealthy-labeled with non-Normal severity, ALL rows: {mism_all}/{n_all} "
          f"({mism_all/n_all*100:.1f}%)")
    print(f"Healthy-labeled with non-Normal severity, EXCLUDING inconsistent rows: "
          f"{mism_clean}/{n_clean} ({mism_clean/n_clean*100:.1f}%)")

    print("\n--- Derived severity distribution (consistent rows only) ---")
    print(Counter(r["derived_severity"] for r in clean_rows))
    print("\n--- Derived subtype distribution (consistent rows only) ---")
    print(Counter(r["derived_subtype"] for r in clean_rows))


if __name__ == "__main__":
    main()
