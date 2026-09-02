"""
AneRBC-I CBC report parser (v6) for the anemia multimodal project.

Parses every CBC_reports/*.txt file (format shown in 001_a.txt) into a
structured row, then derives:
  - severity  : WHO-style banding on HGB (Normal/Mild/Moderate/Severe)
  - subtype   : MCV-based classification (Microcytic/Normocytic/Macrocytic)
  - chromia   : MCHC-based (Hypochromic/Normochromic)

Usage:
    python parse_anerbc_reports.py "C:\\path\\to\\datasets"

It expects the AneRBC dataset folder to contain, under some path, both:
    .../Anemic_individuals/CBC_reports/*.txt
    .../Healthy_individuals/CBC_reports/*.txt
(it searches recursively for any folder named "CBC_reports", so it does not
need the exact top-level folder name typed out.)

Output:
    anerbc_cbc_structured.csv

Each row's image_id matches the report filename (e.g. "001_a" or "001_h"),
which is how you join this back to the actual image files (Original_images/
001_a.png etc.) and to final_split_manifest.csv if you later fold AneRBC-I
into the same manifest format.
"""

import os
import re
import sys
import csv


def find_cbc_report_dirs(root_path):
    hits = []
    for dirpath, dirnames, _ in os.walk(root_path):
        if os.path.basename(dirpath) == "CBC_reports":
            hits.append(dirpath)
    return hits


def parse_report(filepath):
    """
    Parse a CBC report txt file of the form:
        Test,Result/Units,Norm. Range
        WBC,8.49 x10.e 3/µl,4 --- 10
        RBC,4.37 x10.e 6/μl,3.8 --- 4.8
        HGB,* 10.1 g/dL,12 --- 15
        ...
    Returns a dict of test_name -> (value_float, flagged_bool, unit_str)
    """
    values = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines[1:]:  # skip header row
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        test_name = parts[0].strip().upper()
        result_field = parts[1].strip()

        flagged = result_field.startswith("*")
        result_field = result_field.lstrip("*").strip()

        # extract the first number in the result field
        m = re.search(r'[-+]?\d*\.?\d+', result_field)
        if not m:
            continue
        value = float(m.group())

        # unit is whatever text remains after the number
        unit = result_field[m.end():].strip()

        values[test_name] = {"value": value, "flagged": flagged, "unit": unit}

    return values


def classify_severity(hgb):
    """WHO-style severity banding on HGB (g/dL). Adjust thresholds here if
    a different reference population is confirmed later."""
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
    """Standard MCV-based morphological classification."""
    if mcv is None:
        return "Unknown"
    if mcv < 80:
        return "Microcytic"
    elif mcv <= 100:
        return "Normocytic"
    else:
        return "Macrocytic"


def classify_chromia(mchc):
    """MCHC-based chromia classification."""
    if mchc is None:
        return "Unknown"
    if mchc < 31.5:
        return "Hypochromic"
    else:
        return "Normochromic"


def main():
    root_path = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    root_path = os.path.abspath(root_path)

    if not os.path.isdir(root_path):
        print(f"ERROR: '{root_path}' is not a valid directory.")
        sys.exit(1)

    cbc_dirs = find_cbc_report_dirs(root_path)
    if not cbc_dirs:
        print("No folders named 'CBC_reports' found under the given path.")
        print("Double-check the path points to somewhere above AneRBC-I/AneRBC-II.")
        sys.exit(1)

    print(f"Found {len(cbc_dirs)} CBC_reports folder(s):")
    for d in cbc_dirs:
        print(f"  - {d}")
    print()

    rows = []
    skipped = []

    for cbc_dir in cbc_dirs:
        # infer whether this is the Anemic or Healthy branch from the path
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

            try:
                values = parse_report(full)
            except Exception as e:
                skipped.append((full, str(e)))
                continue

            hgb = values.get("HGB", {}).get("value")
            mcv = values.get("MCV", {}).get("value")
            mchc = values.get("MCHC", {}).get("value")
            wbc = values.get("WBC", {}).get("value")
            rbc = values.get("RBC", {}).get("value")
            hct = values.get("HCT", {}).get("value")
            mch = values.get("MCH", {}).get("value")
            plt = values.get("PLT", {}).get("value")
            mpv = values.get("MPV", {}).get("value")

            rows.append({
                "image_id": image_id,
                "source_dir": cbc_dir,
                "provided_label": provided_label,
                "WBC": wbc,
                "RBC": rbc,
                "HGB": hgb,
                "HCT": hct,
                "MCV": mcv,
                "MCH": mch,
                "MCHC": mchc,
                "PLT": plt,
                "MPV": mpv,
                "derived_severity": classify_severity(hgb),
                "derived_subtype": classify_subtype(mcv),
                "derived_chromia": classify_chromia(mchc),
            })

    out_path = os.path.join(os.getcwd(), "anerbc_cbc_structured.csv")
    fieldnames = ["image_id", "source_dir", "provided_label", "WBC", "RBC", "HGB", "HCT",
                  "MCV", "MCH", "MCHC", "PLT", "MPV",
                  "derived_severity", "derived_subtype", "derived_chromia"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed {len(rows)} CBC reports -> {out_path}")
    if skipped:
        print(f"\n{len(skipped)} files failed to parse:")
        for path, err in skipped[:10]:
            print(f"  {path}: {err}")

    # quick summary
    from collections import Counter
    sev_counts = Counter(r["derived_severity"] for r in rows)
    sub_counts = Counter(r["derived_subtype"] for r in rows)
    label_counts = Counter(r["provided_label"] for r in rows)

    print("\n--- Summary ---")
    print("Provided label (from folder):", dict(label_counts))
    print("Derived severity (from HGB):", dict(sev_counts))
    print("Derived subtype (from MCV):", dict(sub_counts))

    # sanity check: do "Healthy" folder patients ever get a non-Normal severity?
    mismatches = [r for r in rows if r["provided_label"] == "Healthy" and r["derived_severity"] != "Normal"]
    if mismatches:
        print(f"\nNOTE: {len(mismatches)} patients folder-labeled 'Healthy' have HGB below the Normal band.")
        print("Worth spot-checking a few -- could mean a stricter clinical HGB cutoff was")
        print("used by the original dataset authors than the WHO threshold used here.")


if __name__ == "__main__":
    main()
