"""
CP-AnemiC severity manifest builder.

Links the xlsx's IMAGE_ID column to actual image files (folders are split
only Anemic/Non-anemic, not by severity -- severity lives in the xlsx).

Usage:
    pip install openpyxl
    python build_severity_manifest.py <cp_anemic_dataset_dir> <xlsx_path>

Example:
    python build_severity_manifest.py \
        "/home/jupyter-mabasha/Anemia/datasets/CP-AnemiC dataset" \
        "/home/jupyter-mabasha/Anemia/datasets/CP-AnemiC dataset/Anemia_Data_Collection_Sheet.xlsx"

Output:
    severity_manifest.csv
        columns: filepath, image_id, severity, hb_level, age_months,
                 gender, hospital, region
"""

import os
import sys
import glob
import pandas as pd


def find_image_file(image_id, search_dirs):
    """Try several naming variants to match IMAGE_ID ('Image_001') to a real file."""
    candidates = [
        image_id,                          # Image_001
        image_id.lower(),                  # image_001
        image_id.replace("Image_", ""),    # 001
        image_id.replace("Image_", "").lstrip("0") or "0",  # 1 (no leading zeros)
    ]
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]

    for d in search_dirs:
        for cand in candidates:
            for ext in exts:
                p = os.path.join(d, cand + ext)
                if os.path.isfile(p):
                    return p
    # fallback: glob search for any file containing the numeric id
    numeric_part = "".join(ch for ch in image_id if ch.isdigit())
    for d in search_dirs:
        matches = glob.glob(os.path.join(d, f"*{numeric_part}*"))
        if len(matches) == 1:
            return matches[0]
    return None


def main():
    if len(sys.argv) < 3:
        print("Usage: python build_severity_manifest.py <cp_anemic_dataset_dir> <xlsx_path>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    xlsx_path = sys.argv[2]

    anemic_dir = os.path.join(dataset_dir, "Anemic")
    non_anemic_dir = os.path.join(dataset_dir, "Non-anemic")
    search_dirs = [d for d in [anemic_dir, non_anemic_dir] if os.path.isdir(d)]
    print(f"Searching in: {search_dirs}")

    df = pd.read_excel(xlsx_path, sheet_name="Anemia_Data_Collection_Sheet")
    print(f"Rows in xlsx: {len(df)}")
    print("Severity distribution:", df["Severity"].value_counts().to_dict())

    rows = []
    unmatched = []
    for _, row in df.iterrows():
        image_id = str(row["IMAGE_ID"]).strip()
        filepath = find_image_file(image_id, search_dirs)
        if filepath is None:
            unmatched.append(image_id)
            continue
        rows.append({
            "filepath": filepath,
            "image_id": image_id,
            "severity": row["Severity"],
            "hb_level": row["HB_LEVEL"],
            "age_months": row["Age(Months)"],
            "gender": row["GENDER"],
            "hospital": row["HOSPITAL"],
            "region": row["REGION"],
        })

    print(f"\nMatched: {len(rows)} / {len(df)}")
    if unmatched:
        print(f"Unmatched IMAGE_IDs ({len(unmatched)}): {unmatched[:10]}"
              f"{'...' if len(unmatched) > 10 else ''}")
        print("\nIf many are unmatched, check actual filenames in the Anemic/")
        print("Non-anemic folders (run: ls with head) and compare against IMAGE_ID format.")

    out_df = pd.DataFrame(rows)
    out_df.to_csv("severity_manifest.csv", index=False)
    print(f"\nSaved: severity_manifest.csv")
    print("\nFinal severity distribution (matched only):")
    print(out_df["severity"].value_counts())


if __name__ == "__main__":
    main()
