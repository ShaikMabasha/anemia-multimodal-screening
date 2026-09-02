"""
AneRBC-I subtype manifest builder.

Links the already-parsed anerbc_cbc_structured_v7.csv (image_id ->
derived_subtype) to actual image files. Uses RGB_segmented images (already
isolated RBC morphology) rather than Original_images, since subtype
classification is specifically about cell size/shape -- segmentation
removes background/staining noise that isn't relevant to the task.

Only includes:
  - provided_label == "Anemic" (subtype only meaningful for confirmed anemic cases)
  - lab_consistent == True (excludes the corrupted reports found earlier)
  - derived_subtype != "Unknown" (excludes rows with missing MCV)

Usage:
    python build_subtype_manifest.py <anerbc_cbc_structured_v7_csv> <anerbc_dataset_dir>

Example:
    python build_subtype_manifest.py \
        "/home/jupyter-mabasha/Anemia/codes/anerbc_cbc_structured_v7.csv" \
        "/home/jupyter-mabasha/Anemia/datasets/AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. httpsdoi.org10.1093databasebaae120/AneRBC_dataset/AneRBC-I"

Output:
    subtype_manifest.csv
        columns: filepath, image_id, subtype, mcv, mch, mchc, hgb
"""

import os
import sys
import pandas as pd


def main():
    if len(sys.argv) < 3:
        print("Usage: python build_subtype_manifest.py <structured_csv> <anerbc1_dir>")
        sys.exit(1)

    structured_csv = sys.argv[1]
    anerbc1_dir = sys.argv[2]

    rgb_anemic_dir = os.path.join(anerbc1_dir, "Anemic_individuals", "RGB_segmented")
    if not os.path.isdir(rgb_anemic_dir):
        print(f"WARNING: expected folder not found: {rgb_anemic_dir}")
        print("Listing anerbc1_dir contents to help locate the right path:")
        for root, dirs, _ in os.walk(anerbc1_dir):
            print(" ", root, dirs)

    df = pd.read_csv(structured_csv)
    print(f"Total rows in structured CSV: {len(df)}")

    filtered = df[
        (df["provided_label"] == "Anemic") &
        (df["lab_consistent"].astype(str) == "True") &
        (df["derived_subtype"] != "Unknown")
    ].copy()
    print(f"After filtering (Anemic, lab-consistent, known subtype): {len(filtered)}")
    print("Subtype distribution in this Anemic-only cohort:")
    print(filtered["derived_subtype"].value_counts())

    rows = []
    missing = []
    for _, row in filtered.iterrows():
        image_id = row["image_id"]  # e.g. "001_a"
        filepath = os.path.join(rgb_anemic_dir, f"{image_id}.png")
        if not os.path.isfile(filepath):
            missing.append(image_id)
            continue
        rows.append({
            "filepath": filepath,
            "image_id": image_id,
            "subtype": row["derived_subtype"],
            "mcv": row["MCV"],
            "mch": row["MCH"],
            "mchc": row["MCHC"],
            "hgb": row["HGB"],
        })

    print(f"\nMatched to actual image files: {len(rows)} / {len(filtered)}")
    if missing:
        print(f"Missing files ({len(missing)}): {missing[:10]}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv("subtype_manifest.csv", index=False)
    print(f"\nSaved: subtype_manifest.csv")
    print("\nFinal subtype distribution:")
    print(out_df["subtype"].value_counts())

    macrocytic_count = (out_df["subtype"] == "Macrocytic").sum()
    if macrocytic_count < 15:
        print(f"\nNOTE: only {macrocytic_count} Macrocytic examples. This is too few for a")
        print("reliable 3-class model (some k-fold folds may have zero test examples of")
        print("this class). RECOMMENDATION: train a 2-class model instead --")
        print("Microcytic vs Non-Microcytic (Normocytic+Macrocytic merged) -- and report")
        print("Macrocytic separately as a small case-study/qualitative note, not a")
        print("trained class.")


if __name__ == "__main__":
    main()
