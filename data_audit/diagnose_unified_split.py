"""
Unified split diagnostic -- investigates the nail accuracy collapse.

Checks two likely explanations:
1. Raw-vs-augmented image ratio mismatch between train and test per modality
   (a shift here could explain a large accuracy swing on its own).
2. Unique patient identity counts per modality/split, to check for any
   unexpected collapse or explosion in identity count vs the original
   per-modality split.

Usage:
    python diagnose_unified_split.py unified_manifest.csv
"""

import sys
import pandas as pd


def classify_source(filepath):
    """Raw vs augmented, based on path -- augmented images live under
    New_Augmented_Anemia_Dataset, raw images live under Fingernails/ or Palm/
    or the original conjuctiva/CP-AnemiC-style folders."""
    fp = filepath.replace("\\", "/").lower()
    if "new_augmented_anemia_dataset" in fp:
        return "augmented"
    else:
        return "raw"


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_unified_split.py unified_manifest.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    df["source_type"] = df["filepath"].apply(classify_source)

    print("=" * 70)
    print("1. RAW vs AUGMENTED ratio per modality / split")
    print("=" * 70)
    for modality in ["conjunctiva", "nail", "palm"]:
        mod_df = df[df["norm_modality"] == modality]
        print(f"\n[{modality}]")
        ct = mod_df.groupby(["final_split", "source_type"]).size().unstack(fill_value=0)
        ct["pct_augmented"] = (ct.get("augmented", 0) / (ct.get("augmented", 0) + ct.get("raw", 0)) * 100).round(1)
        print(ct)

    print("\n" + "=" * 70)
    print("2. Unique patient identity counts per modality / split")
    print("=" * 70)
    for modality in ["conjunctiva", "nail", "palm"]:
        mod_df = df[df["norm_modality"] == modality]
        print(f"\n[{modality}]")
        print(mod_df.drop_duplicates("patient_key").groupby(["final_split", "label"]).size())

    print("\n" + "=" * 70)
    print("3. Label balance per modality / split")
    print("=" * 70)
    for modality in ["conjunctiva", "nail", "palm"]:
        mod_df = df[df["norm_modality"] == modality]
        print(f"\n[{modality}]")
        print(mod_df.groupby(["final_split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
