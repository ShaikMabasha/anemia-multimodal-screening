"""
Fusion manifest builder (v2) -- reuses the unified global split.

Difference from v1: v1 computed its OWN fresh split for the 443 matched
patients, independent of the baselines' split -- this was the root cause
of the leakage found in the first fusion run (baseline checkpoints had
already trained on patients fusion considered "held out"). This version
takes the split directly from unified_manifest.csv's "final_split" column,
so it is guaranteed consistent with whatever the retrained baselines use.

Usage:
    python build_fusion_manifest_v2.py unified_manifest.csv

Output:
    fusion_manifest_long.csv
        columns: patient_key, label, split, modality, filepath
"""

import sys
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_fusion_manifest_v2.py unified_manifest.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    modality_sets = {}
    for modality in ["conjunctiva", "nail", "palm"]:
        modality_sets[modality] = set(df[df["norm_modality"] == modality]["patient_key"])

    matched_patients = modality_sets["conjunctiva"] & modality_sets["nail"] & modality_sets["palm"]
    print(f"Patients matched across all 3 modalities: {len(matched_patients)}")

    fusion_df = df[df["patient_key"].isin(matched_patients)].copy()

    # sanity check: split must already be consistent per patient (guaranteed
    # by construction since it comes from the unified manifest, but verify anyway)
    check = fusion_df.groupby("patient_key")["final_split"].nunique()
    leaked = (check > 1).sum()
    print(f"Patients with inconsistent split (should be 0): {leaked}")

    print("\nPatient counts per split:")
    print(fusion_df.drop_duplicates("patient_key").groupby(["final_split", "label"]).size())

    out_cols = ["patient_key", "label", "final_split", "norm_modality", "filepath"]
    out_df = fusion_df[out_cols].rename(columns={"norm_modality": "modality", "final_split": "split"})
    out_path = "fusion_manifest_long.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
