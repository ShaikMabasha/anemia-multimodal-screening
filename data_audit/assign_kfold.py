"""
K-fold assignment builder.

Assigns every unique patient (label, bare_code) to one of NUM_FOLDS folds,
stratified by label, with a fixed seed for reproducibility. This replaces
the single train/val/test split with a fold assignment that every training
script (baseline and fusion) will consume the same way, guaranteeing
consistency across the whole pipeline.

Usage:
    python assign_kfold.py unified_manifest.csv

Output:
    kfold_manifest.csv
        Same as input, with a new "fold" column (0..NUM_FOLDS-1) replacing
        "final_split". Old split column kept as "old_single_split" for
        reference.
"""

import sys
import random
import pandas as pd
from collections import defaultdict

SEED = 42
NUM_FOLDS = 5


def main():
    if len(sys.argv) < 2:
        print("Usage: python assign_kfold.py unified_manifest.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    patients_by_label = defaultdict(set)
    for _, row in df.drop_duplicates("patient_key").iterrows():
        patients_by_label[row["label"]].add(row["patient_key"])

    rng = random.Random(SEED)
    patient_fold = {}
    for label, patients in patients_by_label.items():
        patients = sorted(patients)
        rng.shuffle(patients)
        for i, p in enumerate(patients):
            patient_fold[p] = i % NUM_FOLDS

    df["old_single_split"] = df.get("final_split", None)
    df["fold"] = df["patient_key"].map(patient_fold)

    # sanity check
    check = df.groupby("patient_key")["fold"].nunique()
    leaked = (check > 1).sum()
    print(f"Patients with inconsistent fold assignment (should be 0): {leaked}")

    print("\nPatient counts per fold (all modalities combined, deduplicated by patient_key):")
    print(df.drop_duplicates("patient_key").groupby(["fold", "label"]).size().unstack(fill_value=0))

    print("\nImage row counts per modality / fold:")
    print(df.groupby(["norm_modality", "fold"]).size().unstack(fill_value=0))

    out_path = "kfold_manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
