"""
Unified patient-level split builder.

Root cause being fixed: the single-modality baselines and the fusion model
were trained on INDEPENDENTLY computed splits. A patient could be in a
baseline's training set and simultaneously in the fusion model's "held-out"
val/test set -- when fusion warm-starts from that baseline checkpoint, this
leaks information about supposedly-unseen patients.

Fix: compute ONE global (label, bare_code) patient identity -> split
assignment, using the UNION of patients across all three modalities. This
single mapping is then applied to every modality's rows, and reused
unchanged for the fusion patient set. No patient can ever be train in one
context and val/test in another, anywhere in the pipeline.

Usage:
    python build_unified_split.py final_split_manifest_linux.csv

Output:
    unified_manifest.csv
        Same columns as the input, but "final_split" is replaced with the
        new globally-consistent assignment (old column kept as
        "old_per_modality_split" for reference/debugging).
"""

import sys
import re
import random
import pandas as pd
from collections import defaultdict

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def bare_code(subject_key):
    tail = subject_key.split("_")[-1]
    tail = re.sub(r'^(FN|P)-?', '', tail, flags=re.IGNORECASE)
    m = re.search(r'\d+', tail)
    return m.group() if m else tail


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_unified_split.py final_split_manifest_linux.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    df = df[df["norm_modality"].isin(["conjunctiva", "nail", "palm"])].copy()
    df = df[df["label"].isin(["Anemic", "Non-Anemic"])].copy()

    df["bare_code"] = df["norm_subject_key"].apply(bare_code)
    df["patient_key"] = df["label"] + "_" + df["bare_code"]

    # global patient set: union across ALL modalities
    patients_by_label = defaultdict(set)
    for _, row in df.drop_duplicates("patient_key").iterrows():
        patients_by_label[row["label"]].add(row["patient_key"])

    total_patients = sum(len(v) for v in patients_by_label.values())
    print(f"Global unique patients (union across all modalities): {total_patients}")
    for label, pset in patients_by_label.items():
        print(f"  {label}: {len(pset)}")

    # assign split once, stratified by label
    rng = random.Random(SEED)
    patient_split = {}
    for label, patients in patients_by_label.items():
        patients = sorted(patients)
        rng.shuffle(patients)
        n = len(patients)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        for i, p in enumerate(patients):
            if i < n_train:
                patient_split[p] = "train"
            elif i < n_train + n_val:
                patient_split[p] = "val"
            else:
                patient_split[p] = "test"

    df["old_per_modality_split"] = df["final_split"]
    df["final_split"] = df["patient_key"].map(patient_split)

    # sanity check: verify no patient has mixed splits, and print per-modality counts
    check = df.groupby("patient_key")["final_split"].nunique()
    leaked = (check > 1).sum()
    print(f"\nPatients with inconsistent split (should be 0): {leaked}")

    print("\nImage counts per modality / split:")
    print(df.groupby(["norm_modality", "final_split"]).size())

    # how many rows changed split vs the old per-modality assignment?
    changed = (df["old_per_modality_split"] != df["final_split"]).sum()
    print(f"\nRows whose split assignment changed vs the old per-modality split: "
          f"{changed} / {len(df)} ({changed/len(df)*100:.1f}%)")
    print("(This is expected and fine -- it's exactly the mismatch we're fixing.)")

    out_path = "unified_manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\nNEXT STEPS:")
    print("1. Retrain all 3 baselines using this unified_manifest.csv (same")
    print("   train_baseline_modality.py script, just point it at the new file).")
    print("2. Rebuild the fusion manifest, reusing final_split from this file")
    print("   directly (do NOT recompute a separate fusion split this time).")
    print("3. Retrain fusion, warm-starting from the NEWLY retrained baselines.")


if __name__ == "__main__":
    main()
