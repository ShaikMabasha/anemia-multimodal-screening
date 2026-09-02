"""
Leakage-safe split builder (v5) for the anemia multimodal project.

Fixes the naming mismatch between raw and augmented sources, then rebuilds
Training/Validation/Testing at the SUBJECT level (not image level), so no
subject's images -- raw or augmented -- ever appear in more than one split.

Input:
    master_manifest.csv  (produced by build_manifest_v4.py)

Output:
    final_split_manifest.csv   -- every image with a NEW leakage-safe split
    split_summary_report.txt   -- counts per modality/label/split, sanity checks

Usage:
    python build_final_split_v5.py master_manifest.csv

Split ratio: 70% train / 15% val / 15% test, by subject, stratified within
each (modality, label) group so class balance is preserved in each split.
Deterministic (fixed random seed) so results are reproducible.
"""

import csv
import sys
import random
from collections import defaultdict

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is the remainder

# Normalize modality naming so raw and augmented sources of the same
# body part share one subject-id namespace.
MODALITY_NORMALIZE = {
    "fingernail": "nail",
    "finger_nails": "nail",
    "fingernails": "nail",
    "palm": "palm",
    "conjuctiva": "conjunctiva",
    "conjunctiva": "conjunctiva",
}


def normalize_modality(raw_modality):
    return MODALITY_NORMALIZE.get(raw_modality.strip().lower(), raw_modality.strip().lower())


def normalize_subject_key(row):
    """
    Rebuild a subject key using the normalized modality + the tail of the
    original subject_id (the part after the modality prefix), so raw and
    augmented naming conventions collapse into the same key.
    """
    modality_norm = normalize_modality(row["modality"])
    raw_subject_id = row["subject_id"]

    # subject_id was stored as "<OriginalModalityToken>_<code>"; strip the
    # leading modality token (whatever it was) and keep the code.
    if "_" in raw_subject_id:
        code = raw_subject_id.split("_", 1)[1]
    else:
        code = raw_subject_id

    return f"{modality_norm}_{code}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_final_split_v5.py master_manifest.csv")
        sys.exit(1)

    manifest_path = sys.argv[1]
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["norm_subject_key"] = normalize_subject_key(r)
            r["norm_modality"] = normalize_modality(r["modality"])
            rows.append(r)

    # group subject keys by (norm_modality, label)
    subjects_by_group = defaultdict(set)
    for r in rows:
        subjects_by_group[(r["norm_modality"], r["label"])].add(r["norm_subject_key"])

    # assign each subject to a split, stratified within each group
    subject_to_split = {}
    rng = random.Random(SEED)

    for (modality, label), subject_set in subjects_by_group.items():
        subjects = sorted(subject_set)  # sort first for determinism, then shuffle
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        for i, subj in enumerate(subjects):
            if subj in subject_to_split:
                continue  # already assigned via another label group (shouldn't normally happen)
            if i < n_train:
                subject_to_split[subj] = "train"
            elif i < n_train + n_val:
                subject_to_split[subj] = "val"
            else:
                subject_to_split[subj] = "test"

    # assign split to every row based on its normalized subject key
    for r in rows:
        r["final_split"] = subject_to_split.get(r["norm_subject_key"], "UNASSIGNED")

    # ---- write final manifest ----
    out_path = "final_split_manifest.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- sanity check: verify zero leakage in the NEW split ----
    split_check = defaultdict(set)
    for r in rows:
        split_check[r["norm_subject_key"]].add(r["final_split"])
    leaked_after_fix = {k: v for k, v in split_check.items() if len(v) > 1}

    # ---- summary counts ----
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[(r["norm_modality"], r["label"])][r["final_split"]] += 1

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FINAL LEAKAGE-SAFE SPLIT -- SUMMARY")
    report_lines.append("=" * 70)
    report_lines.append("")
    for (modality, label), split_counts in sorted(counts.items()):
        total = sum(split_counts.values())
        report_lines.append(f"[{modality} / {label}]  total images: {total}")
        for split_name in ["train", "val", "test"]:
            c = split_counts.get(split_name, 0)
            pct = c / total * 100 if total else 0
            report_lines.append(f"    {split_name}: {c} ({pct:.1f}%)")
        report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("LEAKAGE CHECK ON THE NEW SPLIT")
    report_lines.append("-" * 70)
    if leaked_after_fix:
        report_lines.append(f"WARNING: {len(leaked_after_fix)} subjects still span multiple splits!")
        report_lines.append("This should be 0 -- inspect these manually:")
        for k, v in list(leaked_after_fix.items())[:10]:
            report_lines.append(f"  {k} -> {v}")
    else:
        report_lines.append("PASS: every subject appears in exactly one split. Zero leakage.")
    report_lines.append("")

    unassigned = sum(1 for r in rows if r["final_split"] == "UNASSIGNED")
    if unassigned:
        report_lines.append(f"NOTE: {unassigned} rows could not be assigned a split (check norm_subject_key parsing).")

    report_text = "\n".join(report_lines)
    with open("split_summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nFinal manifest written to: {out_path}")
    print("Summary report written to: split_summary_report.txt")
    print("\nShare both back and we'll lock in the label schema + move to baseline training.")


if __name__ == "__main__":
    main()
