"""
Master manifest + leakage audit script (v4) for the anemia multimodal project.

What it does:
1. Builds one master CSV manifest across ALL image sources with columns:
     filepath, dataset_source, modality, label, provided_split, subject_id
2. For Fingernails/ and Palm/ (raw, flat folders): parses the anemic/
   non-anemic label directly from the filename, and extracts a best-guess
   subject_id.
3. For New_Augmented_Anemia_Dataset: extracts subject_id per image and
   checks whether the SAME subject_id appears in more than one of
   Training/Testing/Validation for that modality+class. This quantifies
   the leakage flagged manually in the last report.
4. Prints a leakage summary: how many subjects, and what % of images,
   are affected -- so you have hard numbers for your methods section.

Usage:
    python build_manifest_v4.py "C:\\path\\to\\datasets"

Output:
    master_manifest.csv          (every image, every source, with best-guess labels)
    leakage_audit_report.txt     (human-readable leakage summary)

NOTE: subject_id extraction uses filename heuristics (regex). It is a
best-effort guess -- spot check a handful of rows in master_manifest.csv
against the actual files before trusting it fully for the final split.
"""

import os
import re
import sys
import csv
from collections import defaultdict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def classify_label(filename):
    """Best-effort anemic/non-anemic classification from filename text."""
    low = filename.lower()
    # handle the "Non-Anrmic" typo variant seen in the Finger_Nails folders
    if re.search(r'non[-_]?an[er]mic', low):
        return "Non-Anemic"
    if "anemic" in low:
        return "Anemic"
    return "Unknown"


def extract_subject_id(filename, modality_hint=""):
    """
    Best-effort subject id extraction. Strips known label/modality tokens,
    then takes the leading alnum run before '_aug', ' (', or file extension.
    """
    base = os.path.splitext(filename)[0]
    # strip label tokens
    base = re.sub(r'(?i)non[-_]?an[er]mic', '', base)
    base = re.sub(r'(?i)anemic', '', base)
    # strip modality tokens that are redundant given modality_hint
    base = re.sub(r'(?i)^p[-_]', '', base)  # leading P- for palm sometimes
    base = base.strip('-_ ')
    # cut at first "_aug" (augmentation suffix) or " (" (copy-number suffix)
    base = re.split(r'(?i)_aug\d*', base)[0]
    base = re.split(r'\s*\(', base)[0]
    base = base.strip('-_ ')
    return base if base else "UNKNOWN"


def scan_flat_labeled_folder(folder_path, dataset_source, modality):
    """For Fingernails/ and Palm/ -- flat folders, label comes from filename."""
    rows = []
    for fname in sorted(os.listdir(folder_path)):
        full = os.path.join(folder_path, fname)
        if not os.path.isfile(full):
            continue
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            continue
        label = classify_label(fname)
        subject_id = extract_subject_id(fname, modality)
        rows.append({
            "filepath": full,
            "dataset_source": dataset_source,
            "modality": modality,
            "label": label,
            "provided_split": "N/A",
            "subject_id": f"{modality}_{subject_id}",
        })
    return rows


def scan_new_augmented(root_path):
    """
    Walks New_Augmented_Anemia_Dataset/<Modality>/<Split>/<Class>/*.png
    Returns rows + a leakage map: (modality,class) -> subject_id -> set(splits)
    """
    base = os.path.join(root_path, "New_Augmented_Anemia_Dataset")
    rows = []
    leakage_map = defaultdict(lambda: defaultdict(set))

    if not os.path.isdir(base):
        return rows, leakage_map

    for modality in sorted(os.listdir(base)):
        mod_path = os.path.join(base, modality)
        if not os.path.isdir(mod_path):
            continue
        for split in sorted(os.listdir(mod_path)):
            split_path = os.path.join(mod_path, split)
            if not os.path.isdir(split_path):
                continue
            for cls in sorted(os.listdir(split_path)):
                cls_path = os.path.join(split_path, cls)
                if not os.path.isdir(cls_path):
                    continue
                label = classify_label(cls)
                for fname in sorted(os.listdir(cls_path)):
                    full = os.path.join(cls_path, fname)
                    if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                        continue
                    subject_id = extract_subject_id(fname, modality)
                    key = f"{modality}_{subject_id}"
                    rows.append({
                        "filepath": full,
                        "dataset_source": "New_Augmented_Anemia_Dataset",
                        "modality": modality,
                        "label": label,
                        "provided_split": split,
                        "subject_id": key,
                    })
                    leakage_map[(modality, label)][key].add(split)

    return rows, leakage_map


def main():
    root_path = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    root_path = os.path.abspath(root_path)

    if not os.path.isdir(root_path):
        print(f"ERROR: '{root_path}' is not a valid directory.")
        sys.exit(1)

    all_rows = []

    fingernails_path = os.path.join(root_path, "Fingernails")
    if os.path.isdir(fingernails_path):
        all_rows.extend(scan_flat_labeled_folder(fingernails_path, "Fingernails_raw", "Fingernail"))

    palm_path = os.path.join(root_path, "Palm")
    if os.path.isdir(palm_path):
        all_rows.extend(scan_flat_labeled_folder(palm_path, "Palm_raw", "Palm"))

    aug_rows, leakage_map = scan_new_augmented(root_path)
    all_rows.extend(aug_rows)

    # ---- write master manifest ----
    manifest_path = os.path.join(os.getcwd(), "master_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filepath", "dataset_source", "modality", "label", "provided_split", "subject_id"
        ])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    # ---- leakage audit ----
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SUBJECT-LEVEL LEAKAGE AUDIT: New_Augmented_Anemia_Dataset")
    report_lines.append("=" * 70)
    report_lines.append("")

    total_subjects = 0
    total_leaked_subjects = 0
    total_images_in_leaked_subjects = 0
    total_images = 0

    for (modality, label), subj_map in sorted(leakage_map.items()):
        n_subjects = len(subj_map)
        leaked = {sid: splits for sid, splits in subj_map.items() if len(splits) > 1}
        total_subjects += n_subjects
        total_leaked_subjects += len(leaked)

        report_lines.append(f"[{modality} / {label}]")
        report_lines.append(f"  Unique subjects: {n_subjects}")
        report_lines.append(f"  Subjects appearing in >1 split: {len(leaked)} "
                             f"({(len(leaked)/n_subjects*100 if n_subjects else 0):.1f}%)")
        if leaked:
            example_ids = list(leaked.items())[:5]
            for sid, splits in example_ids:
                report_lines.append(f"    example: {sid} -> splits: {sorted(splits)}")
        report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append(f"TOTAL unique subjects across all modality/class groups: {total_subjects}")
    report_lines.append(f"TOTAL subjects with cross-split leakage: {total_leaked_subjects}")
    if total_subjects:
        report_lines.append(f"Leakage rate: {total_leaked_subjects/total_subjects*100:.1f}% of subjects")
    report_lines.append("-" * 70)
    report_lines.append("")
    report_lines.append("RECOMMENDATION: Do not use the provided Training/Testing/Validation")
    report_lines.append("split as-is. Rebuild splits by grouping all rows in master_manifest.csv")
    report_lines.append("by subject_id first, then assigning entire subject groups (not individual")
    report_lines.append("images) to train/val/test. This guarantees no subject's images appear in")
    report_lines.append("more than one partition.")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(os.getcwd(), "leakage_audit_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Master manifest written: {manifest_path} ({len(all_rows)} rows)")
    print()
    print(report_text)
    print(f"\nLeakage report saved to: {report_path}")
    print("\nShare both files back and we'll design the leakage-safe split + label schema.")


if __name__ == "__main__":
    main()
