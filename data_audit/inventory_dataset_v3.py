"""
Dataset inventory script v3 for the anemia multimodal project.

New in v3 (building on v2):
- Full recursive directory tree printout per top-level dataset folder,
  down to leaf directories (folders with images and no subfolders),
  not just one level deep. This should reveal class folders that are
  nested inside modality/split folders, e.g.:
      New_Augmented_Anemia_Dataset/Conjuctiva/Anemic/...
      conjuctiva dataset/train/Non-Anemic/...
- Wider metadata file detection: adds .yaml, .yml, .mat, .npy, .xml
  (Roboflow-style exports often ship data.yaml or _annotations.csv/xml).
- For any leaf folder with NO label file anywhere above it in the tree,
  prints 5 sample filenames -- so we can visually check whether labels
  are encoded in filenames (e.g. img045_anemic.png, 12_a.png).

Usage:
    python inventory_dataset_v3.py "C:\\path\\to\\datasets"

Dependencies: standard library only (Pillow optional, for resolution stats).
    pip install pillow
"""

import os
import sys
import json
from collections import Counter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
LABEL_EXTS = {".csv", ".xlsx", ".xls", ".json", ".txt", ".yaml", ".yml", ".mat", ".npy", ".xml"}

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


def build_tree(path, depth=0, max_print_depth=6):
    """
    Recursively describe a directory tree. Returns a list of text lines
    and a flat list of (leaf_folder_path, image_count, has_label_nearby, sample_filenames).
    """
    lines = []
    leaves = []

    def _walk(p, depth, label_files_seen_on_path):
        entries = sorted(os.listdir(p))
        subdirs = [e for e in entries if os.path.isdir(os.path.join(p, e))]
        files = [e for e in entries if os.path.isfile(os.path.join(p, e))]

        images_here = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        labels_here = [f for f in files if os.path.splitext(f)[1].lower() in LABEL_EXTS]

        label_files_seen_on_path = label_files_seen_on_path or bool(labels_here)

        indent = "  " * depth
        tag = ""
        if images_here:
            tag += f" [{len(images_here)} images]"
        if labels_here:
            tag += f" [LABEL FILES: {labels_here[:3]}{'...' if len(labels_here) > 3 else ''}]"
        lines.append(f"{indent}{os.path.basename(p)}/{tag}")

        if not subdirs:
            # leaf folder
            if images_here:
                leaves.append({
                    "path": p,
                    "image_count": len(images_here),
                    "has_label_nearby": label_files_seen_on_path,
                    "sample_filenames": images_here[:5],
                })
            return

        if depth < max_print_depth:
            for sd in subdirs:
                _walk(os.path.join(p, sd), depth + 1, label_files_seen_on_path)
        else:
            lines.append(f"{indent}  ... (max depth reached, {len(subdirs)} more subfolders not expanded)")

    _walk(path, depth, False)
    return lines, leaves


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def main():
    root_path = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    root_path = os.path.abspath(root_path)

    if not os.path.isdir(root_path):
        print(f"ERROR: '{root_path}' is not a valid directory.")
        print("Usage: python inventory_dataset_v3.py /path/to/dataset")
        sys.exit(1)

    if not HAVE_PIL:
        print("Note: Pillow not installed -> skipping resolution stats.\n"
              "Install with: pip install pillow\n")

    top_level_dirs = sorted(
        d for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    )

    all_lines = []
    all_lines.append("=" * 70)
    all_lines.append("DATASET TREE + LABEL DETECTION REPORT (v3)")
    all_lines.append(f"Root: {root_path}")
    all_lines.append("=" * 70)
    all_lines.append("")

    unresolved_leaves = []

    for d in top_level_dirs:
        full = os.path.join(root_path, d)
        all_lines.append(f"### {d} ###")
        tree_lines, leaves = build_tree(full)
        all_lines.extend(tree_lines)
        all_lines.append("")

        for leaf in leaves:
            if not leaf["has_label_nearby"]:
                unresolved_leaves.append(leaf)

    all_lines.append("=" * 70)
    all_lines.append("LEAF FOLDERS WITH NO LABEL FILE ANYWHERE ON THEIR PATH")
    all_lines.append("(these need manual label resolution -- check filenames below)")
    all_lines.append("=" * 70)
    if unresolved_leaves:
        for leaf in unresolved_leaves:
            all_lines.append(f"\n{leaf['path']}")
            all_lines.append(f"  Images: {leaf['image_count']}")
            all_lines.append(f"  Sample filenames: {leaf['sample_filenames']}")
    else:
        all_lines.append("None -- every leaf folder has a label file somewhere on its path. Good.")
    all_lines.append("")

    report_text = "\n".join(all_lines)
    out_txt = os.path.join(os.getcwd(), "dataset_tree_report_v3.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nFull tree report saved to: {out_txt}")
    print("\nShare this file back and we'll pin down the label schema for every folder.")


if __name__ == "__main__":
    main()
