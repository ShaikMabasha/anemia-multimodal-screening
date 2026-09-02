"""
Dataset inventory script v2 for the anemia multimodal project.

Fixes from v1:
- Root-level files are now scanned non-recursively only (v1 bug re-walked
  the whole tree and produced a bogus 47,000+ "duplicate" list -- ignore
  anything from v1's duplicate section).
- Adds a "class folder" breakdown: for every top-level dataset folder,
  lists its immediate subfolders and how many images are inside each.
  This is how we find labels that are encoded as folder names
  (e.g. Fingernails/Anemic/, Fingernails/Non-Anemic/) instead of a CSV.

Usage:
    python inventory_dataset_v2.py "C:\\path\\to\\datasets"

Dependencies: standard library only (Pillow optional, for resolution stats).
    pip install pillow
"""

import os
import sys
import csv
import json
from collections import defaultdict, Counter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
LABEL_EXTS = {".csv", ".xlsx", ".xls", ".json", ".txt"}

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False
    print("Note: Pillow not installed -> resolution stats will be skipped.")
    print("Install with:  pip install pillow\n")


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def scan_recursive(path):
    """Full recursive scan of one top-level dataset folder."""
    img_count = 0
    ext_counter = Counter()
    widths, heights = [], []
    total_bytes = 0
    label_files = []
    corrupt_images = []
    filenames_seen = []

    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0

            if ext in IMAGE_EXTS:
                img_count += 1
                ext_counter[ext] += 1
                total_bytes += size
                filenames_seen.append(fname)
                if HAVE_PIL:
                    try:
                        with Image.open(full) as im:
                            w, h = im.size
                            widths.append(w)
                            heights.append(h)
                    except Exception:
                        corrupt_images.append(full)
            elif ext in LABEL_EXTS:
                label_files.append(full)

    stats = {
        "path": path,
        "image_count": img_count,
        "ext_breakdown": dict(ext_counter),
        "total_size": total_bytes,
        "label_files": label_files,
        "corrupt_images": corrupt_images,
        "filenames": filenames_seen,
    }
    if widths:
        stats["res_min"] = f"{min(widths)}x{min(heights)}"
        stats["res_max"] = f"{max(widths)}x{max(heights)}"
        stats["res_mean"] = f"{sum(widths)//len(widths)}x{sum(heights)//len(heights)}"
        stats["res_unique_count"] = len(set(zip(widths, heights)))
    return stats


def scan_root_files_only(root_path):
    """Only files sitting DIRECTLY in root_path -- no recursion (this is the bug fix)."""
    img_count = 0
    ext_counter = Counter()
    total_bytes = 0
    label_files = []

    for fname in os.listdir(root_path):
        full = os.path.join(root_path, fname)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(fname)[1].lower()
        size = os.path.getsize(full)
        if ext in IMAGE_EXTS:
            img_count += 1
            ext_counter[ext] += 1
            total_bytes += size
        elif ext in LABEL_EXTS:
            label_files.append(full)

    return {
        "path": root_path,
        "image_count": img_count,
        "ext_breakdown": dict(ext_counter),
        "total_size": total_bytes,
        "label_files": label_files,
        "corrupt_images": [],
        "filenames": [],
    }


def get_class_folder_breakdown(dataset_path, max_depth=2):
    """
    Look one level (or two) into a dataset folder and count images per
    immediate subfolder. This is how we detect folder-name-as-label datasets.
    Returns a dict: subfolder_name -> image_count (only for folders with images).
    """
    breakdown = {}

    def count_images_in(p):
        n = 0
        for dirpath, _, filenames in os.walk(p):
            n += sum(1 for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        return n

    try:
        entries = sorted(os.listdir(dataset_path))
    except OSError:
        return breakdown

    for entry in entries:
        full = os.path.join(dataset_path, entry)
        if os.path.isdir(full):
            breakdown[entry] = count_images_in(full)

    return breakdown


def inventory_folder(root_path):
    subfolder_stats = {}
    class_breakdowns = {}
    filename_index = defaultdict(set)  # filename -> set of subfolder names

    top_level_dirs = sorted(
        d for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    )

    for d in top_level_dirs:
        full = os.path.join(root_path, d)
        stats = scan_recursive(full)
        subfolder_stats[d] = stats
        for fname in stats["filenames"]:
            filename_index[fname].add(d)

        # If no label files were found, look for a folder-name-as-label pattern
        if not stats["label_files"]:
            class_breakdowns[d] = get_class_folder_breakdown(full)

    # Root-level loose files (non-recursive -- bug fix from v1)
    root_stats = scan_root_files_only(root_path)
    if root_stats["image_count"] or root_stats["label_files"]:
        subfolder_stats["(root files)"] = root_stats

    return subfolder_stats, class_breakdowns, filename_index


def find_cross_folder_duplicates(filename_index):
    return {
        fname: folders for fname, folders in filename_index.items()
        if len(folders) > 1
    }


def write_report(root_path, subfolder_stats, class_breakdowns, dupes, out_txt, out_csv):
    lines = []
    lines.append("=" * 70)
    lines.append("DATASET INVENTORY REPORT (v2 - bug fixed)")
    lines.append(f"Root: {root_path}")
    lines.append("=" * 70)
    lines.append("")

    total_images = 0
    for name, stats in subfolder_stats.items():
        total_images += stats["image_count"]
        lines.append(f"[{name}]")
        lines.append(f"  Path: {stats['path']}")
        lines.append(f"  Image count: {stats['image_count']}")
        if stats["ext_breakdown"]:
            lines.append(f"  Formats: {stats['ext_breakdown']}")
        if "res_min" in stats:
            lines.append(f"  Resolution range: {stats['res_min']} to {stats['res_max']} "
                          f"(mean ~{stats['res_mean']}, {stats['res_unique_count']} distinct sizes)")
        lines.append(f"  Total size: {human_size(stats['total_size'])}")
        if stats["label_files"]:
            lines.append(f"  Label/metadata files found ({len(stats['label_files'])}):")
            for lf in stats["label_files"][:10]:
                lines.append(f"    - {lf}")
            if len(stats["label_files"]) > 10:
                lines.append(f"    ... and {len(stats['label_files']) - 10} more")
        else:
            lines.append("  Label/metadata files found: NONE")

        if name in class_breakdowns and class_breakdowns[name]:
            lines.append("  Subfolder breakdown (checking for folder-name-as-label):")
            for sub, cnt in class_breakdowns[name].items():
                lines.append(f"    - {sub}: {cnt} images")

        if stats["corrupt_images"]:
            lines.append(f"  WARNING - unreadable/corrupt images: {len(stats['corrupt_images'])}")
        lines.append("")

    lines.append("-" * 70)
    lines.append(f"TOTAL IMAGES ACROSS ALL FOLDERS: {total_images}")
    lines.append("-" * 70)
    lines.append("")

    lines.append("=" * 70)
    lines.append("CROSS-FOLDER FILENAME OVERLAP (fixed - genuine cross-dataset only)")
    lines.append("=" * 70)
    if dupes:
        lines.append(f"Found {len(dupes)} filenames appearing in more than one TOP-LEVEL dataset folder.")
        lines.append("Spot-check a few manually -- could be true overlap or coincidental naming.\n")
        shown = 0
        for fname, folders in dupes.items():
            if shown >= 30:
                lines.append(f"... and {len(dupes) - 30} more (see CSV for full list)")
                break
            lines.append(f"  {fname}  ->  found in: {', '.join(sorted(folders))}")
            shown += 1
    else:
        lines.append("No overlapping filenames found across top-level dataset folders. Good sign.")
    lines.append("")

    report_text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subfolder", "image_count", "formats", "res_min", "res_max",
                          "res_mean", "total_size_bytes", "num_label_files",
                          "num_corrupt_images", "class_breakdown"])
        for name, stats in subfolder_stats.items():
            writer.writerow([
                name,
                stats["image_count"],
                json.dumps(stats["ext_breakdown"]),
                stats.get("res_min", ""),
                stats.get("res_max", ""),
                stats.get("res_mean", ""),
                stats["total_size"],
                len(stats["label_files"]),
                len(stats["corrupt_images"]),
                json.dumps(class_breakdowns.get(name, {})),
            ])

    return report_text


def main():
    root_path = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    root_path = os.path.abspath(root_path)

    if not os.path.isdir(root_path):
        print(f"ERROR: '{root_path}' is not a valid directory.")
        print("Usage: python inventory_dataset_v2.py /path/to/dataset")
        sys.exit(1)

    print(f"Scanning: {root_path}")
    print("This may take a minute depending on dataset size...\n")

    subfolder_stats, class_breakdowns, filename_index = inventory_folder(root_path)
    dupes = find_cross_folder_duplicates(filename_index)

    out_txt = os.path.join(os.getcwd(), "dataset_inventory_report_v2.txt")
    out_csv = os.path.join(os.getcwd(), "dataset_inventory_summary_v2.csv")
    report_text = write_report(root_path, subfolder_stats, class_breakdowns, dupes, out_txt, out_csv)

    print(report_text)
    print(f"\nFull report saved to: {out_txt}")
    print(f"Summary CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
