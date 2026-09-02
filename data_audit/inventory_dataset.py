"""
Dataset inventory script for the anemia multimodal project.

What it does:
- Walks a root "dataset" folder (with subfolders like AneRBC, conjunctiva dataset,
  CP-AnemiC dataset, New_Augmented_Anemia_Dataset, Fingernails, Palm, etc.)
- For each subfolder, counts images, records formats and resolution ranges
- Flags candidate label files (.csv, .xlsx, .json, .txt)
- Detects likely duplicate filenames across folders (helps spot dataset overlap,
  e.g. CP-AnemiC vs "conjunctiva dataset" vs the augmented version)
- Writes a single summary report: dataset_inventory_report.txt (and .csv)

Usage:
    python inventory_dataset.py /path/to/dataset

If you don't pass a path, it defaults to a folder named "dataset" in the
current working directory.

Dependencies: only the Python standard library + Pillow (PIL) if available.
If Pillow isn't installed, resolution info is skipped (everything else still works).
    pip install pillow    # optional, for resolution stats
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


def inventory_folder(root_path):
    """Walk root_path; return per-subfolder stats and a global filename index."""
    subfolder_stats = {}
    filename_index = defaultdict(list)  # filename -> list of (subfolder, full_path)

    top_level_entries = sorted(
        d for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    )
    # Treat loose files directly under root as their own bucket
    top_level_files = sorted(
        f for f in os.listdir(root_path)
        if os.path.isfile(os.path.join(root_path, f))
    )

    def scan_one(subfolder_name, subfolder_path):
        img_count = 0
        ext_counter = Counter()
        widths, heights = [], []
        total_bytes = 0
        label_files = []
        corrupt_images = []

        for dirpath, _, filenames in os.walk(subfolder_path):
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
                    filename_index[fname].append((subfolder_name, full))
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
            "path": subfolder_path,
            "image_count": img_count,
            "ext_breakdown": dict(ext_counter),
            "total_size": total_bytes,
            "label_files": label_files,
            "corrupt_images": corrupt_images,
        }
        if widths:
            stats["res_min"] = f"{min(widths)}x{min(heights)}"
            stats["res_max"] = f"{max(widths)}x{max(heights)}"
            stats["res_mean"] = f"{sum(widths)//len(widths)}x{sum(heights)//len(heights)}"
            stats["res_unique_count"] = len(set(zip(widths, heights)))
        return stats

    for d in top_level_entries:
        subfolder_stats[d] = scan_one(d, os.path.join(root_path, d))

    if top_level_files:
        stats = scan_one("(root files)", root_path)
        # avoid double counting: only keep label files that are directly in root
        stats["label_files"] = [
            f for f in stats["label_files"]
            if os.path.dirname(f) == root_path
        ]
        stats["image_count"] = sum(
            1 for f in top_level_files
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        subfolder_stats["(root files)"] = stats

    return subfolder_stats, filename_index


def find_cross_folder_duplicates(filename_index):
    """Filenames that appear in more than one subfolder -> possible dataset overlap."""
    dupes = {
        fname: locs for fname, locs in filename_index.items()
        if len(set(loc[0] for loc in locs)) > 1
    }
    return dupes


def write_report(root_path, subfolder_stats, dupes, out_txt, out_csv):
    lines = []
    lines.append("=" * 70)
    lines.append("DATASET INVENTORY REPORT")
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
            for lf in stats["label_files"][:15]:
                lines.append(f"    - {lf}")
            if len(stats["label_files"]) > 15:
                lines.append(f"    ... and {len(stats['label_files']) - 15} more")
        else:
            lines.append("  Label/metadata files found: NONE (check labels are embedded in folder names?)")
        if stats["corrupt_images"]:
            lines.append(f"  WARNING - unreadable/corrupt images: {len(stats['corrupt_images'])}")
            for ci in stats["corrupt_images"][:5]:
                lines.append(f"    - {ci}")
        lines.append("")

    lines.append("-" * 70)
    lines.append(f"TOTAL IMAGES ACROSS ALL FOLDERS: {total_images}")
    lines.append("-" * 70)
    lines.append("")

    lines.append("=" * 70)
    lines.append("CROSS-FOLDER FILENAME OVERLAP (possible duplicate datasets)")
    lines.append("=" * 70)
    if dupes:
        lines.append(f"Found {len(dupes)} filenames appearing in more than one subfolder.")
        lines.append("This can mean: (a) genuine duplicate/overlapping datasets, or")
        lines.append("(b) coincidentally identical filenames (e.g. img001.jpg) from different sources.")
        lines.append("Spot-check a few of these manually before assuming true overlap.\n")
        shown = 0
        for fname, locs in dupes.items():
            if shown >= 30:
                lines.append(f"... and {len(dupes) - 30} more overlapping filenames (see CSV for full list)")
                break
            folders = sorted(set(loc[0] for loc in locs))
            lines.append(f"  {fname}  ->  found in: {', '.join(folders)}")
            shown += 1
    else:
        lines.append("No overlapping filenames found across subfolders.")
    lines.append("")

    report_text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subfolder", "image_count", "formats", "res_min", "res_max",
                          "res_mean", "total_size_bytes", "num_label_files", "num_corrupt_images"])
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
            ])

    return report_text


def main():
    root_path = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    root_path = os.path.abspath(root_path)

    if not os.path.isdir(root_path):
        print(f"ERROR: '{root_path}' is not a valid directory.")
        print("Usage: python inventory_dataset.py /path/to/dataset")
        sys.exit(1)

    print(f"Scanning: {root_path}")
    print("This may take a minute depending on dataset size...\n")

    subfolder_stats, filename_index = inventory_folder(root_path)
    dupes = find_cross_folder_duplicates(filename_index)

    out_txt = os.path.join(os.getcwd(), "dataset_inventory_report.txt")
    out_csv = os.path.join(os.getcwd(), "dataset_inventory_summary.csv")
    report_text = write_report(root_path, subfolder_stats, dupes, out_txt, out_csv)

    print(report_text)
    print(f"\nFull report saved to: {out_txt}")
    print(f"Summary CSV saved to: {out_csv}")
    print("\nShare both files back in chat and we'll design the label schema"
          " and train/val/test split plan from them.")


if __name__ == "__main__":
    main()
