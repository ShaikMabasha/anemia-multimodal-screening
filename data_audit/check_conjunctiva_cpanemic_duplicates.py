"""
Perceptual duplicate check: New_Augmented Conjunctiva vs CP-AnemiC.

The external validation result (95% ensemble accuracy on supposedly unseen
data) is implausibly high -- higher than internal cross-validation, which
should never happen for a genuine external test set. Leading hypothesis:
New_Augmented_Anemia_Dataset's Conjunctiva images were built by augmenting
CP-AnemiC's own 710 images, making "external validation" actually test on
(transformed) training data.

This uses perceptual hashing (pHash), which is robust to resizing,
compression, and minor color/contrast changes (the kind of transformations
augmentation applies) -- unlike exact filename or byte-level matching,
which we already checked and found nothing.

Usage:
    pip install imagehash
    python check_conjunctiva_cpanemic_duplicates.py <new_augmented_conjunctiva_dir> <cp_anemic_dataset_dir>

Output:
    duplicate_check_report.txt
    duplicate_pairs.csv  (every CP-AnemiC image matched to its closest
                           New_Augmented Conjunctiva image + hash distance)
"""

import os
import sys

import pandas as pd
from PIL import Image
import imagehash


def collect_images(root_dir, max_images=None):
    exts = (".png", ".jpg", ".jpeg")
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(exts):
                paths.append(os.path.join(dirpath, f))
    if max_images:
        paths = paths[:max_images]
    return paths


def compute_hashes(paths, label):
    hashes = {}
    failed = 0
    for p in paths:
        try:
            with Image.open(p) as img:
                h = imagehash.phash(img, hash_size=16)  # larger hash = more precision
                hashes[p] = h
        except Exception:
            failed += 1
    print(f"[{label}] Hashed {len(hashes)} images ({failed} failed to open)")
    return hashes


def main():
    if len(sys.argv) < 3:
        print("Usage: python check_conjunctiva_cpanemic_duplicates.py "
              "<new_augmented_conjunctiva_dir> <cp_anemic_dataset_dir>")
        sys.exit(1)

    new_aug_dir = sys.argv[1]
    cp_anemic_dir = sys.argv[2]

    print("Collecting CP-AnemiC images (all 710, this is the full comparison target)...")
    cp_paths = collect_images(cp_anemic_dir)
    print(f"Found {len(cp_paths)} CP-AnemiC images")

    print("\nCollecting New_Augmented Conjunctiva images...")
    print("(This could be 10,000+ images -- hashing all of them, this may take a few minutes)")
    new_aug_paths = collect_images(new_aug_dir)
    print(f"Found {len(new_aug_paths)} New_Augmented Conjunctiva images")

    cp_hashes = compute_hashes(cp_paths, "CP-AnemiC")
    new_aug_hashes = compute_hashes(new_aug_paths, "New_Augmented Conjunctiva")

    print("\nComparing every CP-AnemiC image against every New_Augmented Conjunctiva image...")
    print("(this is O(n*m) -- may take a while given the dataset sizes)")

    results = []
    new_aug_items = list(new_aug_hashes.items())

    for i, (cp_path, cp_hash) in enumerate(cp_hashes.items()):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(cp_hashes)} CP-AnemiC images...")

        best_match_path = None
        best_distance = None
        for aug_path, aug_hash in new_aug_items:
            dist = cp_hash - aug_hash  # Hamming distance
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_match_path = aug_path

        results.append({
            "cp_anemic_image": cp_path,
            "closest_new_augmented_match": best_match_path,
            "hash_distance": best_distance,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("duplicate_pairs.csv", index=False)

    near_dupe_threshold = 20
    n_near_dupes = (results_df["hash_distance"] <= near_dupe_threshold).sum()

    report = (
        f"PERCEPTUAL DUPLICATE CHECK: CP-AnemiC vs New_Augmented Conjunctiva\n"
        f"{'='*70}\n"
        f"CP-AnemiC images checked: {len(cp_hashes)}\n"
        f"New_Augmented Conjunctiva images compared against: {len(new_aug_hashes)}\n\n"
        f"Hash distance distribution (lower = more similar, 0 = identical):\n"
        f"{results_df['hash_distance'].describe().to_string()}\n\n"
        f"CP-AnemiC images with a near-duplicate match (distance <= {near_dupe_threshold}): "
        f"{n_near_dupes} / {len(results_df)} ({n_near_dupes/len(results_df)*100:.1f}%)\n\n"
        f"INTERPRETATION:\n"
        f"If most CP-AnemiC images have a near-duplicate in New_Augmented Conjunctiva,\n"
        f"the external validation result is invalid (it tested on the model's own\n"
        f"training data, just re-augmented). If very few/no near-duplicates are found,\n"
        f"the two datasets are genuinely independent, and the high external accuracy\n"
        f"needs a different explanation (e.g. CP-AnemiC's more visually distinct,\n"
        f"clinically cleaner pediatric population is simply an easier task).\n"
    )
    with open("duplicate_check_report.txt", "w") as f:
        f.write(report)

    print("\n" + report)
    print("Saved: duplicate_check_report.txt, duplicate_pairs.csv")


if __name__ == "__main__":
    main()
