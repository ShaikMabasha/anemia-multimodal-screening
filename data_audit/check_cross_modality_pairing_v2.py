"""
Cross-modality subject-matching check (v2 - fixed).

Bug fixed from v1: subject numbering restarts independently within each
class (Anemic patient '020' and Non-Anemic patient '020' are different
real people). The identity key for matching across modalities must be
(label, bare_code) together, not bare_code alone.

Usage:
    python check_cross_modality_pairing_v2.py final_split_manifest_linux.csv
"""

import sys
import re
import pandas as pd
from collections import defaultdict


def bare_code(subject_key):
    tail = subject_key.split("_")[-1]
    tail = re.sub(r'^(FN|P)-?', '', tail, flags=re.IGNORECASE)
    m = re.search(r'\d+', tail)
    return m.group() if m else tail


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_cross_modality_pairing_v2.py final_split_manifest_linux.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    # identity = (modality, label, bare_code) -> we just need, per modality,
    # the SET of (label, code) identities present.
    modality_identities = defaultdict(set)

    for _, row in df.iterrows():
        modality = row["norm_modality"]
        code = bare_code(row["norm_subject_key"])
        label = row["label"]
        modality_identities[modality].add((label, code))

    modalities = list(modality_identities.keys())
    print("Modalities found:", modalities)
    for m in modalities:
        print(f"  {m}: {len(modality_identities[m])} unique (label, code) identities")

    print("\n--- Pairwise (label, code) identity overlap ---")
    overlap_sets = {}
    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            m1, m2 = modalities[i], modalities[j]
            ids1 = modality_identities[m1]
            ids2 = modality_identities[m2]
            overlap = ids1 & ids2
            overlap_sets[(m1, m2)] = overlap
            pct1 = len(overlap) / len(ids1) * 100 if ids1 else 0
            pct2 = len(overlap) / len(ids2) * 100 if ids2 else 0
            print(f"{m1} vs {m2}: {len(overlap)} matching (label, code) identities "
                  f"({pct1:.1f}% of {m1}, {pct2:.1f}% of {m2})")

    # three-way overlap (identities present in ALL three modalities)
    if len(modalities) == 3:
        m1, m2, m3 = modalities
        three_way = modality_identities[m1] & modality_identities[m2] & modality_identities[m3]
        print(f"\nIdentities present in ALL THREE modalities: {len(three_way)}")
        print("Sample matched identities:", list(three_way)[:10])

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("High 3-way overlap (most codes present in all 3 modalities, consistent")
    print("label) -> genuine patient-matched multimodal fusion is justified.")
    print("Low overlap -> codes don't correspond to the same patients; use")
    print("class-stratified synthetic pairing instead, and disclose as a limitation.")


if __name__ == "__main__":
    main()
