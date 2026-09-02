"""
Cross-modality subject-matching check.

Question: do the numeric subject codes in Conjunctiva/Nail/Palm actually
refer to the SAME patients (genuine multimodal pairing), or are they
independent numbering per modality that just happens to overlap in range?

Method: strip modality-specific prefixes (FN-, P-, etc.) down to the bare
numeric/alnum code, then check overlap across modalities' subject sets,
AND check whether the class label (Anemic/Non-Anemic) agrees for matching
codes -- if the same code is "Anemic" in one modality and "Non-Anemic" in
another, that's strong evidence the codes are coincidental, not the same
patient. If labels agree consistently, that's strong evidence of genuine
patient matching.

Usage:
    python check_cross_modality_pairing.py final_split_manifest_linux.csv
"""

import sys
import re
import pandas as pd
from collections import defaultdict


def bare_code(subject_key):
    """
    subject_key looks like 'conjunctiva_Conjuctiva_004' or
    'nail_Finger_Nails_FN-004' or 'palm_Palm_262' (norm_modality prefix,
    underscore, then the original token). Strip everything down to the
    trailing numeric/alnum identifier for cross-modality comparison.
    """
    # take the part after the last modality-ish token
    tail = subject_key.split("_")[-1]
    # strip a leading "FN-" / "P-" style prefix if present
    tail = re.sub(r'^(FN|P)-?', '', tail, flags=re.IGNORECASE)
    # keep only the numeric core
    m = re.search(r'\d+', tail)
    return m.group() if m else tail


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_cross_modality_pairing.py final_split_manifest_linux.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    # only look at the New_Augmented_Anemia_Dataset source (the one under
    # question) -- raw Fingernails/Palm folders already got merged into
    # the same norm_subject_key namespace earlier, which is fine to include too.
    modality_subjects = defaultdict(dict)  # modality -> bare_code -> set of labels seen

    for _, row in df.iterrows():
        modality = row["norm_modality"]
        code = bare_code(row["norm_subject_key"])
        modality_subjects[modality].setdefault(code, set()).add(row["label"])

    modalities = list(modality_subjects.keys())
    print("Modalities found:", modalities)
    for m in modalities:
        print(f"  {m}: {len(modality_subjects[m])} unique bare codes")

    # pairwise overlap
    print("\n--- Pairwise bare-code overlap ---")
    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            m1, m2 = modalities[i], modalities[j]
            codes1 = set(modality_subjects[m1].keys())
            codes2 = set(modality_subjects[m2].keys())
            overlap = codes1 & codes2
            print(f"{m1} vs {m2}: {len(overlap)} overlapping codes "
                  f"(out of {len(codes1)} / {len(codes2)})")

            # label agreement check on the overlap
            agree, disagree = 0, 0
            disagree_examples = []
            for code in overlap:
                labels1 = modality_subjects[m1][code]
                labels2 = modality_subjects[m2][code]
                # each modality might have augmented copies with consistent label (should be single-label per subject)
                if len(labels1) == 1 and len(labels2) == 1 and labels1 == labels2:
                    agree += 1
                else:
                    disagree += 1
                    if len(disagree_examples) < 5:
                        disagree_examples.append((code, labels1, labels2))

            total_checked = agree + disagree
            if total_checked > 0:
                print(f"  Label agreement on overlapping codes: {agree}/{total_checked} "
                      f"({agree/total_checked*100:.1f}%)")
                if disagree_examples:
                    print(f"  Example disagreements: {disagree_examples}")
            print()

    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("High overlap (>80%) + high label agreement (>90%) on overlap")
    print("  -> Strong evidence these ARE the same patients across modalities.")
    print("  -> Genuine patient-matched fusion is justified.")
    print()
    print("Low overlap, OR high overlap but low label agreement")
    print("  -> Codes are likely coincidental numbering, not the same patients.")
    print("  -> Use synthetic (class-stratified random) pairing for fusion,")
    print("     and disclose this as a limitation in the methods section.")


if __name__ == "__main__":
    main()
