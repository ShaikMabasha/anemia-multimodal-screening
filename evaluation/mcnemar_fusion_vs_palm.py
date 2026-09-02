"""
Statistical significance test: Fusion vs. Palm baseline.

Uses McNemar's test, the statistically correct test for comparing two
classifiers' paired binary correct/incorrect outcomes on the SAME test
subjects -- not a t-test/Wilcoxon on 5 fold-level means, which has
essentially no power with n=5 folds.

Method:
  For each fold, evaluate BOTH the fusion model AND the palm baseline
  model (that fold's checkpoints) on the SAME fusion-test patients
  (the 443-patient matched subset). This gives two paired
  correct/incorrect arrays over the identical set of patients. Pool
  across all 5 folds (each patient appears in exactly one fold's test
  set, so pooling is valid, not double-counting).

  Build the 2x2 discordant-pair table:
      b = fusion correct, palm incorrect
      c = fusion incorrect, palm correct
  McNemar's exact test (binomial) on (b, c) tests whether the two
  classifiers differ significantly in how they get things right, using
  only the pairs where they disagree.

Usage:
    python mcnemar_fusion_vs_palm.py <kfold_manifest_csv> <base_dir>

Output:
    mcnemar_results.txt
"""

import os
import sys

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from scipy.stats import binomtest, chi2

IMG_SIZE = 224
BATCH_SIZE = 64
EMBED_DIM = 256
NUM_FOLDS = 5
LABEL_MAP = {"Anemic": 1, "Non-Anemic": 0}


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=4, dropout=0.3):
        super().__init__()
        self.proj_conj = nn.Linear(512, embed_dim)
        self.proj_nail = nn.Linear(512, embed_dim)
        self.proj_palm = nn.Linear(512, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim // 2, 2),
        )

    def forward(self, feat_conj, feat_nail, feat_palm):
        tok_conj = self.proj_conj(feat_conj).unsqueeze(1)
        tok_nail = self.proj_nail(feat_nail).unsqueeze(1)
        tok_palm = self.proj_palm(feat_palm).unsqueeze(1)
        tokens = torch.cat([tok_conj, tok_nail, tok_palm], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        fused = self.norm(tokens + attn_out)
        pooled = fused.mean(dim=1)
        return self.classifier(pooled)


class FullFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conj = self._build()
        self.enc_nail = self._build()
        self.enc_palm = self._build()
        self.fusion = CrossAttentionFusion()

    def _build(self):
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(m.fc.in_features, 2))
        m.fc = nn.Identity()
        return m

    def forward(self, img_conj, img_nail, img_palm):
        f_conj = self.enc_conj(img_conj)
        f_nail = self.enc_nail(img_nail)
        f_palm = self.enc_palm(img_palm)
        return self.fusion(f_conj, f_nail, f_palm)


def load_palm_baseline(ckpt_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


class FusionDataset(Dataset):
    def __init__(self, patient_to_paths, patients, transform):
        self.patient_to_paths = patient_to_paths
        self.patients = patients
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_key = self.patients[idx]
        info = self.patient_to_paths[patient_key]
        imgs = {}
        for modality in ["conjunctiva", "nail", "palm"]:
            path = info[modality][0]
            img = Image.open(path).convert("RGB")
            imgs[modality] = self.transform(img)
        label = LABEL_MAP[info["label"]]
        return imgs["conjunctiva"], imgs["nail"], imgs["palm"], label, patient_key


def build_patient_dicts(df):
    patient_to_paths = {}
    for patient_key, group in df.groupby("patient_key"):
        entry = {"label": group["label"].iloc[0]}
        for modality in ["conjunctiva", "nail", "palm"]:
            entry[modality] = group[group["norm_modality"] == modality]["filepath"].tolist()
        patient_to_paths[patient_key] = entry
    return patient_to_paths


def main():
    if len(sys.argv) < 3:
        print("Usage: python mcnemar_fusion_vs_palm.py <kfold_manifest_csv> <base_dir>")
        sys.exit(1)

    manifest_path = sys.argv[1]
    base_dir = sys.argv[2]
    codes_dir = os.path.join(base_dir, "codes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(manifest_path)
    modality_sets = {m: set(df[df["norm_modality"] == m]["patient_key"]) for m in ["conjunctiva", "nail", "palm"]}
    matched_patients = modality_sets["conjunctiva"] & modality_sets["nail"] & modality_sets["palm"]
    df = df[df["patient_key"].isin(matched_patients)]
    print(f"Matched patients across all 3 modalities: {len(matched_patients)}")

    tf = build_eval_transform()
    all_records = []

    for fold_idx in range(NUM_FOLDS):
        test_df = df[df["fold"] == fold_idx]
        test_dict = build_patient_dicts(test_df)
        test_patients = list(test_dict.keys())
        if not test_patients:
            continue

        loader = DataLoader(FusionDataset(test_dict, test_patients, tf),
                             batch_size=BATCH_SIZE, shuffle=False)

        fusion_ckpt = os.path.join(codes_dir, "kfold_fusion", f"fold{fold_idx}_model.pt")
        palm_ckpt = os.path.join(codes_dir, "kfold_palm", f"fold{fold_idx}_model.pt")

        fusion_model = FullFusionModel().to(device)
        fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        fusion_model.eval()

        palm_model = load_palm_baseline(palm_ckpt, device)

        with torch.no_grad():
            for img_c, img_n, img_p, labels, patient_keys in loader:
                img_c, img_n, img_p = img_c.to(device), img_n.to(device), img_p.to(device)

                fusion_logits = fusion_model(img_c, img_n, img_p)
                fusion_preds = torch.argmax(fusion_logits, dim=1).cpu().numpy()

                palm_logits = palm_model(img_p)
                palm_preds = torch.argmax(palm_logits, dim=1).cpu().numpy()

                labels_np = labels.numpy()

                for i in range(len(labels_np)):
                    all_records.append({
                        "patient_key": patient_keys[i],
                        "fold": fold_idx,
                        "true_label": int(labels_np[i]),
                        "fusion_correct": int(fusion_preds[i] == labels_np[i]),
                        "palm_correct": int(palm_preds[i] == labels_np[i]),
                    })

        print(f"Fold {fold_idx}: scored {len(test_patients)} patients "
              f"(fusion+palm, same patients, same images)")

    results_df = pd.DataFrame(all_records)
    results_df.to_csv("mcnemar_per_patient.csv", index=False)
    print(f"\nTotal paired patients pooled across all folds: {len(results_df)}")

    fusion_acc = results_df["fusion_correct"].mean()
    palm_acc = results_df["palm_correct"].mean()
    print(f"Pooled fusion accuracy on matched patients: {fusion_acc:.4f}")
    print(f"Pooled palm accuracy on the SAME matched patients: {palm_acc:.4f}")

    both_correct = ((results_df["fusion_correct"] == 1) & (results_df["palm_correct"] == 1)).sum()
    fusion_only = ((results_df["fusion_correct"] == 1) & (results_df["palm_correct"] == 0)).sum()
    palm_only = ((results_df["fusion_correct"] == 0) & (results_df["palm_correct"] == 1)).sum()
    both_wrong = ((results_df["fusion_correct"] == 0) & (results_df["palm_correct"] == 0)).sum()

    b, c = fusion_only, palm_only

    if b + c > 0:
        exact_result = binomtest(min(b, c), b + c, 0.5, alternative="two-sided")
        exact_p = exact_result.pvalue
    else:
        exact_p = 1.0

    if b + c > 0:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        chi2_p = 1 - chi2.cdf(chi2_stat, df=1)
    else:
        chi2_stat, chi2_p = 0.0, 1.0

    report = (
        f"McNEMAR'S TEST: FUSION vs. PALM BASELINE (paired, same patients)\n"
        f"{'='*65}\n"
        f"Total paired patients (pooled across 5 folds): {len(results_df)}\n\n"
        f"Pooled accuracy -- Fusion: {fusion_acc:.4f}   Palm (same patients): {palm_acc:.4f}\n\n"
        f"2x2 contingency table:\n"
        f"                    Palm correct   Palm incorrect\n"
        f"Fusion correct      {both_correct:>10d}   {fusion_only:>13d}\n"
        f"Fusion incorrect    {palm_only:>10d}   {both_wrong:>13d}\n\n"
        f"Discordant pairs: b (fusion-only-correct) = {b}, c (palm-only-correct) = {c}\n\n"
        f"Exact McNemar (binomial) p-value: {exact_p:.4f}\n"
        f"Chi-square (continuity-corrected) statistic: {chi2_stat:.4f}, p-value: {chi2_p:.4f}\n\n"
        f"INTERPRETATION:\n"
        f"p < 0.05 -> the difference in which patients each model gets right/wrong\n"
        f"is unlikely to be due to chance; fusion's improvement over palm is\n"
        f"statistically supported, not just a favorable average.\n"
        f"p >= 0.05 -> the observed improvement, while directionally consistent,\n"
        f"is not statistically distinguishable from chance at this sample size;\n"
        f"report the effect as suggestive/descriptive rather than confirmed.\n"
    )

    with open("mcnemar_results.txt", "w") as f:
        f.write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
