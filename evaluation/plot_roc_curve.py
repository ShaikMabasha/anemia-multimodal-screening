"""
ROC curve for the fusion model, computed across all 5 cross-validation folds.

Standard cross-validation ROC presentation: each fold's own ROC curve
(thin, semi-transparent), interpolated onto a common FPR grid, plus the
mean curve (bold) with a shaded +/-1 std band, plus the mean AUC.

Usage:
    python plot_roc_curve.py <kfold_manifest_csv> <base_dir>

Output:
    roc_curve_fusion.png
"""

import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import roc_curve, auc

IMG_SIZE = 224
BATCH_SIZE = 64
EMBED_DIM = 256
NUM_FOLDS = 5
LABEL_MAP = {"Anemic": 1, "Non-Anemic": 0}

mpl.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


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
        return imgs["conjunctiva"], imgs["nail"], imgs["palm"], label


def build_patient_dicts(df):
    patient_to_paths = {}
    for patient_key, group in df.groupby("patient_key"):
        entry = {"label": group["label"].iloc[0]}
        for modality in ["conjunctiva", "nail", "palm"]:
            entry[modality] = group[group["norm_modality"] == modality]["filepath"].tolist()
        patient_to_paths[patient_key] = entry
    return patient_to_paths


def get_probs(model, loader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for img_c, img_n, img_p, labels in loader:
            img_c, img_n, img_p = img_c.to(device), img_n.to(device), img_p.to(device)
            logits = model(img_c, img_n, img_p)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_roc_curve.py <kfold_manifest_csv> <base_dir>")
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

    tf = build_eval_transform()
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    fig, ax = plt.subplots(figsize=(7, 7), dpi=200)

    for fold_idx in range(NUM_FOLDS):
        test_df = df[df["fold"] == fold_idx]
        test_dict = build_patient_dicts(test_df)
        test_patients = list(test_dict.keys())
        loader = DataLoader(FusionDataset(test_dict, test_patients, tf), batch_size=BATCH_SIZE, shuffle=False)

        ckpt_path = os.path.join(codes_dir, "kfold_fusion", f"fold{fold_idx}_model.pt")
        model = FullFusionModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        probs, labels = get_probs(model, loader, device)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(fpr, tpr, alpha=0.35, linewidth=1.2, color="#4C72B0",
                label=f"Fold {fold_idx} (AUC = {roc_auc:.3f})")

        print(f"Fold {fold_idx}: AUC = {roc_auc:.4f}  (n={len(labels)})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color="#4C72B0", alpha=0.15,
                     label=r"$\pm$ 1 std. dev.")

    ax.plot(mean_fpr, mean_tpr, color="#C44E52", linewidth=2.8,
            label=f"Mean ROC (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2, label="Chance")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve \u2013 Fusion Model (5-fold cross-validation)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(codes_dir, "roc_curve_fusion.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    print(f"Mean AUC across folds: {mean_auc:.4f} +/- {std_auc:.4f}")


if __name__ == "__main__":
    main()
