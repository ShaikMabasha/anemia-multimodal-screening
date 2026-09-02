"""
Fusion model calibration: split conformal prediction + ECE.

For each of the 5 folds: uses that fold's VAL set as the calibration set
(split conformal prediction, standard approach), then evaluates coverage,
average prediction-set size, and Expected Calibration Error on that fold's
TEST set. Reports mean +/- std across folds -- this is the trustworthiness
evidence for the paper's "calibrated uncertainty" claim.

Usage:
    python calibrate_fusion_model.py <fusion_manifest_long_csv> <base_dir>

Output (into <base_dir>/codes/calibration/):
    calibration_summary_report.txt
    reliability_diagram.png
"""

import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

IMG_SIZE = 224
BATCH_SIZE = 64
EMBED_DIM = 256
NUM_FOLDS = 5
ALPHA = 0.10  # target 90% coverage
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
        attn_out, attn_weights = self.attn(tokens, tokens, tokens)
        fused = self.norm(tokens + attn_out)
        pooled = fused.mean(dim=1)
        return self.classifier(pooled), attn_weights


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
            logits, _ = model(img_c, img_n, img_p)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def split_conformal_calibrate(cal_probs, cal_labels, alpha):
    n = len(cal_labels)
    nonconformity = 1 - cal_probs[np.arange(n), cal_labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    threshold = np.quantile(nonconformity, q_level, method="higher")
    return threshold


def conformal_predict_sets(test_probs, threshold):
    nonconformity = 1 - test_probs
    included = nonconformity <= threshold
    return included


def expected_calibration_error(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_data.append((lo, hi, conf_in_bin, acc_in_bin, prop_in_bin))
    return ece, bin_data


def main():
    if len(sys.argv) < 3:
        print("Usage: python calibrate_fusion_model.py <fusion_manifest_csv> <base_dir>")
        sys.exit(1)

    manifest_path = sys.argv[1]
    base_dir = sys.argv[2]
    codes_dir = os.path.join(base_dir, "codes")
    out_dir = os.path.join(codes_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(manifest_path)
    modality_sets = {m: set(df[df["norm_modality"] == m]["patient_key"]) for m in ["conjunctiva", "nail", "palm"]}
    matched_patients = modality_sets["conjunctiva"] & modality_sets["nail"] & modality_sets["palm"]
    print(f"Patients matched across all 3 modalities: {len(matched_patients)}")
    df = df[df["patient_key"].isin(matched_patients)]
    tf = build_eval_transform()

    fold_results = []
    all_test_probs_for_plot, all_test_labels_for_plot = [], []

    for fold_idx in range(NUM_FOLDS):
        if "fold" not in df.columns:
            print("ERROR: manifest needs a 'fold' column (the k-fold fusion_manifest_long.csv).")
            sys.exit(1)
        val_df = df[df["fold"] == (fold_idx + 1) % NUM_FOLDS]
        test_df = df[df["fold"] == fold_idx]

        val_dict = build_patient_dicts(val_df)
        test_dict = build_patient_dicts(test_df)
        val_patients = list(val_dict.keys())
        test_patients = list(test_dict.keys())

        val_loader = DataLoader(FusionDataset(val_dict, val_patients, tf), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(FusionDataset(test_dict, test_patients, tf), batch_size=BATCH_SIZE, shuffle=False)

        ckpt_path = os.path.join(codes_dir, "kfold_fusion", f"fold{fold_idx}_model.pt")
        model = FullFusionModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        cal_probs, cal_labels = get_probs(model, val_loader, device)
        test_probs, test_labels = get_probs(model, test_loader, device)

        threshold = split_conformal_calibrate(cal_probs, cal_labels, ALPHA)
        pred_sets = conformal_predict_sets(test_probs, threshold)

        covered = pred_sets[np.arange(len(test_labels)), test_labels]
        coverage = covered.mean()
        avg_set_size = pred_sets.sum(axis=1).mean()

        ece, _ = expected_calibration_error(test_probs, test_labels)

        print(f"Fold {fold_idx}: coverage={coverage:.4f} (target {1-ALPHA:.2f}) "
              f"avg_set_size={avg_set_size:.3f} ECE={ece:.4f}")

        fold_results.append(dict(fold=fold_idx, coverage=coverage, avg_set_size=avg_set_size, ece=ece,
                                  conformal_threshold=threshold))
        all_test_probs_for_plot.append(test_probs)
        all_test_labels_for_plot.append(test_labels)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(out_dir, "fold_calibration_results.csv"), index=False)

    cov_mean, cov_std = results_df["coverage"].mean(), results_df["coverage"].std()
    size_mean, size_std = results_df["avg_set_size"].mean(), results_df["avg_set_size"].std()
    ece_mean, ece_std = results_df["ece"].mean(), results_df["ece"].std()

    summary = (
        f"FUSION MODEL CALIBRATION SUMMARY\n"
        f"{'='*60}\n"
        f"Target coverage (1-alpha): {1-ALPHA:.2f}\n\n"
        f"Per-fold results:\n{results_df.to_string(index=False)}\n\n"
        f"Empirical coverage : {cov_mean:.4f} +/- {cov_std:.4f}  (target: {1-ALPHA:.2f})\n"
        f"Avg prediction-set size : {size_mean:.3f} +/- {size_std:.3f}  (1.0 = fully confident, 2.0 = fully uncertain)\n"
        f"Expected Calibration Error (ECE) : {ece_mean:.4f} +/- {ece_std:.4f}  (lower is better, 0 = perfectly calibrated)\n\n"
        f"INTERPRETATION:\n"
        f"- Coverage should be close to {1-ALPHA:.2f}. If it's much lower, the model is\n"
        f"  overconfident (prediction sets too small/miss the true class too often).\n"
        f"  If much higher, prediction sets are unnecessarily large.\n"
        f"- avg_set_size near 1.0 means the model is usually confident and correct in\n"
        f"  its single prediction; near 2.0 means it's frequently including BOTH\n"
        f"  classes in its prediction set (genuinely uncertain cases).\n"
        f"- ECE below ~0.05-0.10 is generally considered reasonably well-calibrated\n"
        f"  for a deep learning model without post-hoc calibration.\n"
    )
    with open(os.path.join(out_dir, "calibration_summary_report.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)

    all_probs = np.concatenate(all_test_probs_for_plot, axis=0)
    all_labels = np.concatenate(all_test_labels_for_plot, axis=0)
    ece_all, bin_data = expected_calibration_error(all_probs, all_labels)

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.edgecolor": "#333333",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200)
    bin_centers = [(lo + hi) / 2 for lo, hi, _, _, _ in bin_data]
    bin_accs = [acc for _, _, _, acc, _ in bin_data]
    bin_confs = [conf for _, _, conf, _, _ in bin_data]

    # perfect calibration reference: a genuine straight y=x diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.5,
             label="Perfect calibration", zorder=1)

    ax.bar(bin_centers, bin_accs, width=0.085, alpha=0.85, color="#4C72B0",
           edgecolor="black", linewidth=0.8, label="Observed accuracy (per bin)", zorder=2)
    ax.scatter(bin_centers, bin_confs, color="#C44E52", s=60, zorder=3,
               edgecolor="black", linewidth=0.6, label="Average confidence (per bin)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(f"Reliability Diagram (pooled across folds)\nECE = {ece_all:.4f}",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reliability_diagram.png"), bbox_inches="tight")
    print(f"\nSaved reliability diagram: {out_dir}/reliability_diagram.png")


if __name__ == "__main__":
    main()
