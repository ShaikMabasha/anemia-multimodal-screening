"""
External validation: conjunctiva model on CP-AnemiC.

CP-AnemiC was never used to train the binary conjunctiva model (it was
held out from the start; it was separately used to train the SEVERITY
model, which is a different model/task, so no leakage here). This makes
it a genuine external test set: different population (Ghanaian hospital
patients vs. the New_Augmented_Anemia_Dataset source), different capture
conditions, entirely unseen by this model.

Evaluates ALL 5 conjunctiva fold checkpoints (from k-fold CV) on the full
CP-AnemiC set, reports:
  - overall performance per fold-model (mean +/- std, matching the
    internal CV reporting style for direct comparison)
  - performance broken down by HOSPITAL and REGION -- this is the actual
    domain-generalization / multi-site evidence for the paper

Usage:
    python evaluate_external_cpanemic.py <cp_anemic_dataset_dir> <base_dir>
"""

import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_FOLDS = 5
LABEL_MAP = {"Anemic": 1, "Non-anemic": 0}


class CPAnemiCDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        return self.transform(img), LABEL_MAP[row["remark"]], idx


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_conjunctiva_model(ckpt_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def build_dataframe(dataset_dir):
    """Uses the xlsx (already has REMARK, HOSPITAL, REGION) joined to actual
    image files, same matching logic as build_severity_manifest.py."""
    import glob

    xlsx_path = os.path.join(dataset_dir, "Anemia_Data_Collection_Sheet.xlsx")
    anemic_dir = os.path.join(dataset_dir, "Anemic")
    non_anemic_dir = os.path.join(dataset_dir, "Non-anemic")
    search_dirs = [d for d in [anemic_dir, non_anemic_dir] if os.path.isdir(d)]

    df = pd.read_excel(xlsx_path, sheet_name="Anemia_Data_Collection_Sheet")

    def find_image_file(image_id):
        candidates = [image_id, image_id.lower(), image_id.replace("Image_", ""),
                      image_id.replace("Image_", "").lstrip("0") or "0"]
        exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]
        for d in search_dirs:
            for cand in candidates:
                for ext in exts:
                    p = os.path.join(d, cand + ext)
                    if os.path.isfile(p):
                        return p
        numeric_part = "".join(ch for ch in image_id if ch.isdigit())
        for d in search_dirs:
            matches = glob.glob(os.path.join(d, f"*{numeric_part}*"))
            if len(matches) == 1:
                return matches[0]
        return None

    rows = []
    for _, row in df.iterrows():
        image_id = str(row["IMAGE_ID"]).strip()
        filepath = find_image_file(image_id)
        if filepath is None:
            continue
        rows.append({
            "filepath": filepath,
            "remark": row["REMARK"],
            "hospital": row["HOSPITAL"],
            "region": row["REGION"],
        })
    return pd.DataFrame(rows)


def evaluate_model(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []
    all_indices = []
    with torch.no_grad():
        for imgs, labels, idxs in tqdm(loader, leave=False):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_indices.extend(idxs.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), np.array(all_indices)


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_external_cpanemic.py <cp_anemic_dataset_dir> <base_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    base_dir = sys.argv[2]
    codes_dir = os.path.join(base_dir, "codes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = build_dataframe(dataset_dir)
    print(f"CP-AnemiC images matched: {len(df)}")
    print(f"Label distribution: {df['remark'].value_counts().to_dict()}")
    print(f"Hospitals: {df['hospital'].nunique()}, Regions: {df['region'].nunique()}")

    eval_tf = build_eval_transform()
    dataset = CPAnemiCDataset(df, eval_tf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0 if os.name == "nt" else 6)

    fold_results = []
    all_fold_probs = []  # for an ensemble-average view too

    for fold_idx in range(NUM_FOLDS):
        ckpt_path = os.path.join(codes_dir, "kfold_conjunctiva", f"fold{fold_idx}_model.pt")
        if not os.path.isfile(ckpt_path):
            print(f"WARNING: checkpoint not found: {ckpt_path} -- skipping fold {fold_idx}")
            continue

        model = load_conjunctiva_model(ckpt_path, device)
        labels, preds, probs, indices = evaluate_model(model, loader, device)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")

        print(f"Fold {fold_idx} on CP-AnemiC (external): acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
        fold_results.append(dict(fold=fold_idx, acc=acc, f1=f1, auc=auc))

        # keep per-row predictions from fold 0 ordering for the breakdown table
        # (indices should be consistent across folds since loader isn't shuffled)
        all_fold_probs.append(probs)

        # attach this fold's predictions to the dataframe for hospital/region breakdown
        df[f"fold{fold_idx}_pred"] = preds
        df[f"fold{fold_idx}_correct"] = (preds == labels).astype(int)

    df["true_label"] = df["remark"].map(LABEL_MAP)

    results_df = pd.DataFrame(fold_results)
    acc_mean, acc_std = results_df["acc"].mean(), results_df["acc"].std()
    f1_mean, f1_std = results_df["f1"].mean(), results_df["f1"].std()
    auc_mean, auc_std = results_df["auc"].mean(), results_df["auc"].std()

    # ensemble: average probability across folds, then threshold at 0.5
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    true_labels = df["true_label"].values
    ens_acc = accuracy_score(true_labels, ensemble_preds)
    ens_f1 = f1_score(true_labels, ensemble_preds)
    ens_auc = roc_auc_score(true_labels, ensemble_probs)

    correct_cols = [c for c in df.columns if c.endswith("_correct")]
    df["mean_correct_rate"] = df[correct_cols].mean(axis=1)

    hospital_breakdown = df.groupby("hospital")["mean_correct_rate"].agg(["mean", "count"]).sort_values("mean")
    region_breakdown = df.groupby("region")["mean_correct_rate"].agg(["mean", "count"]).sort_values("mean")

    summary = (
        f"EXTERNAL VALIDATION -- CONJUNCTIVA MODEL ON CP-ANEMIC\n"
        f"{'='*60}\n"
        f"CP-AnemiC images: {len(df)} (never used to train this model)\n\n"
        f"Per-fold-model external performance:\n{results_df.to_string(index=False)}\n\n"
        f"Accuracy : {acc_mean:.4f} +/- {acc_std:.4f}\n"
        f"F1 score : {f1_mean:.4f} +/- {f1_std:.4f}\n"
        f"AUC-ROC  : {auc_mean:.4f} +/- {auc_std:.4f}\n\n"
        f"5-fold ENSEMBLE (averaged probabilities):\n"
        f"  Accuracy: {ens_acc:.4f}  F1: {ens_f1:.4f}  AUC: {ens_auc:.4f}\n\n"
        f"{'='*60}\n"
        f"BREAKDOWN BY HOSPITAL (mean per-image correct rate across 5 fold-models)\n"
        f"{'='*60}\n"
        f"{hospital_breakdown.to_string()}\n\n"
        f"{'='*60}\n"
        f"BREAKDOWN BY REGION\n"
        f"{'='*60}\n"
        f"{region_breakdown.to_string()}\n"
    )

    out_dir = os.path.join(codes_dir, "external_validation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "external_validation_report.txt"), "w") as f:
        f.write(summary)
    df.to_csv(os.path.join(out_dir, "external_validation_per_image.csv"), index=False)

    print("\n" + summary)
    print(f"Saved to: {out_dir}/")


if __name__ == "__main__":
    main()
