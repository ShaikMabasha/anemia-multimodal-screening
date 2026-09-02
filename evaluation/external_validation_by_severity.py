"""
External validation stratified by severity.

Tests (rather than merely asserts) the hypothesis that CP-AnemiC's case
mix (66% of Anemic cases are Moderate/Severe) explains why external
accuracy exceeded internal cross-validation accuracy. If the hypothesis
is correct, per-image correctness should be higher for Moderate/Severe
cases than for Mild cases.

Usage:
    pip install openpyxl
    python external_validation_by_severity.py <cp_anemic_dataset_dir> <base_dir>

Output:
    external_by_severity_report.txt
"""

import os
import sys
import glob

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_FOLDS = 5
LABEL_MAP = {"Anemic": 1, "Non-anemic": 0}
SEVERITY_ORDER = ["Non-Anemic", "Mild", "Moderate", "Severe"]


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
            "severity": row["Severity"],
            "hospital": row["HOSPITAL"],
        })
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 3:
        print("Usage: python external_validation_by_severity.py <cp_anemic_dataset_dir> <base_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    base_dir = sys.argv[2]
    codes_dir = os.path.join(base_dir, "codes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = build_dataframe(dataset_dir)
    print(f"CP-AnemiC images matched: {len(df)}")
    print(f"Severity distribution:\n{df['severity'].value_counts()}")

    eval_tf = build_eval_transform()
    dataset = CPAnemiCDataset(df, eval_tf)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0 if os.name == "nt" else 6)

    all_fold_correct = []

    for fold_idx in range(NUM_FOLDS):
        ckpt_path = os.path.join(codes_dir, "kfold_conjunctiva", f"fold{fold_idx}_model.pt")
        if not os.path.isfile(ckpt_path):
            print(f"WARNING: missing checkpoint fold {fold_idx}, skipping")
            continue
        model = load_conjunctiva_model(ckpt_path, device)

        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels, idxs in tqdm(loader, leave=False):
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                preds_all.extend(preds)
                labels_all.extend(labels.numpy())

        correct = (np.array(preds_all) == np.array(labels_all)).astype(int)
        all_fold_correct.append(correct)
        df[f"fold{fold_idx}_correct"] = correct

    correct_cols = [c for c in df.columns if c.endswith("_correct")]
    df["mean_correct_rate"] = df[correct_cols].mean(axis=1)

    severity_breakdown = df.groupby("severity")["mean_correct_rate"].agg(["mean", "std", "count"])
    severity_breakdown = severity_breakdown.reindex(
        [s for s in SEVERITY_ORDER if s in severity_breakdown.index]
    )

    anemic_only = df[df["severity"] != "Non-Anemic"]
    anemic_severity_breakdown = anemic_only.groupby("severity")["mean_correct_rate"].agg(["mean", "std", "count"])
    anemic_severity_breakdown = anemic_severity_breakdown.reindex(
        [s for s in ["Mild", "Moderate", "Severe"] if s in anemic_severity_breakdown.index]
    )

    mild_acc = anemic_only[anemic_only["severity"] == "Mild"]["mean_correct_rate"].mean()
    modsev_acc = anemic_only[anemic_only["severity"].isin(["Moderate", "Severe"])]["mean_correct_rate"].mean()

    report = (
        f"EXTERNAL VALIDATION STRATIFIED BY SEVERITY\n"
        f"{'='*60}\n"
        f"Total CP-AnemiC images: {len(df)}\n\n"
        f"Accuracy by severity grade (mean correct rate across {len(all_fold_correct)} fold-models):\n"
        f"{severity_breakdown.to_string()}\n\n"
        f"Anemic cases only, by severity:\n"
        f"{anemic_severity_breakdown.to_string()}\n\n"
        f"Mild accuracy: {mild_acc:.4f}\n"
        f"Moderate+Severe accuracy: {modsev_acc:.4f}\n"
        f"Difference (Moderate+Severe minus Mild): {modsev_acc - mild_acc:+.4f}\n\n"
        f"INTERPRETATION:\n"
        f"If Moderate+Severe accuracy is meaningfully higher than Mild accuracy,\n"
        f"this directly supports the case-mix explanation for the external\n"
        f"validation result exceeding internal cross-validation accuracy.\n"
        f"If accuracy is similar or Mild is HIGHER, the case-mix explanation is\n"
        f"not supported and the true cause needs re-examination (e.g. image\n"
        f"quality/protocol differences between CP-AnemiC and the training source).\n"
    )

    with open("external_by_severity_report.txt", "w") as f:
        f.write(report)
    df.to_csv("external_validation_by_severity_per_image.csv", index=False)

    print("\n" + report)


if __name__ == "__main__":
    main()
