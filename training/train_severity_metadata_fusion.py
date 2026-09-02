"""
Severity grading: image + age/gender metadata fusion.

Rationale: WHO anemia severity thresholds are age-dependent, so age isn't
just a helpful auxiliary feature -- it's likely part of how the ground-truth
severity label itself was assigned. Fusing it in is a principled fix, not
a shortcut.

Architecture: ResNet18 image encoder (512-dim) + small MLP over
[normalized age, gender one-hot] (32-dim) -> concatenate -> classifier head.

Usage:
    python train_severity_metadata_fusion.py <severity_manifest_csv> <base_dir>

Output (into <base_dir>/codes/kfold_severity_metadata/):
    fold{0..4}_model.pt
    fold{0..4}_training_log.csv
    fold_results_summary.csv
    kfold_summary_report.txt
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

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
PATIENCE = 6
SEED = 42
NUM_FOLDS = 5
NUM_CLASSES = 4

LABEL_MAP = {"Non-Anemic": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
CLASS_NAMES = ["Non-Anemic", "Mild", "Moderate", "Severe"]
GENDER_MAP = {"Male": 0, "Female": 1}

torch.manual_seed(SEED)
np.random.seed(SEED)


def safe_multiclass_auc(all_labels, all_probs, num_classes):
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    aucs = []
    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true_bin, y_probs[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


class ImageMetaDataset(Dataset):
    def __init__(self, dataframe, transform, age_mean, age_std):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.age_mean = age_mean
        self.age_std = age_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.transform(img)

        age_norm = (row["age_months"] - self.age_mean) / (self.age_std + 1e-6)
        gender_idx = GENDER_MAP.get(row["gender"], 0)
        gender_onehot = [0.0, 0.0]
        gender_onehot[gender_idx] = 1.0
        meta = torch.tensor([age_norm] + gender_onehot, dtype=torch.float32)

        label = LABEL_MAP[row["severity"]]
        return img, meta, label


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


class ImageMetaFusionModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, meta_dim=3, dropout=0.5):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone  # outputs 512-dim

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, meta):
        img_feat = self.backbone(img)
        meta_feat = self.meta_mlp(meta)
        combined = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(combined)


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.set_grad_enabled(train):
        for imgs, meta, labels in tqdm(loader, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(imgs, meta)
                loss = criterion(outputs, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    auc = safe_multiclass_auc(all_labels, all_probs, NUM_CLASSES)
    return avg_loss, acc, f1_macro, auc, all_labels, all_preds


def run_one_fold(fold_idx, train_idx, val_idx, test_idx, df, device, out_dir):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    age_mean = train_df["age_months"].mean()
    age_std = train_df["age_months"].std()

    print(f"\n--- Fold {fold_idx} --- Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_tf, eval_tf = build_transforms()
    num_workers = 0 if os.name == "nt" else 6

    train_loader = DataLoader(ImageMetaDataset(train_df, train_tf, age_mean, age_std),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ImageMetaDataset(val_df, eval_tf, age_mean, age_std),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ImageMetaDataset(test_df, eval_tf, age_mean, age_std),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = ImageMetaFusionModel().to(device)

    mapped_labels = train_df["severity"].map(LABEL_MAP)
    counts = mapped_labels.value_counts().sort_index()
    total = counts.sum()
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts.get(c, 1)) for c in range(NUM_CLASSES)],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_f1, patience_counter = -1, 0
    log_rows = []
    ckpt_path = os.path.join(out_dir, f"fold{fold_idx}_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, tr_auc, *_ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc, val_f1, val_auc, *_ = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step(val_f1)

        print(f"  Epoch {epoch:2d}: train_f1={tr_f1:.4f} | val_f1={val_f1:.4f} val_auc={val_auc:.4f}")
        log_rows.append(dict(epoch=epoch, train_loss=tr_loss, train_f1=tr_f1,
                              val_loss=val_loss, val_f1=val_f1, val_auc=val_auc))

        if val_f1 > best_val_f1:
            best_val_f1, patience_counter = val_f1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    pd.DataFrame(log_rows).to_csv(os.path.join(out_dir, f"fold{fold_idx}_training_log.csv"), index=False)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc, test_f1, test_auc, test_labels, test_preds = run_epoch(
        model, test_loader, criterion, optimizer, scaler, device, train=False)
    print(f"  FOLD {fold_idx} TEST: acc={test_acc:.4f} macro_f1={test_f1:.4f} auc={test_auc:.4f}")

    cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_CLASSES)))
    return dict(fold=fold_idx, test_n=len(test_df), test_acc=test_acc, test_macro_f1=test_f1,
                test_auc=test_auc, confusion_matrix=cm)


def main():
    if len(sys.argv) < 3:
        print("Usage: python train_severity_metadata_fusion.py <severity_manifest_csv> <base_dir>")
        sys.exit(1)

    manifest_path = sys.argv[1]
    base_dir = sys.argv[2]
    out_dir = os.path.join(base_dir, "codes", "kfold_severity_metadata")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(manifest_path)
    df = df[df["severity"].isin(LABEL_MAP.keys())].reset_index(drop=True)
    print(f"Total images: {len(df)}")
    print(f"Class distribution:\n{df['severity'].value_counts()}")

    mapped = df["severity"].map(LABEL_MAP)
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(df, mapped)):
        trainval_labels = mapped.iloc[trainval_idx]
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_sub_idx, val_sub_idx = next(inner_skf.split(trainval_idx, trainval_labels))
        train_idx = trainval_idx[train_sub_idx]
        val_idx = trainval_idx[val_sub_idx]

        result = run_one_fold(fold_idx, train_idx, val_idx, test_idx, df, device, out_dir)
        fold_results.append(result)

    total_cm = sum(r["confusion_matrix"] for r in fold_results)
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"} for r in fold_results])
    results_df.to_csv(os.path.join(out_dir, "fold_results_summary.csv"), index=False)

    acc_mean, acc_std = results_df["test_acc"].mean(), results_df["test_acc"].std()
    f1_mean, f1_std = results_df["test_macro_f1"].mean(), results_df["test_macro_f1"].std()
    auc_mean, auc_std = results_df["test_auc"].mean(), results_df["test_auc"].std()

    summary = (
        f"K-FOLD CROSS-VALIDATION SUMMARY -- SEVERITY (IMAGE + AGE/GENDER FUSION)\n"
        f"{'='*60}\n"
        f"Classes: {CLASS_NAMES}\n\n"
        f"Per-fold results:\n{results_df.to_string(index=False)}\n\n"
        f"Accuracy   : {acc_mean:.4f} +/- {acc_std:.4f}\n"
        f"Macro-F1   : {f1_mean:.4f} +/- {f1_std:.4f}\n"
        f"AUC-ROC    : {auc_mean:.4f} +/- {auc_std:.4f}\n\n"
        f"Aggregated confusion matrix (summed across all 5 folds' test sets):\n"
        f"Rows = true class, Columns = predicted class, order = {CLASS_NAMES}\n"
        f"{total_cm}\n"
    )
    with open(os.path.join(out_dir, "kfold_summary_report.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
