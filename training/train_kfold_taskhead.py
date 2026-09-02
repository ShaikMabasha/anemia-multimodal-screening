"""
K-fold training for severity (4-class) or subtype (2-class merged) grading.

Usage:
    python train_kfold_taskhead.py severity <severity_manifest_csv> <base_dir>
    python train_kfold_taskhead.py subtype <subtype_manifest_csv> <base_dir>

Output (into <base_dir>/codes/kfold_<task>/):
    fold{0..4}_model.pt
    fold{0..4}_training_log.csv
    fold_results_summary.csv
    kfold_summary_report.txt
"""

import os
import sys
import random

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


def safe_multiclass_auc(all_labels, all_probs, num_classes):
    """One-vs-rest AUC per class, skipping any class that isn't present
    (or is entirely present) in this particular evaluation set, then
    macro-averaging over whatever classes COULD be computed. This avoids
    the systematic NaN failure of sklearn's roc_auc_score(multi_class='ovr')
    when a rare class has too few examples in a given fold."""
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

IMG_SIZE = 224
BATCH_SIZE = 32          # smaller datasets -- smaller batch is fine and more stable
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
PATIENCE = 6
SEED = 42
NUM_FOLDS = 5

TASK_CONFIG = {
    "severity": {
        "label_col": "severity",
        "label_map": {"Non-Anemic": 0, "Mild": 1, "Moderate": 2, "Severe": 3},
        "class_names": ["Non-Anemic", "Mild", "Moderate", "Severe"],
    },
    "subtype": {
        "label_col": "subtype",
        # merge Normocytic + Macrocytic -> "Non-Microcytic" (0), Microcytic -> 1
        "label_map": {"Microcytic": 1, "Normocytic": 0, "Macrocytic": 0},
        "class_names": ["Non-Microcytic", "Microcytic"],
    },
}

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class ImageLabelDataset(Dataset):
    def __init__(self, dataframe, transform, label_col, label_map):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        label = self.label_map[row[self.label_col]]
        return self.transform(img), label


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


def run_epoch(model, loader, criterion, optimizer, scaler, device, num_classes, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(imgs)
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
    try:
        if num_classes == 2:
            probs_pos = [p[1] for p in all_probs]
            auc = roc_auc_score(all_labels, probs_pos)
        else:
            auc = safe_multiclass_auc(all_labels, all_probs, num_classes)
    except ValueError:
        auc = float("nan")
    return avg_loss, acc, f1_macro, auc, all_labels, all_preds


def run_one_fold(fold_idx, train_idx, val_idx, test_idx, df, config, device, out_dir):
    label_col = config["label_col"]
    label_map = config["label_map"]
    num_classes = len(set(label_map.values()))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    print(f"\n--- Fold {fold_idx} --- Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_tf, eval_tf = build_transforms()
    num_workers = 0 if os.name == "nt" else 6

    train_loader = DataLoader(ImageLabelDataset(train_df, train_tf, label_col, label_map),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ImageLabelDataset(val_df, eval_tf, label_col, label_map),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ImageLabelDataset(test_df, eval_tf, label_col, label_map),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, num_classes))
    model = model.to(device)

    # class weights from training fold
    mapped_labels = train_df[label_col].map(label_map)
    counts = mapped_labels.value_counts().sort_index()
    total = counts.sum()
    weights = torch.tensor(
        [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    print(f"  Class weights: {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_f1, patience_counter = -1, 0
    log_rows = []
    ckpt_path = os.path.join(out_dir, f"fold{fold_idx}_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, tr_auc, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, num_classes, train=True)
        val_loss, val_acc, val_f1, val_auc, _, _ = run_epoch(model, val_loader, criterion, optimizer, scaler, device, num_classes, train=False)
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
        model, test_loader, criterion, optimizer, scaler, device, num_classes, train=False)
    print(f"  FOLD {fold_idx} TEST: acc={test_acc:.4f} macro_f1={test_f1:.4f} auc={test_auc:.4f}")

    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_classes)))

    return dict(fold=fold_idx, test_n=len(test_df), test_acc=test_acc, test_macro_f1=test_f1,
                test_auc=test_auc, confusion_matrix=cm)


def main():
    if len(sys.argv) < 4:
        print("Usage: python train_kfold_taskhead.py <severity|subtype> <manifest_csv> <base_dir>")
        sys.exit(1)

    task = sys.argv[1]
    manifest_path = sys.argv[2]
    base_dir = sys.argv[3]

    if task not in TASK_CONFIG:
        print(f"Unknown task '{task}'. Must be 'severity' or 'subtype'.")
        sys.exit(1)

    config = TASK_CONFIG[task]
    out_dir = os.path.join(base_dir, "codes", f"kfold_{task}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(manifest_path)
    df = df[df[config["label_col"]].isin(config["label_map"].keys())].reset_index(drop=True)
    print(f"[{task}] Total images: {len(df)}")
    print(f"Class distribution:\n{df[config['label_col']].value_counts()}")

    mapped = df[config["label_col"]].map(config["label_map"])
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(df, mapped)):
        # further split trainval into train/val (inner stratified split)
        trainval_labels = mapped.iloc[trainval_idx]
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_sub_idx, val_sub_idx = next(inner_skf.split(trainval_idx, trainval_labels))
        train_idx = trainval_idx[train_sub_idx]
        val_idx = trainval_idx[val_sub_idx]

        result = run_one_fold(fold_idx, train_idx, val_idx, test_idx, df, config, device, out_dir)
        fold_results.append(result)

    # sum confusion matrices across all folds before dropping them from the results table
    total_cm = sum(r["confusion_matrix"] for r in fold_results)

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"} for r in fold_results])
    results_df.to_csv(os.path.join(out_dir, "fold_results_summary.csv"), index=False)

    acc_mean, acc_std = results_df["test_acc"].mean(), results_df["test_acc"].std()
    f1_mean, f1_std = results_df["test_macro_f1"].mean(), results_df["test_macro_f1"].std()
    auc_mean, auc_std = results_df["test_auc"].mean(), results_df["test_auc"].std()

    summary = (
        f"K-FOLD CROSS-VALIDATION SUMMARY -- {task.upper()}\n"
        f"{'='*60}\n"
        f"Classes: {config['class_names']}\n\n"
        f"Per-fold results:\n{results_df.to_string(index=False)}\n\n"
        f"Accuracy   : {acc_mean:.4f} +/- {acc_std:.4f}\n"
        f"Macro-F1   : {f1_mean:.4f} +/- {f1_std:.4f}\n"
        f"AUC-ROC    : {auc_mean:.4f} +/- {auc_std:.4f}\n\n"
        f"Aggregated confusion matrix (summed across all 5 folds' test sets):\n"
        f"Rows = true class, Columns = predicted class, order = {config['class_names']}\n"
        f"{total_cm}\n"
    )
    with open(os.path.join(out_dir, "kfold_summary_report.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
