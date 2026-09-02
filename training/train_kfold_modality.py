"""
K-fold single-modality baseline training.

For each of the 5 folds: trains on 3 folds, validates (for early stopping)
on 1 fold, tests on the remaining held-out fold. Reports per-fold results
and the mean +/- std across all 5 folds -- THIS aggregated number, not any
single fold, is what should be reported in the paper.

Usage:
    python train_kfold_modality.py <kfold_manifest_csv> <modality> <base_dir>

Output (into <base_dir>/codes/kfold_<modality>/):
    fold{0..4}_model.pt
    fold{0..4}_training_log.csv
    fold_results_summary.csv       -- per-fold metrics
    kfold_summary_report.txt       -- mean +/- std, the headline result
"""

import os
import sys
import time

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
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
PATIENCE = 5
SEED = 42
NUM_FOLDS = 5

LABEL_MAP = {"Anemic": 1, "Non-Anemic": 0}

torch.manual_seed(SEED)
np.random.seed(SEED)


class ImageLabelDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        return self.transform(img), LABEL_MAP[row["label"]]


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


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
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
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return avg_loss, acc, f1, auc


def run_one_fold(fold_idx, mod_df, device, out_dir):
    test_df = mod_df[mod_df["fold"] == fold_idx]
    val_fold = (fold_idx + 1) % NUM_FOLDS
    val_df = mod_df[mod_df["fold"] == val_fold]
    train_df = mod_df[~mod_df["fold"].isin([fold_idx, val_fold])]

    print(f"\n--- Fold {fold_idx} --- (test=fold{fold_idx}, val=fold{val_fold}, "
          f"train=remaining 3 folds)")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_tf, eval_tf = build_transforms()
    num_workers = 0 if os.name == "nt" else 8

    train_loader = DataLoader(ImageLabelDataset(train_df, train_tf), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ImageLabelDataset(val_df, eval_tf), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ImageLabelDataset(test_df, eval_tf), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))
    model = model.to(device)

    label_counts = train_df["label"].value_counts()
    n_a, n_na = label_counts.get("Anemic", 1), label_counts.get("Non-Anemic", 1)
    total = n_a + n_na
    class_weights = torch.tensor([total / (2 * n_na), total / (2 * n_a)], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_f1, patience_counter = -1, 0
    log_rows = []
    ckpt_path = os.path.join(out_dir, f"fold{fold_idx}_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc, val_f1, val_auc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step(val_f1)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:2d}: train_f1={tr_f1:.4f} | val_f1={val_f1:.4f} val_auc={val_auc:.4f} ({elapsed:.1f}s)")
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
    test_loss, test_acc, test_f1, test_auc = run_epoch(model, test_loader, criterion, optimizer, scaler, device, train=False)
    print(f"  FOLD {fold_idx} TEST: acc={test_acc:.4f} f1={test_f1:.4f} auc={test_auc:.4f}")

    return dict(fold=fold_idx, test_images=len(test_df), test_acc=test_acc,
                test_f1=test_f1, test_auc=test_auc)


def main():
    if len(sys.argv) < 4:
        print("Usage: python train_kfold_modality.py <kfold_manifest_csv> <modality> <base_dir>")
        sys.exit(1)

    manifest_path = sys.argv[1]
    modality = sys.argv[2]
    base_dir = sys.argv[3]
    out_dir = os.path.join(base_dir, "codes", f"kfold_{modality}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(manifest_path)
    mod_df = df[(df["norm_modality"] == modality) & (df["label"].isin(LABEL_MAP.keys()))]
    print(f"[{modality}] Total images across all folds: {len(mod_df)}")

    fold_results = []
    for fold_idx in range(NUM_FOLDS):
        result = run_one_fold(fold_idx, mod_df, device, out_dir)
        fold_results.append(result)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(out_dir, "fold_results_summary.csv"), index=False)

    acc_mean, acc_std = results_df["test_acc"].mean(), results_df["test_acc"].std()
    f1_mean, f1_std = results_df["test_f1"].mean(), results_df["test_f1"].std()
    auc_mean, auc_std = results_df["test_auc"].mean(), results_df["test_auc"].std()

    summary = (
        f"K-FOLD CROSS-VALIDATION SUMMARY -- {modality.upper()}\n"
        f"{'='*60}\n"
        f"Folds: {NUM_FOLDS}\n\n"
        f"Per-fold results:\n{results_df.to_string(index=False)}\n\n"
        f"Accuracy : {acc_mean:.4f} +/- {acc_std:.4f}\n"
        f"F1 score : {f1_mean:.4f} +/- {f1_std:.4f}\n"
        f"AUC-ROC  : {auc_mean:.4f} +/- {auc_std:.4f}\n"
    )
    with open(os.path.join(out_dir, "kfold_summary_report.txt"), "w") as f:
        f.write(summary)

    print("\n" + summary)


if __name__ == "__main__":
    main()
