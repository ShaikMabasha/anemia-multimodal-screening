"""
Generalized single-modality anemia classifier baseline.

Same recipe validated on conjunctiva (dropout + AdamW weight decay + label
smoothing + mixed precision) -- run this once per modality.

Usage:
    python train_baseline_modality.py <manifest_csv> <modality> <base_dir>

Example:
    python train_baseline_modality.py \
        "/home/USERNAME/Anemia/data-audit phase/final_split_manifest_linux.csv" \
        nail \
        "/home/USERNAME/Anemia"

    python train_baseline_modality.py \
        "/home/USERNAME/Anemia/data-audit phase/final_split_manifest_linux.csv" \
        palm \
        "/home/USERNAME/Anemia"

<modality> must be one of: conjunctiva, nail, palm
(must match the norm_modality column values in the manifest)

Output (written into <base_dir>/codes/):
    best_<modality>_model.pt
    training_log_<modality>.csv
    test_report_<modality>.txt
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

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# ---------------- config ----------------
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
PATIENCE = 5
SEED = 42

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
    return avg_loss, acc, f1, auc, all_labels, all_preds, all_probs


def main():
    if len(sys.argv) < 4:
        print("Usage: python train_baseline_modality.py <manifest_csv> <modality> <base_dir>")
        sys.exit(1)

    manifest_path = sys.argv[1]
    modality = sys.argv[2]
    base_dir = sys.argv[3]

    codes_dir = os.path.join(base_dir, "codes")
    os.makedirs(codes_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(manifest_path)
    mod_df = df[(df["norm_modality"] == modality) & (df["label"].isin(LABEL_MAP.keys()))]

    train_df = mod_df[mod_df["final_split"] == "train"]
    val_df = mod_df[mod_df["final_split"] == "val"]
    test_df = mod_df[mod_df["final_split"] == "test"]

    print(f"[{modality}] Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    if len(train_df) == 0:
        print(f"ERROR: no training rows found for norm_modality == '{modality}'. "
              f"Check the manifest's norm_modality column values.")
        sys.exit(1)

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
    class_weights = torch.tensor(
        [total / (2 * n_na), total / (2 * n_a)], dtype=torch.float32
    ).to(device)
    print(f"Class weights [Non-Anemic, Anemic]: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_f1, patience_counter = -1, 0
    log_rows = []
    ckpt_path = os.path.join(codes_dir, f"best_{modality}_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, tr_auc, *_ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc, val_f1, val_auc, *_ = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step(val_f1)
        elapsed = time.time() - t0

        print(f"[{modality}] Epoch {epoch:2d}/{NUM_EPOCHS}  "
              f"train_f1={tr_f1:.4f} | val_f1={val_f1:.4f} val_auc={val_auc:.4f} val_loss={val_loss:.4f}  "
              f"({elapsed:.1f}s)")

        log_rows.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                              val_loss=val_loss, val_acc=val_acc, val_f1=val_f1, val_auc=val_auc))

        if val_f1 > best_val_f1:
            best_val_f1, patience_counter = val_f1, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best (val_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(log_rows).to_csv(os.path.join(codes_dir, f"training_log_{modality}.csv"), index=False)

    # ---- final test evaluation using best checkpoint ----
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc, test_f1, test_auc, y_true, y_pred, y_prob = run_epoch(
        model, test_loader, criterion, optimizer, scaler, device, train=False)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Non-Anemic", "Anemic"])
    report_text = (
        f"BASELINE {modality.upper()} MODEL -- HELD-OUT TEST SET RESULTS\n"
        f"{'='*60}\n"
        f"Test images: {len(test_df)}\n"
        f"Accuracy : {test_acc:.4f}\n"
        f"F1 score : {test_f1:.4f}\n"
        f"AUC-ROC  : {test_auc:.4f}\n\n"
        f"Confusion matrix (rows=true, cols=pred), order [Non-Anemic, Anemic]:\n"
        f"{cm}\n\n"
        f"Classification report:\n{report}\n"
    )
    report_path = os.path.join(codes_dir, f"test_report_{modality}.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"Saved: {ckpt_path}")
    print(f"Saved: training_log_{modality}.csv, {report_path}")


if __name__ == "__main__":
    main()
