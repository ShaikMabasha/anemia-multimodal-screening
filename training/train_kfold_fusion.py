"""
K-fold multimodal fusion training.

Critical detail: fold i's fusion model warm-starts from fold i's OWN
baseline checkpoints (codes/kfold_conjunctiva/fold{i}_model.pt, etc.) --
NOT from a single shared baseline checkpoint. This preserves the leakage
fix: fold i's baseline was trained excluding fold i's patients, so
warm-starting fold i's fusion model from it is safe.

Usage:
    python train_kfold_fusion.py <kfold_manifest_csv> <base_dir>

Output (into <base_dir>/codes/kfold_fusion/):
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

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 25
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
PATIENCE = 6
SEED = 42
EMBED_DIM = 256
NUM_FOLDS = 5

LABEL_MAP = {"Anemic": 1, "Non-Anemic": 0}

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


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
            path = random.choice(info[modality])
            img = Image.open(path).convert("RGB")
            imgs[modality] = self.transform(img)
        label = LABEL_MAP[info["label"]]
        return imgs["conjunctiva"], imgs["nail"], imgs["palm"], label


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


def build_encoder(checkpoint_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))
    if checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {checkpoint_path}\n"
            f"Run train_kfold_modality.py for all 3 modalities BEFORE fusion."
        )
    model.fc = nn.Identity()
    return model.to(device)


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=4, dropout=0.3):
        super().__init__()
        self.proj_conj = nn.Linear(512, embed_dim)
        self.proj_nail = nn.Linear(512, embed_dim)
        self.proj_palm = nn.Linear(512, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )

    def forward(self, feat_conj, feat_nail, feat_palm):
        tok_conj = self.proj_conj(feat_conj).unsqueeze(1)
        tok_nail = self.proj_nail(feat_nail).unsqueeze(1)
        tok_palm = self.proj_palm(feat_palm).unsqueeze(1)
        tokens = torch.cat([tok_conj, tok_nail, tok_palm], dim=1)
        attn_out, attn_weights = self.attn(tokens, tokens, tokens)
        fused = self.norm(tokens + attn_out)
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, attn_weights


class FullFusionModel(nn.Module):
    def __init__(self, enc_conj, enc_nail, enc_palm):
        super().__init__()
        self.enc_conj = enc_conj
        self.enc_nail = enc_nail
        self.enc_palm = enc_palm
        self.fusion = CrossAttentionFusion()

    def forward(self, img_conj, img_nail, img_palm):
        f_conj = self.enc_conj(img_conj)
        f_nail = self.enc_nail(img_nail)
        f_palm = self.enc_palm(img_palm)
        return self.fusion(f_conj, f_nail, f_palm)


def build_patient_dicts(df):
    patient_to_paths = {}
    for patient_key, group in df.groupby("patient_key"):
        entry = {"label": group["label"].iloc[0]}
        for modality in ["conjunctiva", "nail", "palm"]:
            entry[modality] = group[group["norm_modality"] == modality]["filepath"].tolist()
        patient_to_paths[patient_key] = entry
    return patient_to_paths


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.set_grad_enabled(train):
        for img_conj, img_nail, img_palm, labels in tqdm(loader, leave=False):
            img_conj = img_conj.to(device, non_blocking=True)
            img_nail = img_nail.to(device, non_blocking=True)
            img_palm = img_palm.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model(img_conj, img_nail, img_palm)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
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


def run_one_fold(fold_idx, fusion_df, codes_dir, out_dir, device):
    test_df = fusion_df[fusion_df["fold"] == fold_idx]
    val_fold = (fold_idx + 1) % NUM_FOLDS
    val_df = fusion_df[fusion_df["fold"] == val_fold]
    train_df = fusion_df[~fusion_df["fold"].isin([fold_idx, val_fold])]

    train_dict = build_patient_dicts(train_df)
    val_dict = build_patient_dicts(val_df)
    test_dict = build_patient_dicts(test_df)
    train_patients, val_patients, test_patients = list(train_dict), list(val_dict), list(test_dict)

    print(f"\n--- Fusion Fold {fold_idx} --- Patients -> train:{len(train_patients)} "
          f"val:{len(val_patients)} test:{len(test_patients)}")

    train_tf, eval_tf = build_transforms()
    num_workers = 0 if os.name == "nt" else 6

    train_loader = DataLoader(FusionDataset(train_dict, train_patients, train_tf), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(FusionDataset(val_dict, val_patients, eval_tf), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(FusionDataset(test_dict, test_patients, eval_tf), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    # CRITICAL: warm-start from the SAME fold's baseline checkpoints
    enc_conj = build_encoder(os.path.join(codes_dir, "kfold_conjunctiva", f"fold{fold_idx}_model.pt"), device)
    enc_nail = build_encoder(os.path.join(codes_dir, "kfold_nail", f"fold{fold_idx}_model.pt"), device)
    enc_palm = build_encoder(os.path.join(codes_dir, "kfold_palm", f"fold{fold_idx}_model.pt"), device)
    model = FullFusionModel(enc_conj, enc_nail, enc_palm).to(device)

    label_series = pd.Series([train_dict[p]["label"] for p in train_patients])
    counts = label_series.value_counts()
    n_a, n_na = counts.get("Anemic", 1), counts.get("Non-Anemic", 1)
    total = n_a + n_na
    class_weights = torch.tensor([total / (2 * n_na), total / (2 * n_a)], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_f1, patience_counter = -1, 0
    log_rows = []
    ckpt_path = os.path.join(out_dir, f"fold{fold_idx}_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, tr_auc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc, val_f1, val_auc = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
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
    test_loss, test_acc, test_f1, test_auc = run_epoch(model, test_loader, criterion, optimizer, scaler, device, train=False)
    print(f"  FUSION FOLD {fold_idx} TEST: acc={test_acc:.4f} f1={test_f1:.4f} auc={test_auc:.4f}")

    return dict(fold=fold_idx, test_patients=len(test_patients), test_acc=test_acc,
                test_f1=test_f1, test_auc=test_auc)


def main():
    if len(sys.argv) < 3:
        print("Usage: python train_kfold_fusion.py <kfold_manifest_csv> <base_dir>")
        sys.exit(1)

    manifest_path = sys.argv[1]
    base_dir = sys.argv[2]
    codes_dir = os.path.join(base_dir, "codes")
    out_dir = os.path.join(codes_dir, "kfold_fusion")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(manifest_path)
    modality_sets = {m: set(df[df["norm_modality"] == m]["patient_key"]) for m in ["conjunctiva", "nail", "palm"]}
    matched_patients = modality_sets["conjunctiva"] & modality_sets["nail"] & modality_sets["palm"]
    print(f"Patients matched across all 3 modalities: {len(matched_patients)}")

    fusion_df = df[df["patient_key"].isin(matched_patients)].copy()

    fold_results = []
    for fold_idx in range(NUM_FOLDS):
        result = run_one_fold(fold_idx, fusion_df, codes_dir, out_dir, device)
        fold_results.append(result)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(out_dir, "fold_results_summary.csv"), index=False)

    acc_mean, acc_std = results_df["test_acc"].mean(), results_df["test_acc"].std()
    f1_mean, f1_std = results_df["test_f1"].mean(), results_df["test_f1"].std()
    auc_mean, auc_std = results_df["test_auc"].mean(), results_df["test_auc"].std()

    summary = (
        f"K-FOLD CROSS-VALIDATION SUMMARY -- FUSION\n"
        f"{'='*60}\n"
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
