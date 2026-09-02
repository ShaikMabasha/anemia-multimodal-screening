"""
Fusion model explainability (v2 -- styled, publication-ready).

Same real computation as explain_fusion_model.py: genuine Grad-CAM on the
trained fusion model, genuine cross-attention weights. This version adds:
  - a real colorbar for the Grad-CAM heatmap intensity
  - exact attention weight values printed AND saved to CSV (so captions/
    tables can cite verified numbers, not visual estimates)
  - cleaner typography, gridlines, value labels on the attention bar chart
  - an optional COMBINED two-patient figure (side-by-side subfigure style)

Usage:
    pip install grad-cam
    python explain_fusion_model_v2.py <fusion_manifest_csv> <base_dir> <fold_idx>

Output (into <base_dir>/codes/explainability/):
    patient_<key>_explain.png   (one per sampled patient, styled)
    combined_two_patient_figure.png   (one Anemic + one Non-Anemic, side by side)
    attention_weights.csv       (exact per-patient, per-modality attention values)
"""

import os
import sys
import random

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMG_SIZE = 224
EMBED_DIM = 256
LABEL_MAP = {"Anemic": 1, "Non-Anemic": 0}
SEED = 42
N_PATIENTS_PER_CLASS = 3

random.seed(SEED)

mpl.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
})

MODALITY_COLORS = {"conjunctiva": "#4C72B0", "nail": "#55A868", "palm": "#DD8452"}


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
        attn_out, attn_weights = self.attn(tokens, tokens, tokens, average_attn_weights=True)
        fused = self.norm(tokens + attn_out)
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, attn_weights


class FullFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conj = self._build_resnet_encoder()
        self.enc_nail = self._build_resnet_encoder()
        self.enc_palm = self._build_resnet_encoder()
        self.fusion = CrossAttentionFusion()

    def _build_resnet_encoder(self):
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))
        model.fc = nn.Identity()
        return model

    def forward(self, img_conj, img_nail, img_palm):
        f_conj = self._features(self.enc_conj, img_conj)
        f_nail = self._features(self.enc_nail, img_nail)
        f_palm = self._features(self.enc_palm, img_palm)
        return self.fusion(f_conj, f_nail, f_palm)

    @staticmethod
    def _features(resnet, x):
        x = resnet.conv1(x); x = resnet.bn1(x); x = resnet.relu(x); x = resnet.maxpool(x)
        x = resnet.layer1(x); x = resnet.layer2(x); x = resnet.layer3(x); x = resnet.layer4(x)
        x = resnet.avgpool(x)
        return torch.flatten(x, 1)


def build_patient_dicts(df):
    patient_to_paths = {}
    for patient_key, group in df.groupby("patient_key"):
        entry = {"label": group["label"].iloc[0]}
        for modality in ["conjunctiva", "nail", "palm"]:
            entry[modality] = group[group["norm_modality"] == modality]["filepath"].tolist()
        patient_to_paths[patient_key] = entry
    return patient_to_paths


class SingleModalityWrapper(nn.Module):
    def __init__(self, full_model, modality, fixed_feats):
        super().__init__()
        self.full_model = full_model
        self.modality = modality
        self.fixed_feats = fixed_feats

    def forward(self, x):
        feat = self.full_model._features(getattr(self.full_model, f"enc_{self.modality[:4]}"), x)
        feats = dict(self.fixed_feats)
        feats[self.modality] = feat
        logits, _ = self.full_model.fusion(feats["conjunctiva"], feats["nail"], feats["palm"])
        return logits


def compute_patient_gradcam(model, imgs_pil, imgs_tensor, label_val, device):
    with torch.no_grad():
        feats = {
            "conjunctiva": model._features(model.enc_conj, imgs_tensor["conjunctiva"]),
            "nail": model._features(model.enc_nail, imgs_tensor["nail"]),
            "palm": model._features(model.enc_palm, imgs_tensor["palm"]),
        }

    results = {}
    for modality in ["conjunctiva", "nail", "palm"]:
        rgb_img = np.array(imgs_pil[modality].resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
        wrapper = SingleModalityWrapper(model, modality, feats).to(device)
        wrapper.eval()
        encoder = getattr(model, f"enc_{modality[:4]}")
        target_layers = [encoder.layer4[-1]]
        cam = GradCAM(model=wrapper, target_layers=target_layers)
        target = [ClassifierOutputTarget(label_val)]
        grayscale_cam = cam(input_tensor=imgs_tensor[modality], targets=target)[0]
        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        results[modality] = (rgb_img, cam_overlay, grayscale_cam)
    return results


def render_patient_panel(fig, gs_slice, patient_key, label_name, gradcam_results, attn_weights, panel_label=None):
    inner_gs = gs_slice.subgridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1.1],
                                     hspace=0.15, wspace=0.25)

    modalities = ["conjunctiva", "nail", "palm"]
    im_handle = None
    for col, modality in enumerate(modalities):
        rgb_img, cam_overlay, _ = gradcam_results[modality]

        ax_top = fig.add_subplot(inner_gs[0, col])
        ax_top.imshow(rgb_img)
        ax_top.set_title(modality.capitalize(), fontsize=11, fontweight="bold")
        ax_top.axis("off")

        ax_bot = fig.add_subplot(inner_gs[1, col])
        im_handle = ax_bot.imshow(cam_overlay)
        ax_bot.axis("off")

    ax_bar = fig.add_subplot(inner_gs[:, 3])
    mods = ["conjunctiva", "nail", "palm"]
    vals = [attn_weights[m] for m in mods]
    colors = [MODALITY_COLORS[m] for m in mods]
    bars = ax_bar.bar([m.capitalize() for m in mods], vals, color=colors, edgecolor="black", linewidth=0.8)
    ax_bar.set_ylim(0, 0.5)
    ax_bar.set_ylabel("Attention weight", fontsize=10)
    ax_bar.set_title("Cross-attention weights", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bar.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}",
                     ha="center", fontsize=9, fontweight="bold")

    title = f"Patient {patient_key} \u2013 correctly classified {label_name}"
    if panel_label:
        title = f"({panel_label}) {title}"
    return im_handle, title


def main():
    if len(sys.argv) < 4:
        print("Usage: python explain_fusion_model_v2.py <fusion_manifest_csv> <base_dir> <fold_idx>")
        sys.exit(1)

    manifest_path = sys.argv[1]
    base_dir = sys.argv[2]
    fold_idx = int(sys.argv[3])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = os.path.join(base_dir, "codes", "kfold_fusion", f"fold{fold_idx}_model.pt")
    model = FullFusionModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    df = pd.read_csv(manifest_path)
    modality_sets = {m: set(df[df["norm_modality"] == m]["patient_key"]) for m in ["conjunctiva", "nail", "palm"]}
    matched_patients = modality_sets["conjunctiva"] & modality_sets["nail"] & modality_sets["palm"]
    print(f"Patients matched across all 3 modalities: {len(matched_patients)}")

    df = df[df["patient_key"].isin(matched_patients)]
    test_df = df[df["fold"] == fold_idx] if "fold" in df.columns else df[df["split"] == "test"]
    patient_dict = build_patient_dicts(test_df)
    patients = list(patient_dict.keys())

    tf = build_eval_transform()
    out_dir = os.path.join(base_dir, "codes", "explainability")
    os.makedirs(out_dir, exist_ok=True)

    random.shuffle(patients)
    chosen = {0: [], 1: []}
    attention_records = []

    for p in patients:
        if len(chosen[0]) >= N_PATIENTS_PER_CLASS and len(chosen[1]) >= N_PATIENTS_PER_CLASS:
            break
        info = patient_dict[p]
        true_label = LABEL_MAP[info["label"]]

        imgs_pil, imgs_tensor = {}, {}
        for modality in ["conjunctiva", "nail", "palm"]:
            path = info[modality][0]
            img = Image.open(path).convert("RGB")
            imgs_pil[modality] = img
            imgs_tensor[modality] = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, attn_weights = model(imgs_tensor["conjunctiva"], imgs_tensor["nail"], imgs_tensor["palm"])
            pred = torch.argmax(logits, dim=1).item()

        if pred != true_label or len(chosen[true_label]) >= N_PATIENTS_PER_CLASS:
            continue

        attn = attn_weights.cpu().numpy()[0].mean(axis=0)
        attn_dict = {"conjunctiva": float(attn[0]), "nail": float(attn[1]), "palm": float(attn[2])}
        chosen[true_label].append((p, info, imgs_pil, imgs_tensor, attn_dict))

        attention_records.append({
            "patient_key": p, "true_label": info["label"],
            "attn_conjunctiva": attn_dict["conjunctiva"],
            "attn_nail": attn_dict["nail"],
            "attn_palm": attn_dict["palm"],
        })

    print(f"Selected {len(chosen[0])} Non-Anemic + {len(chosen[1])} Anemic correctly-classified patients")

    attn_df = pd.DataFrame(attention_records)
    attn_csv_path = os.path.join(out_dir, "attention_weights.csv")
    attn_df.to_csv(attn_csv_path, index=False)
    print(f"\nExact attention weights saved to: {attn_csv_path}")
    print(attn_df.to_string(index=False))

    all_patients_data = []
    for label_val, patient_list in chosen.items():
        label_name = "Anemic" if label_val == 1 else "Non-Anemic"
        for patient_key, info, imgs_pil, imgs_tensor, attn_dict in patient_list:
            gradcam_results = compute_patient_gradcam(model, imgs_pil, imgs_tensor, label_val, device)
            all_patients_data.append((patient_key, label_name, label_val, gradcam_results, attn_dict))

            fig = plt.figure(figsize=(14, 6.5))
            gs = fig.add_gridspec(1, 1)
            im_handle, title = render_patient_panel(fig, gs[0], patient_key, label_name, gradcam_results, attn_dict)
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

            cbar = fig.colorbar(im_handle, ax=fig.get_axes()[:6], location="bottom",
                                 fraction=0.05, pad=0.08, aspect=40)
            cbar.set_label("Grad-CAM activation (low \u2192 high)", fontsize=10)
            cbar.set_ticks([])

            safe_key = patient_key.replace("/", "_")
            out_path = os.path.join(out_dir, f"patient_{safe_key}_explain.png")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

    if chosen[1] and chosen[0]:
        anemic_data = next(d for d in all_patients_data if d[2] == 1)
        nonanemic_data = next(d for d in all_patients_data if d[2] == 0)

        fig = plt.figure(figsize=(16, 12))
        outer_gs = fig.add_gridspec(2, 1, hspace=0.35)

        for row, (patient_key, label_name, label_val, gradcam_results, attn_dict) in enumerate(
                [anemic_data, nonanemic_data]):
            panel_label = "a" if row == 0 else "b"
            im_handle, title = render_patient_panel(fig, outer_gs[row], patient_key, label_name,
                                                      gradcam_results, attn_dict, panel_label=panel_label)
            fig.text(0.02, 0.98 - row * 0.5,
                     f"({panel_label}) Patient {patient_key} \u2013 correctly classified {label_name}",
                     fontsize=13, fontweight="bold")

        combined_path = os.path.join(out_dir, "combined_two_patient_figure.png")
        plt.savefig(combined_path, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved combined figure: {combined_path}")

    print(f"\nAll explainability figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
