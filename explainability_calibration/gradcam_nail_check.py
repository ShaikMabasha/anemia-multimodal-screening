"""
Grad-CAM sanity check for the nail baseline model.

Purpose: the nail modality scored much higher (92.7% acc) than conjunctiva
and palm (~75% each). Before trusting that number, this script visualizes
WHERE the model is looking when it makes correct predictions -- if
attention concentrates on the nail bed itself, the result is likely
genuine. If attention concentrates on borders, corners, or background,
the model may be exploiting a capture/cropping artifact instead of real
pallor signal.

Usage:
    pip install grad-cam   (this is the 'pytorch-grad-cam' package)
    python gradcam_nail_check.py <manifest_csv> <base_dir>

Output:
    <base_dir>/codes/gradcam_nail_samples.png
        A grid of 10 correctly-classified test images (5 Anemic, 5
        Non-Anemic) with Grad-CAM heatmap overlays.
"""

import os
import sys
import random

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMG_SIZE = 224
LABEL_MAP = {"Anemic": 1, "Non-Anemic": 0}
SEED = 42

random.seed(SEED)


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(ckpt_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    if len(sys.argv) < 3:
        print("Usage: python gradcam_nail_check.py <manifest_csv> <base_dir>")
        sys.exit(1)

    manifest_path = sys.argv[1]
    base_dir = sys.argv[2]
    ckpt_path = os.path.join(base_dir, "codes", "best_nail_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(manifest_path)
    test_df = df[(df["norm_modality"] == "nail") &
                 (df["label"].isin(LABEL_MAP.keys())) &
                 (df["final_split"] == "test")].reset_index(drop=True)
    print(f"Nail test set size: {len(test_df)}")

    model = load_model(ckpt_path, device)
    tf = build_eval_transform()

    # last conv block of ResNet18 -- standard Grad-CAM target layer
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # find correctly-classified examples: 5 Anemic, 5 Non-Anemic
    correct_anemic, correct_non_anemic = [], []
    indices = list(range(len(test_df)))
    random.shuffle(indices)

    for idx in indices:
        if len(correct_anemic) >= 5 and len(correct_non_anemic) >= 5:
            break
        row = test_df.iloc[idx]
        true_label = LABEL_MAP[row["label"]]

        img = Image.open(row["filepath"]).convert("RGB")
        input_tensor = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        if pred != true_label:
            continue  # only want correctly-classified examples for this check

        if true_label == 1 and len(correct_anemic) < 5:
            correct_anemic.append((row["filepath"], img, input_tensor, true_label))
        elif true_label == 0 and len(correct_non_anemic) < 5:
            correct_non_anemic.append((row["filepath"], img, input_tensor, true_label))

    samples = correct_anemic + correct_non_anemic
    if len(samples) < 10:
        print(f"WARNING: only found {len(samples)} correctly-classified examples "
              f"(wanted 5+5). Proceeding with what was found.")

    fig, axes = plt.subplots(2, len(samples), figsize=(3 * len(samples), 6))
    if len(samples) == 1:
        axes = axes.reshape(2, 1)

    for i, (filepath, pil_img, input_tensor, true_label) in enumerate(samples):
        rgb_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

        target = [ClassifierOutputTarget(true_label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)[0]
        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        label_name = "Anemic" if true_label == 1 else "Non-Anemic"

        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"{label_name}\n{os.path.basename(filepath)}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(cam_overlay)
        axes[1, i].set_title("Grad-CAM", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(base_dir, "codes", "gradcam_nail_samples.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved Grad-CAM grid: {out_path}")
    print("\nWhat to look for:")
    print("  GOOD sign: heatmap (red/yellow) concentrated on the nail bed / nail plate itself")
    print("  BAD sign:  heatmap concentrated on image borders, corners, background,")
    print("             or a consistent artifact unrelated to the nail tissue")


if __name__ == "__main__":
    main()
