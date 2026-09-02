"""
Model efficiency benchmarking script.

Measures, for every model type used in this paper (single-modality
baselines, fusion, severity, subtype):
  - total parameter count
  - on-disk model size (MB)
  - CPU inference latency (single image, batch=1 -- realistic for
    point-of-care / single-patient screening)
  - GPU inference latency, if a GPU is available

This directly addresses the "Model efficiency" item in Limitations,
replacing an acknowledged gap with actual measured numbers.

Usage:
    python benchmark_model_efficiency.py <base_dir>

Output:
    model_efficiency_report.txt
    model_efficiency_table.csv
"""

import os
import sys
import time

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = 224
EMBED_DIM = 256
N_WARMUP = 10
N_TIMED_RUNS = 100


def build_binary_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))
    return model


def build_severity_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 4))
    return model


def build_subtype_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 2))
    return model


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
        attn_out, _ = self.attn(tokens, tokens, tokens)
        fused = self.norm(tokens + attn_out)
        pooled = fused.mean(dim=1)
        return self.classifier(pooled)


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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model):
    tmp_path = "_tmp_model_size_check.pt"
    torch.save(model.state_dict(), tmp_path)
    size_bytes = os.path.getsize(tmp_path)
    os.remove(tmp_path)
    return size_bytes / (1024 ** 2)


def benchmark_latency(forward_fn, device, n_warmup=N_WARMUP, n_runs=N_TIMED_RUNS):
    for _ in range(n_warmup):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def benchmark_single_input_model(model, device):
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    def forward():
        with torch.no_grad():
            model(dummy_input)

    return benchmark_latency(forward, device)


def benchmark_fusion_model(model, device):
    model = model.to(device)
    model.eval()
    dummy_conj = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    dummy_nail = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    dummy_palm = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    def forward():
        with torch.no_grad():
            model(dummy_conj, dummy_nail, dummy_palm)

    return benchmark_latency(forward, device)


def main():
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda") if torch.cuda.is_available() else None
    print("CPU benchmarking: yes")
    print(f"GPU benchmarking: {'yes (' + torch.cuda.get_device_name(0) + ')' if gpu_device else 'no GPU available'}")

    model_specs = [
        ("Conjunctiva baseline", build_binary_resnet18, False),
        ("Nail baseline", build_binary_resnet18, False),
        ("Palm baseline", build_binary_resnet18, False),
        ("Severity (4-class)", build_severity_resnet18, False),
        ("Subtype (binary)", build_subtype_resnet18, False),
        ("Fusion (3-modality)", FullFusionModel, True),
    ]

    rows = []
    for name, builder, is_fusion in model_specs:
        print(f"\nBenchmarking: {name}")
        model = builder()
        total_params, trainable_params = count_parameters(model)
        size_mb = get_model_size_mb(model)

        bench_fn = benchmark_fusion_model if is_fusion else benchmark_single_input_model
        cpu_mean, cpu_std = bench_fn(model, cpu_device)

        gpu_mean, gpu_std = (None, None)
        if gpu_device is not None:
            model_gpu = builder()
            gpu_mean, gpu_std = bench_fn(model_gpu, gpu_device)

        rows.append({
            "model": name,
            "total_params": total_params,
            "total_params_M": total_params / 1e6,
            "model_size_MB": size_mb,
            "cpu_latency_ms_mean": cpu_mean,
            "cpu_latency_ms_std": cpu_std,
            "gpu_latency_ms_mean": gpu_mean,
            "gpu_latency_ms_std": gpu_std,
        })

        print(f"  Params: {total_params/1e6:.2f}M   Size: {size_mb:.2f} MB   "
              f"CPU: {cpu_mean:.2f}+-{cpu_std:.2f} ms" +
              (f"   GPU: {gpu_mean:.2f}+-{gpu_std:.2f} ms" if gpu_mean is not None else ""))

    results_df = pd.DataFrame(rows)
    results_df.to_csv("model_efficiency_table.csv", index=False)

    lines = []
    lines.append("MODEL EFFICIENCY BENCHMARK")
    lines.append("=" * 70)
    lines.append("Batch size: 1 (single-image inference, point-of-care scenario)")
    lines.append(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    lines.append(f"Timed runs per model: {N_TIMED_RUNS} (after {N_WARMUP} warmup runs)")
    lines.append("")
    lines.append(f"{'Model':<22}{'Params (M)':>12}{'Size (MB)':>12}{'CPU (ms)':>14}{'GPU (ms)':>14}")
    lines.append("-" * 74)
    for r in rows:
        gpu_str = f"{r['gpu_latency_ms_mean']:.2f}" if r['gpu_latency_ms_mean'] is not None else "N/A"
        lines.append(f"{r['model']:<22}{r['total_params_M']:>12.2f}{r['model_size_MB']:>12.2f}"
                      f"{r['cpu_latency_ms_mean']:>14.2f}{gpu_str:>14}")

    report_text = "\n".join(lines)
    with open("model_efficiency_report.txt", "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print("\nSaved: model_efficiency_report.txt, model_efficiency_table.csv")


if __name__ == "__main__":
    main()
