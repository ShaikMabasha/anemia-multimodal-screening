# Multimodal Anemia Screening — Project Reference

This document consolidates the problem statement, novelty positioning, dataset audit
findings, and final label/split schema developed for this project. Keep it alongside
the code repo as the source of truth for dataset decisions.

---

## 1. Problem Statement

Despite more than a decade of AI research on anemia detection (RBC microscopy,
CBC-based classification, and non-invasive imaging of the conjunctiva, nail bed, and
palm), no existing system has been shown to reliably generalize outside the exact
conditions it was trained on. Three failures recur across the literature:

1. Models are validated on a single device/site/population and never re-tested elsewhere.
2. Non-invasive image-based systems classify only *anemic vs. non-anemic* — never
   severity or probable subtype, which is what actually changes clinical management.
3. Reported "explainability" is a post-hoc visualization, not a calibrated statement
   of how much the model's prediction should be trusted.

**Research problem**: There is no validated framework for non-invasive, multimodal
anemia screening that simultaneously (a) grades severity and estimates likely subtype,
(b) is proven robust across imaging devices, lighting, and skin tones, and (c) reports
calibrated, clinically interpretable confidence alongside every prediction.

## 2. Novelty Statement

Prior closest work: **BPANet** (Zhang et al., 2024, *J. Imaging Informatics in
Medicine*) — cross-attention fusion of conjunctiva+palm+nail, binary/regression only,
no calibration, no fairness audit. **AnemiaVision** (2026 preprint) lists similar
fusion + Grad-CAM as future work.

What remains unclaimed and forms this project's contribution:
- **Severity + subtype grading** from non-invasive multimodal images (not just binary)
- **Explicit cross-device / cross-skin-tone domain-shift evaluation** as a first-class
  experimental axis
- **Calibrated uncertainty** (conformal prediction / ECE) — absent from all reviewed
  prior work
- **Documented, leakage-safe evaluation methodology** — see Section 4; this alone is a
  defensible, citable contribution given what was found in the public data (below)

## 3. Proposed Architecture (summary)

Conjunctiva + Nail + Palm images (+ demographics) → modality encoders → cross-attention
fusion module → two task heads (severity grading, subtype classification) → trust/
calibration layer (conformal uncertainty + Grad-CAM/SHAP validated against clinician
reasoning) → external multi-site validation (cross-device, cross-skin-tone).

## 4. Dataset Inventory & Audit — Key Findings

### 4.1 Sources

| Source | Images | Role |
|---|---|---|
| New_Augmented_Anemia_Dataset (Conjunctiva/Finger_Nails/Palm) | 30,917 | Primary binary training pool |
| Fingernails (raw) | 4,260 | Merged into nail subject pool (same subjects as augmented set) |
| Palm (raw) | 4,260 | Merged into palm subject pool |
| CP-AnemiC | 710 (424 Anemic / 286 Non-anemic) | **Held out** — external cross-domain test set + severity ground truth |
| AneRBC-I | 1,000 (500 Anemic / 500 Healthy) | Binary label + auxiliary CBC tabular features + subtype |
| AneRBC-II | 12,000 (crops of AneRBC-I's same 500+500 patients) | Pretraining pool only — **not used for evaluation** (same patients as AneRBC-I) |
| conjuctiva dataset (Roboflow, 83 image+mask pairs) | 83 | Segmentation/ROI-cropping preprocessing only — not a classification source |
| CBC_data_for_meandeley_csv.csv (364 patients) | — | Independent standalone tabular CBC set — not linked to AneRBC images |

### 4.2 Critical issues found and resolved

1. **Subject-level leakage in New_Augmented_Anemia_Dataset's provided
   Training/Testing/Validation split.** Audit found **98.0% of subjects (1,553/1,585)
   appear in more than one split** (up to 100% in some modality/class groups). The
   provided split must never be used as-is. **Fix**: rebuilt a 70/15/15 split at the
   subject level (after normalizing naming across raw and augmented sources), verified
   zero leakage.

2. **AneRBC-I / AneRBC-II are the same patients.** AneRBC-II's 12,000 images are crops
   of AneRBC-I's 500+500 patients (12 crops each, confirmed via readme + identical CBC
   reports). **Fix**: AneRBC-II restricted to pretraining use only, never mixed into an
   evaluation set with AneRBC-I.

3. **AneRBC folder label ≠ simple Hb threshold.** After parsing all 1,000 AneRBC-I CBC
   reports and cross-checking internal lab consistency (HCT ≈ RBC×MCV/10 to exclude
   corrupted reports, 46 excluded), **~38–44% of patients' Hb values did not match
   their folder-assigned Anemic/Healthy label** in either direction. This is not
   parsing error or data corruption — it reflects that the original label is a
   holistic hematologist diagnosis, not a single-value cutoff. **Fix**: AneRBC's
   provided binary label is used as-is (never overridden); the CBC panel is used as
   **auxiliary tabular input features**, not as a re-derived label source.

4. **conjuctiva dataset (Roboflow export) is a segmentation dataset**, not a
   classification source (image+mask pairs, no anemic/non-anemic labels). Used only
   for conjunctiva ROI-cropping preprocessing.

5. **CBC_data_for_meandeley_csv.csv is an independent tabular dataset**, not linked to
   AneRBC's images (AneRBC's real per-patient values live in its own `.txt` reports).
   Usable as a standalone structured-data baseline comparison, not for image fusion.

## 5. Final Label Schema

| Task | Ground truth source(s) | Notes |
|---|---|---|
| **Primary: Binary Anemic/Non-Anemic** | Each source's own provided label (folder-based for AneRBC/CP-AnemiC/New_Augmented) | Never re-derived from Hb |
| **Severity (Non-Anemic/Mild/Moderate/Severe)** | CP-AnemiC categorical severity (primary); AneRBC CBC panel as auxiliary tabular features only | CP-AnemiC is pediatric (6–60mo, Ghana); scope this explicitly in the paper |
| **Subtype (Microcytic/Normocytic/Macrocytic)** | MCV-based classification on AneRBC's Anemic-labeled cohort | Standard clinical cutoffs: <80fL / 80–100fL / >100fL |
| **External validation** | CP-AnemiC's 10 hospitals / 4 regions | Leave-one-hospital-out testing for domain generalization claim |

## 6. Final Split Strategy

- Subject-level (not image-level) 70/15/15 train/val/test, stratified within each
  (modality, label) group, verified zero cross-split subject leakage.
- Modality naming normalized (`Fingernail`/`Finger_Nails`/`Fingernails` → `nail`, etc.)
  so raw and augmented images of the same subject share one identity.
- CP-AnemiC and AneRBC-II held out entirely from this split (external test / pretraining
  pool respectively — see Section 4.2).

| Modality | Total | Train | Val | Test |
|---|---|---|---|---|
| Conjunctiva | 10,256 | 7,111 | 1,633 | 1,512 |
| Nail | 14,647 (–44 unresolved labels excluded) | 10,370 | 2,206 | 2,071 |
| Palm | 14,534 | 10,244 | 2,125 | 2,165 |
| **Total** | **39,437** | **27,725** | **5,964** | **5,748** |

## 7. Target Journals (Q1)

- BioData Mining (Springer)
- Computers in Biology and Medicine (Elsevier)
- Scientific Reports (Nature)
- IEEE Access

## 8. Useful Code References

See `anemia_code_refs.zip` (cloned repos): conjunctiva CNN pipeline, conjunctiva RF+SMOTE
baseline, conjunctiva Hb regression, nail-bed Hb stats. Treat as scaffolding, not
validated methodology.

Tooling: `MAPIE` (conformal prediction), `pytorch-grad-cam` / `shap` (explainability),
`netcal` (calibration metrics/ECE).

## 9. Baseline & Fusion Results (5-fold cross-validation, patient-level, leakage-audited)

Two leakage issues were found and fixed during model development, both documented here
since they're citable methodology points:
1. Initial per-modality splits were computed independently, so a patient could be in
   one modality's train set and another's test set. Fixed with a unified global
   patient-level split (by (label, bare_code) identity, union across modalities).
2. Warm-starting fusion encoders from single-split baseline checkpoints caused
   pretraining leakage (fusion "test" patients had already been seen during baseline
   training) — surfaced as an implausible val_f1=1.0 in epoch 1. Fixed by moving to
   5-fold cross-validation where each fold's fusion model warm-starts only from that
   same fold's baseline checkpoints (trained excluding that fold's patients).

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Conjunctiva | 0.720 ± 0.028 | 0.747 ± 0.031 | 0.781 ± 0.026 |
| Nail | 0.701 ± 0.012 | 0.739 ± 0.020 | 0.764 ± 0.016 |
| Palm | 0.790 ± 0.027 | 0.812 ± 0.022 | 0.848 ± 0.035 |
| **Fusion (cross-attention)** | **0.812 ± 0.045** | **0.843 ± 0.036** | **0.887 ± 0.038** |

Fusion outperforms every single modality on every metric, on average, across 5 folds.
Std is larger for fusion (smaller patient count: 443 matched patients vs ~450-530 per
single modality) — worth a formal paired significance test (Wilcoxon signed-rank or
McNemar's) before final submission to strengthen the claim beyond "consistent average
improvement."

## 11. Explainability & Calibration Results

**Grad-CAM (per-modality, on fusion model)**: Across all sampled correctly-classified
test patients (both classes), heatmaps consistently concentrate on genuine physiological
tissue (conjunctiva strip, nail plate, palm skin) rather than background or crop
borders, across all three modalities. Supports trustworthiness of the learned features.

**Cross-attention weights**: Stay close to uniform (~0.30-0.38 per modality) across
sampled patients, showing limited evidence of dynamic, case-specific modality
prioritization — reported honestly as a limitation rather than oversold as "learned
modality trust." Worth flagging as future work (e.g. an attention-entropy regularizer).

**Calibration (split conformal prediction, target 90% coverage)**:
- Empirical coverage: 91.9% ± 3.1% (close to nominal target)
- Avg prediction-set size: 1.38 (model appropriately flags genuine uncertainty on
  ~38% of cases rather than being falsely overconfident)
- ECE: 0.075 (pooled across all folds' test predictions — the more statistically
  stable estimate; per-fold ECE averaged 0.121 ± 0.067, noisier due to smaller
  per-fold sample sizes)

## 12. Status: All Planned Pillars Complete

| Component | Result |
|---|---|
| Binary fusion (leakage-audited, k-fold CV) | Acc 0.812, F1 0.843, AUC 0.887 |
| Severity grading (CP-AnemiC, image-only) | Acc 0.445, macro-F1 0.383 (ablations: metadata fusion and CORAL both underperformed) |
| Subtype grading (AneRBC, Microcytic vs Non-Microcytic) | Acc 0.822, macro-F1 0.707 |
| External validation (CP-AnemiC, 10 hospitals) | Acc 0.862 ± 1.5% (ensemble 0.951), stable across hospitals/regions |
| Explainability | Grad-CAM: positive. Attention weights: near-uniform (honest limitation) |
| Calibration | Coverage 91.9% (target 90%), ECE 0.075 |

Ready to move to paper writing.
