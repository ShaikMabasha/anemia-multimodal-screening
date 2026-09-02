# Multimodal Deep Learning for Non-Invasive Anemia Screening

Code accompanying the paper *"Multimodal Deep Learning for Non-Invasive
Anemia Screening: A Leakage-Audited, Externally Validated, and Calibrated
Framework"* (submitted to *Computers in Biology and Medicine*).

This repository implements a cross-attention fusion architecture combining
conjunctiva, fingernail, and palm images for non-invasive anemia screening,
severity grading, and subtype classification — together with the full data
auditing pipeline, external validation, calibration, and explainability
analysis described in the paper.

## Repository structure

```
data_audit/       Dataset inventory, subject-identity resolution, leakage
                   detection and correction, unified k-fold split assignment
training/         Single-modality baseline, multimodal fusion, severity,
                   and subtype model training (5-fold cross-validation)
evaluation/       External validation, statistical significance testing
                   (McNemar's test), ROC curves, model efficiency benchmarks
explainability_calibration/
                   Grad-CAM explainability and split conformal prediction
docs/             Project reference document (dataset decisions, schema,
                   final results)
```

## Key methodological contribution: data quality auditing

Before any model training, this pipeline audits every data source for
subject-level leakage and label consistency. Two leakage failure modes
were identified and corrected:

1. **Subject-level leakage in a public dataset's provided train/test
   split** — 98% of subjects were found in more than one partition.
2. **Pretraining leakage** in an initial multimodal fusion attempt, caused
   by inconsistent splits between single-modality baselines and the fusion
   model.

See `data_audit/` and `docs/PROJECT_REFERENCE.md` for full details, and the
paper's Methods section (Data Quality Auditing and Leakage Mitigation) for
the complete writeup.

## Reproduction pipeline (run in order)

```bash
# 1. Inventory and audit raw datasets
python data_audit/inventory_dataset_v3.py <path_to_datasets>
python data_audit/build_manifest_v4.py <path_to_datasets>
python data_audit/parse_anerbc_reports_v7.py <path_to_datasets>

# 2. Build the unified, leakage-safe k-fold split
python data_audit/build_unified_split.py final_split_manifest_linux.csv
python data_audit/assign_kfold.py unified_manifest.csv

# 3. Train single-modality baselines (repeat for conjunctiva/nail/palm)
python training/train_kfold_modality.py kfold_manifest.csv conjunctiva <base_dir>

# 4. Train multimodal fusion (warm-started from baseline checkpoints)
python training/build_fusion_manifest_v2.py unified_manifest.csv
python training/train_kfold_fusion.py kfold_manifest.csv <base_dir>

# 5. Train severity and subtype models
python training/build_severity_manifest.py <cp_anemic_dir> <xlsx_path>
python training/build_subtype_manifest.py <structured_csv> <anerbc1_dir>
python training/train_kfold_taskhead.py severity severity_manifest.csv <base_dir>
python training/train_kfold_taskhead.py subtype subtype_manifest.csv <base_dir>

# 6. Evaluate
python evaluation/evaluate_external_cpanemic.py <cp_anemic_dir> <base_dir>
python evaluation/mcnemar_fusion_vs_palm.py kfold_manifest.csv <base_dir>
python evaluation/plot_roc_curve.py kfold_manifest.csv <base_dir>
python evaluation/benchmark_model_efficiency.py

# 7. Explainability and calibration
python explainability_calibration/explain_fusion_model_v2.py kfold_manifest.csv <base_dir> 0
python explainability_calibration/calibrate_fusion_model.py kfold_manifest.csv <base_dir>
```

## Data availability

This project uses only publicly available, previously de-identified
datasets. No data is redistributed in this repository; see the paper's
*Data Availability* statement and Methods §3.1 for links to each source
(AneRBC, CP-AnemiC, and the underlying Mendeley Data repositories for the
conjunctiva/nail/palm imaging sources).

## Requirements

```bash
pip install -r requirements.txt
```

Developed and tested with Python 3.12, PyTorch 2.5.1, CUDA 12.1, on an
NVIDIA RTX A6000.

## Citation

If you use this code, please cite:
```bibtex
@article{TODO_citekey,
  title   = {Multimodal Deep Learning for Non-Invasive Anemia Screening: A
             Leakage-Audited, Externally Validated, and Calibrated Framework},
  author  = {Shaik, Mabasha and Gupta, Sumit and Kancharagunta, Kishan Babu},
  journal = {Computers in Biology and Medicine},
  year    = {2026},
  note    = {Under review}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
