# Notebook Python Exports

The original notebooks are kept as references. These Python files are direct legacy exports with notebook cells preserved as `# %%` sections.

They are disabled by default because some cells belong to the old notebook workflow and can run setup, dataset, or legacy-model steps out of order. Set `RUN_LEGACY_NOTEBOOK_EXPORT=1` only when you intentionally want to debug a historical export.

The old setup/download cells in `kaggle_test_swin.ipynb` have also been made no-op. The maintained route should be run through the scripts listed in `TRANSFORMER_SYSTEM.md`.

- `kaggle_test_swin.py`: original Swin route, including environment setup, YOLO crop generation, 5-fold Swin training, OOF saving, test submission, and feature extraction.
- `kaggle_test_be.py`: original BEiT route, including local pretrained weight check, 5-fold BEiT training, OOF saving, test submission, and feature extraction.
- `and2.py`: original level-2 logistic-regression stacking over Swin, EffB3, and BEiT probabilities.
- `knn.py`: original Swin feature extraction plus old test-only KNN graph smoothing submission.
- `kl.py`: original EffB0 student/KL distillation and probability sharpening experiments.

For the current competition pipeline, prefer `train_base_route.py`, `run_transformer_system.py`, and the strict asset scripts. These exports are mainly for traceability, debugging, or reusing old notebook code without opening Jupyter.
