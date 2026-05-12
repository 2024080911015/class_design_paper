# Transformer Route Upgrade

Run commands from the project root.

## One Command

The Transformer route now uses Swin and BEiT only. The old EffB3/CNN branch is no longer part of this route.

Recommended full order:

```bash
python kaggle/yolo_crop_images.py --split both --model yolov8n.pt
python kaggle/train_base_route.py --route swin
python kaggle/train_base_route.py --route beit
python kaggle/run_transformer_system.py
```

`train_base_route.py` does not download pretrained models. Put these local files in the project root first:

```text
swin_base.safetensors
beit_large.safetensors
```

Each base run is archived under `models/runs/base_<route>_<run_id>/`, while the latest successful weights and base OOF/test files are also copied to the canonical paths expected by the rest of the pipeline. Pass `--no-canonical` if you only want the archived run outputs.

Before running `run_transformer_system.py`, these supervised base fold weights should exist:

```text
models/best_model_swin_fold_{0..4}.pth
models/beit/best_model_beit_fold_{0..4}.pth
```

The default command is a full two-round post-base loop:

```bash
python kaggle/run_transformer_system.py
```

By default, base assets `swin` and `beit` regenerate OOF/test probabilities from the supervised fold weights. This keeps old post-processed CSV/NPY files from being silently treated as raw base probabilities. Use `--use-existing-base-probs` only when you are sure the existing `oof_preds_*.npy` and submission CSV files are clean single-model outputs.

It executes:

```text
Round 1 base Swin/BEiT strict assets
-> DINOv2 frozen crop/full features
-> transductive graph KNN + conservative graph KNN
-> OOF calibration
-> round1 teacher blend + trans/conservative agreement-filtered soft pseudo labels
-> Swin/BEiT soft-KL pseudo fine-tuning
-> pseudo Swin/BEiT test inference
-> pseudo Swin/BEiT strict assets
-> Round 2 graph KNN + calibration
-> OOF-searched final candidate blender
-> submission_transformer_system.csv
```

Useful variants:

```bash
# Print every command without running it.
python kaggle/run_transformer_system.py --dry-run

# DINOv2 is local/cache-only by default. Explicitly allow download only if you want that.
python kaggle/run_transformer_system.py --allow-dinov2-download

# Skip DINOv2 if the server has no cached model or network.
python kaggle/run_transformer_system.py --skip-dinov2 --feature-models swin beit

# Stop after teacher and pseudo labels, without student fine-tuning.
python kaggle/run_transformer_system.py --skip-finetune

# Fine-tune only Swin as the student route.
python kaggle/run_transformer_system.py --train-routes transformer

# Use existing pseudo student weights and rebuild the second-round ensemble.
python kaggle/run_transformer_system.py --skip-pseudo --skip-finetune --second-round-from-existing

# Regenerate OOF and test probabilities with the same TTA views before graph/calibration.
python kaggle/run_transformer_system.py --prob-tta-modes base zoom hflip

# Trust existing base OOF/test probability files instead of re-inferring from fold weights.
python kaggle/run_transformer_system.py --use-existing-base-probs
```

## Produced Files

Round 1 teacher:

```text
crop_flags_train.csv
crop_flags_test.csv
models/strict_assets/assets_manifest.json
models/strict_assets/graph_knn_calibrated_test_preds.csv
models/strict_assets/graph_knn_conservative_calibrated_test_preds.csv
models/strict_assets/round1_teacher_blend_test_preds.csv
pseudo_soft_labels.csv
```

Round 2 final:

```text
models/strict_assets/test_preds_pseudo_swin.csv
models/strict_assets/test_preds_pseudo_beit.csv
models/strict_assets/final_graph_knn_calibrated_test_preds.csv
models/strict_assets/final_graph_knn_conservative_calibrated_test_preds.csv
models/strict_assets/transformer_candidate_blend_test_preds.csv
submission_transformer_system.csv
```

## Manual Order

If you want to run pieces by hand, the order is:

```bash
python kaggle/build_strict_oof_assets.py --route swin --infer-probs-from-weights --prob-tta-modes base --normalize-features
python kaggle/build_strict_oof_assets.py --route beit --infer-probs-from-weights --prob-tta-modes base --normalize-features

python kaggle/extract_dinov2_features.py --model-name facebook/dinov2-base --asset-name dinov2_crop --train-dir dataset/imgs/train_cropped_v2 --train-fallback-dir dataset/imgs/train --test-dir dataset/imgs/test_cropped_v2 --test-fallback-dir dataset/imgs/test --normalize-features
python kaggle/extract_dinov2_features.py --model-name facebook/dinov2-base --asset-name dinov2_full --train-dir dataset/imgs/train --test-dir dataset/imgs/test --normalize-features

python kaggle/validate_strict_assets.py --asset-dir models/strict_assets --fail-on-issues

python kaggle/graph_smoothing.py --prob-models swin beit --feature-models swin beit dinov2_crop dinov2_full --feature-dirichlet-count 5 --output-prefix graph_knn
python kaggle/graph_smoothing.py --prob-models swin beit --feature-models swin beit dinov2_crop dinov2_full --feature-dirichlet-count 5 --output-prefix graph_knn_conservative --oof-neighbor-mode train_only

python kaggle/calibrate_predictions.py --oof-preds graph_knn_oof_preds.npy --test-preds graph_knn_test_preds.npy --per-class --output-prefix graph_knn_calibrated
python kaggle/calibrate_predictions.py --oof-preds graph_knn_conservative_oof_preds.npy --test-preds graph_knn_conservative_test_preds.npy --per-class --output-prefix graph_knn_conservative_calibrated

python kaggle/final_candidate_blender.py --candidate-prefixes graph_knn_calibrated graph_knn_conservative_calibrated --output-prefix round1_teacher_blend --submission models/strict_assets/round1_teacher_blend_test_preds.csv

python kaggle/make_soft_pseudo.py --teacher-csv models/strict_assets/round1_teacher_blend_test_preds.csv --agreement-csvs models/strict_assets/graph_knn_calibrated_test_preds.csv models/strict_assets/graph_knn_conservative_calibrated_test_preds.csv --require-agreement --output pseudo_soft_labels.csv

python kaggle/train_with_pseudo.py --route transformer --pseudo-csv pseudo_soft_labels.csv --soft-pseudo --pseudo-sample-ratio 1.0
python kaggle/train_with_pseudo.py --route beit --pseudo-csv pseudo_soft_labels.csv --soft-pseudo --pseudo-sample-ratio 1.0

python kaggle/pseudo_labeling.py --route transformer --weights models/pseudo_best_model_swin_fold_{fold}.pth --prob-output models/strict_assets/test_preds_pseudo_swin.csv --output models/strict_assets/hard_pseudo_from_pseudo_swin.csv
python kaggle/pseudo_labeling.py --route beit --weights models/beit/pseudo_best_model_beit_fold_{fold}.pth --prob-output models/strict_assets/test_preds_pseudo_beit.csv --output models/strict_assets/hard_pseudo_from_pseudo_beit.csv

python kaggle/build_strict_oof_assets.py --route transformer --asset-name pseudo_swin --weights models/pseudo_best_model_swin_fold_{fold}.pth --oof-preds models/pseudo_oof_preds_swin.npy --test-preds-csv models/strict_assets/test_preds_pseudo_swin.csv --normalize-features
python kaggle/build_strict_oof_assets.py --route beit --asset-name pseudo_beit --weights models/beit/pseudo_best_model_beit_fold_{fold}.pth --oof-preds models/beit/pseudo_oof_preds_beit.npy --test-preds-csv models/strict_assets/test_preds_pseudo_beit.csv --normalize-features

python kaggle/graph_smoothing.py --prob-models swin beit pseudo_swin pseudo_beit --feature-models swin beit dinov2_crop dinov2_full pseudo_swin pseudo_beit --feature-dirichlet-count 5 --output-prefix final_graph_knn
python kaggle/calibrate_predictions.py --oof-preds final_graph_knn_oof_preds.npy --test-preds final_graph_knn_test_preds.npy --per-class --output-prefix final_graph_knn_calibrated

python kaggle/final_candidate_blender.py
```

## Import CNN Assets

If the CNN/top3 route is trained on another machine, first export:

```text
oof_preds_top3cnn_raw.csv
test_preds_top3cnn_raw.csv
oof_preds_top3cnn_hardcut.csv
test_preds_top3cnn_hardcut.csv
```

Then import both CNN candidates into the Transformer strict-asset directory:

```bash
python kaggle/import_cnn_fusion_assets.py \
  --cnn-oof-csv path/to/oof_preds_top3cnn_raw.csv \
  --cnn-test-csv path/to/test_preds_top3cnn_raw.csv \
  --asset-name top3cnn_raw

python kaggle/import_cnn_fusion_assets.py \
  --cnn-oof-csv path/to/oof_preds_top3cnn_hardcut.csv \
  --cnn-test-csv path/to/test_preds_top3cnn_hardcut.csv \
  --asset-name top3cnn_hardcut
```

Now `top3cnn_raw` and `top3cnn_hardcut` are available as:

```text
models/strict_assets/oof_preds_top3cnn_raw.npy
models/strict_assets/test_preds_top3cnn_raw.npy
models/strict_assets/oof_preds_top3cnn_hardcut.npy
models/strict_assets/test_preds_top3cnn_hardcut.npy
```

The root `fuse_transformer_cnn.py` will auto-discover them together with the transformer graph candidates.

## Guardrails

- `build_strict_oof_assets.py` now checks OOF/test row counts. CSV OOF files are aligned by metadata; plain NPY OOF files are accepted only by shape and must already match `train_with_folds.csv`.
- Each strict asset writes `asset_meta_<name>.json`, recording whether probabilities came from fold weights or existing prediction files.
- `validate_strict_assets.py` writes `assets_manifest.json` and is called by the one-command runner unless `--skip-asset-manifest` is passed.
- `yolo_crop_images.py` writes `crop_flags_train.csv` and `crop_flags_test.csv`; the final fusion script reads those names by default when they exist.
- Route-specific normalization is preserved when weights are re-inferred: Swin uses ImageNet mean/std, while BEiT uses `[0.5, 0.5, 0.5]` mean/std to match the original BEiT training notebook.
- `graph_smoothing.py` searches model probability weights on OOF before KNN smoothing. Use `--no-prob-weight-search` only for ablations.
- `graph_smoothing.py` also adds a small Dirichlet feature-weight search by default, so DINOv2 crop/full, Swin, and BEiT feature weights are searched on OOF without making the graph stage too slow.
- `final_candidate_blender.py` blends round1/round2 transductive/conservative transformer candidates in log space, then recalibrates the result.
- Before pseudo training, `round1_teacher_blend` blends the transductive and conservative teachers, and `make_soft_pseudo.py` keeps only samples where both teachers agree on top1 by default.
- TTA is optional and safe: `--prob-tta-modes base zoom hflip` regenerates OOF and test probabilities together, including left/right class swaps for horizontal flip. Do not TTA only test.
- The final fusion script does not rely only on missing crop files. It also compares crop/full image size and aspect ratio, and can force YOLO-suspicious rows back to the CNN candidate when Transformer is low-confidence but CNN is confident.
- The default graph is transductive. The conservative graph uses `--oof-neighbor-mode train_only` and should be kept as a safety candidate.
- Default soft pseudo selection is intentionally tighter: `threshold=0.90`, `max_pseudo=40000`, `per_class_limit=4000`.
- `train_with_pseudo.py` defaults to balanced pseudo sampling with `--pseudo-sample-ratio 1.0`, meaning roughly real:pseudo = 1:1 per epoch.
