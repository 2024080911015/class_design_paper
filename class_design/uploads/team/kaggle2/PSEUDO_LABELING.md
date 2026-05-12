# Pseudo Labeling Workflow

This is the legacy/manual pseudo-labeling workflow. The current Transformer
pipeline normally uses `make_soft_pseudo.py` through `run_transformer_system.py`.

## 1. Generate pseudo labels

Best option: use the strongest ensemble/submission probability CSV as teacher.

```bash
python kaggle/pseudo_labeling.py \
  --route ensemble \
  --prediction-csv models/final_magic_knn_submission.csv \
  --threshold 0.98 \
  --min-margin 0.25 \
  --per-class-limit 6000 \
  --output pseudo_labels.csv
```

Route-specific teachers also work if the ensemble file is not ready.

```bash
python kaggle/pseudo_labeling.py --route transformer --threshold 0.98 --min-margin 0.25 --output pseudo_labels_swin.csv
python kaggle/pseudo_labeling.py --route beit --threshold 0.98 --min-margin 0.25 --output pseudo_labels_beit.csv
```

The output has the columns expected by the old notebooks:
`subject,classname,img,label_int,fold`, plus `pseudo_confidence`,
`pseudo_margin`, and the original `c0..c9` probabilities.

## 2. Fine-tune with pseudo labels

Transformer route:

```bash
python kaggle/train_with_pseudo.py \
  --route transformer \
  --pseudo-csv pseudo_labels.csv \
  --epochs 2 \
  --pseudo-weight 0.35
```

BEiT is supported too:

```bash
python kaggle/train_with_pseudo.py \
  --route beit \
  --pseudo-csv pseudo_labels.csv \
  --epochs 2 \
  --pseudo-weight 0.35
```

The scripts load the original fold weights and save new weights with a
`pseudo_best_model_*_fold_{fold}.pth` pattern, so the old weights are not
overwritten.

## Practical defaults

- Start with `threshold=0.98` and `min-margin=0.25`.
- Keep `pseudo-weight` lower than real labels. `0.35` to `0.5` is a good range.
- If one class dominates, set `--per-class-limit 4000` or `6000`.
- If GPU memory is tight, reduce `--batch-size`. The defaults are conservative
  for Swin/BEiT.
- The scripts load weights on CPU first, then move the model to GPU. This avoids
  the OOM pattern that can happen with `torch.load(..., map_location="cuda")`.
