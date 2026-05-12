# Pseudo Label Flow

This flow summarizes the pseudo-labeling part of the late-stage `kaggle2`
Transformer system. It starts after supervised Swin/BEiT base models and strict
assets are available.

```mermaid
flowchart TD
    A["Supervised base models\nSwin + BEiT fold weights"] --> B["build_strict_oof_assets.py\nRegenerate clean OOF/test probabilities\nExtract train/test features"]
    B --> C["Strict assets\nOOF probs, test probs, features,\ntrain_index.csv, test_index.csv"]

    C --> D1["graph_smoothing.py\nTransductive graph KNN"]
    C --> D2["graph_smoothing.py\nConservative train_only graph KNN"]

    D1 --> E1["calibrate_predictions.py\nTransductive calibrated teacher"]
    D2 --> E2["calibrate_predictions.py\nConservative calibrated teacher"]

    E1 --> F["final_candidate_blender.py\nRound-1 teacher blend"]
    E2 --> F

    F --> G["make_soft_pseudo.py\nUse teacher probabilities"]
    E1 --> H["Agreement filter\nTop-1 label must match"]
    E2 --> H
    H --> G

    G --> I["pseudo_soft_labels.csv\nSoft labels with c0..c9 probabilities\nDefault threshold 0.90\nDefault max_pseudo 40000\nDefault per_class_limit 4000"]

    I --> J1["train_with_pseudo.py --route transformer\nSwin student fine-tuning\nSoft KL + optional hard CE"]
    I --> J2["train_with_pseudo.py --route beit\nBEiT student fine-tuning\nSoft KL + optional hard CE"]

    J1 --> K1["pseudo_best_model_swin_fold_{fold}.pth\npseudo_oof_preds_swin.npy"]
    J2 --> K2["pseudo_best_model_beit_fold_{fold}.pth\npseudo_oof_preds_beit.npy"]

    K1 --> L1["pseudo_labeling.py\nPseudo Swin test probabilities"]
    K2 --> L2["pseudo_labeling.py\nPseudo BEiT test probabilities"]

    L1 --> M1["build_strict_oof_assets.py\npseudo_swin strict asset"]
    L2 --> M2["build_strict_oof_assets.py\npseudo_beit strict asset"]
    K1 --> M1
    K2 --> M2

    M1 --> N["Round-2 candidate pool\nbase Swin + base BEiT\npseudo_swin + pseudo_beit\nDINOv2/features"]
    M2 --> N
    C --> N

    N --> O1["graph_smoothing.py\nFinal transductive graph"]
    N --> O2["graph_smoothing.py\nFinal conservative graph"]

    O1 --> P1["calibrate_predictions.py\nfinal_graph_knn_calibrated"]
    O2 --> P2["calibrate_predictions.py\nfinal_graph_knn_conservative_calibrated"]

    P1 --> Q["final_candidate_blender.py\nOOF-searched final blend"]
    P2 --> Q
    E1 --> Q
    E2 --> Q

    Q --> R["submission_transformer_system.csv"]
```

## Key Guardrails

- The teacher is not a single raw model. It is built from graph-smoothed and
  calibrated Swin/BEiT candidates.
- Soft pseudo labels keep the full `c0..c9` distribution, instead of only the
  hard top-1 class.
- By default, pseudo samples are kept only when transductive and conservative
  teachers agree on top-1.
- Pseudo fine-tuning starts from supervised fold weights and saves separate
  `pseudo_best_model_*` weights, so base weights are not overwritten.
- Round 2 treats pseudo students as new candidates, then repeats graph smoothing,
  calibration, and final blending.

