# Auto-converted from and2.ipynb.
# Original notebook is kept for reference; this script preserves code-cell order.
from __future__ import annotations

import os
import subprocess


def run_shell(command: str) -> None:
    kwargs = {"shell": True, "check": True}
    if os.name != "nt":
        kwargs["executable"] = "/bin/bash"
    subprocess.run(command, **kwargs)


if os.environ.get("RUN_LEGACY_NOTEBOOK_EXPORT") != "1":
    raise SystemExit(
        "This legacy notebook export is disabled by default. "
        "Use final_candidate_blender.py/run_transformer_system.py, or set "
        "RUN_LEGACY_NOTEBOOK_EXPORT=1 to run the historical export."
    )


# %% Notebook cell 1
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
# ==========================================
# 1. 璇诲彇鏁版嵁涓庢爣绛?
# ==========================================
CSV_PATH = 'train_with_folds.csv'
df_train = pd.read_csv(CSV_PATH)
y_true = df_train['label_int'].values
# ==========================================
# 2. 璇诲彇鍚勪釜妯″瀷鐨?OOF 棰勬祴姒傜巼 (璁粌闆?
# ==========================================
oof_swin = np.load('models/oof_preds_swin.npy')
oof_effb3 = np.load('models/effb3/oof_preds_effb3.npy')
# 鈴?[涓夋ā鍨嬭瀺鍚堝偍澶嘳 绛?BEiT 璁粌瀹屾垚鍚庯紝鍙栨秷涓嬮潰杩欒鐨勬敞閲婏細
oof_beit = np.load('models/beit/oof_preds_beit.npy')
# ==========================================
# 3. 璇诲彇鍚勪釜妯″瀷鐨勬祴璇曢泦 Submission (鎺ㄧ悊闆?
# ==========================================
cols = [f'c{i}' for i in range(10)]
sub_swin = pd.read_csv('models/submission_swin_5fold_fixed.csv')
sub_effb3 = pd.read_csv('models/effb3/submission_effb3_5fold.csv')
# 鈴?[涓夋ā鍨嬭瀺鍚堝偍澶嘳 绛?BEiT 鎺ㄧ悊瀹屾垚鍚庯紝鍙栨秷涓嬮潰杩欒鐨勬敞閲婏細
sub_beit = pd.read_csv('models/beit/submission_beit_5fold.csv')
# ==========================================
# 4. 灏嗘鐜囨嫾鎺ユ垚 Stacking 鐗瑰緛鐭╅樀
# ==========================================
# 馃憠 褰撳墠锛氬弻妯″瀷鎷兼帴 (N x 20 缁寸壒寰?
#X_train = np.hstack([oof_swin, oof_effb3])
#X_test = np.hstack([sub_swin[cols].values, sub_effb3[cols].values])
# 鈴?[涓夋ā鍨嬭瀺鍚堝偍澶嘳 绛?BEiT 灏辩华鍚庯紝銆愭敞閲婃帀涓婇潰涓よ銆戯紝骞躲€愬彇娑堜笅闈袱琛岀殑娉ㄩ噴銆?(鍙樻垚 N x 30 缁寸壒寰?
X_train = np.hstack([oof_swin, oof_effb3, oof_beit])
X_test = np.hstack([sub_swin[cols].values, sub_effb3[cols].values, sub_beit[cols].values])
# ==========================================
# 5. 浣跨敤 5 鎶樹氦鍙夐獙璇佽缁冮€昏緫鍥炲綊 (鍏冩ā鍨?
# ==========================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta = np.zeros((len(X_train), 10))
test_meta_preds = np.zeros((len(X_test), 10))
print(f"馃 鍚姩 Stacking 璁粌... 褰撳墠杈撳叆鐗瑰緛缁村害: {X_train.shape[1]}")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_true)):
    X_trn, y_trn = X_train[trn_idx], y_true[trn_idx]
    X_val, y_val = X_train[val_idx], y_true[val_idx]
    
    # 閫昏緫鍥炲綊鍏冩ā鍨?(C=0.1 鏄瀬浣崇殑姝ｅ垯鍖栧弬鏁帮紝闃叉杩囨嫙鍚?
    meta_model = LogisticRegression(max_iter=1000, C=0.1)
    meta_model.fit(X_trn, y_trn)
    
    # 棰勬祴褰撳墠鎶樼殑楠岃瘉闆?
    oof_meta[val_idx] = meta_model.predict_proba(X_val)
    # 棰勬祴鐪熷疄娴嬭瘯闆嗗苟鍋氬钩鍧囩疮鍔?
    test_meta_preds += meta_model.predict_proba(X_test) / 5
# ==========================================
# 6. 璇勪及涓庝繚瀛?
# ==========================================
stacking_loss = log_loss(y_true, oof_meta)
print(f"馃敟 Stacking 铻嶅悎鍚庣殑鏋佽嚧 CV Log Loss: {stacking_loss:.5f}")
# 鍊熺敤绗竴涓?submission 鐨勫澹冲瓨鏀炬渶缁堢粨鏋?
final_sub = sub_swin.copy()
final_sub[cols] = test_meta_preds
final_sub.to_csv('models/stacking_ensemble_submission.csv', index=False)
print("Stacking submission generated.")

