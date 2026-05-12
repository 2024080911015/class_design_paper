# Auto-converted from knn.ipynb.
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
        "Use graph_smoothing.py/run_transformer_system.py, or set "
        "RUN_LEGACY_NOTEBOOK_EXPORT=1 to run the historical export."
    )


# %% Notebook cell 1
import os
import torch
import numpy as np
import pandas as pd
import timm
from tqdm.auto import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = 'dataset/imgs/test_cropped_v2'
SAMPLE_SUBMISSION_PATH = './dataset/sample_submission.csv'
IMG_SIZE = 384  # 馃憟 Swin 涓撳睘灏哄
BATCH_SIZE = 128
FOLDS = [0, 1, 2, 3, 4]
# 鍔犺浇娴嬭瘯闆嗚矾寰?
df_test = pd.read_csv(SAMPLE_SUBMISSION_PATH)
image_names = df_test['img'].values
# 鏍囧噯 ImageNet 褰掍竴鍖?
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
class TestFeatureDataset(Dataset):
    def __len__(self): return len(image_names)
    def __getitem__(self, idx):
        img_path = os.path.join(TEST_DIR, image_names[idx])
        image = cv2.imread(img_path)
        if image is None:
            image = cv2.imread(os.path.join('./dataset/imgs/test', image_names[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return transform(image=image)['image']
print("馃殌 鍑嗗鎻愬彇 Swin-Base 5鎶橀骞茬壒寰?..")
test_loader = DataLoader(TestFeatureDataset(), batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
all_fold_features = []
for fold in FOLDS:
    print(f"\n馃懆鈥嶁殩锔?姝ｅ湪鍔犺浇 Swin Fold {fold} 妯″瀷...")
    
    # 1. 姣忔寰幆鍒涘缓涓€涓共鍑€鐨勬ā鍨?
    model = timm.create_model('swin_base_patch4_window12_384.ms_in22k', pretrained=False, num_classes=10)
    
    # 2. 璇诲彇瀵瑰簲鎶樼殑鏉冮噸 (娉ㄦ剰璺緞涓?BEiT 鍜?EffB3 涓嶅悓)
    weight_path = f'models/best_model_swin_fold_{fold}.pth'
    if not os.path.exists(weight_path):
        print(f"鈿狅笍 鎵句笉鍒版潈閲?{weight_path}锛岃烦杩囨鎶橈紒")
        continue
        
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    # 3. 鍒囬櫎鍒嗙被澶达紒
    model.reset_classifier(0)
    model.to(DEVICE)
    model.eval()
    # 缂栬瘧鍔犻€?
    if int(torch.__version__.split('.')[0]) >= 2:
        model = torch.compile(model)
    fold_features = []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for images in tqdm(test_loader, desc=f"馃拵 鎻愬彇 Fold {fold} 鐗瑰緛"):
                features = model(images.to(DEVICE))
                fold_features.append(features.cpu().numpy())
    current_fold_final_features = np.concatenate(fold_features, axis=0)
    all_fold_features.append(current_fold_final_features)
    
    del model
    torch.cuda.empty_cache()
if not all_fold_features:
    raise ValueError("馃毃 娌℃湁鎴愬姛鎻愬彇鍒颁换浣曠壒寰侊紒")
# 馃幆 瀵?5 鎶樼壒寰佹眰绠楁湳骞冲潎
print("\n馃 姝ｅ湪瀵?5 鎶樼壒寰佽繘琛岀畻鏈钩鍧囪瀺鍚?..")
all_fold_features = np.array(all_fold_features) 
final_avg_features = np.mean(all_fold_features, axis=0) 
save_path = 'models/test_features_swin.npy'
np.save(save_path, final_avg_features)
print(f"鉁?Swin 5鎶樼壒寰佽瀺鍚堝畬姣曪紒鐭╅樀褰㈢姸: {final_avg_features.shape}")
print(f"馃搧 宸插畨鍏ㄤ繚瀛樿嚦: {save_path}")

# %% Notebook cell 2
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
# ==========================================
# 鈿欙笍 1. 鏍稿績瓒呭弬鏁伴厤缃尯 (渚涚嚎涓婄柉鐙傝皟鍙?
# ==========================================
BASE_SUBMISSION_PATH = 'models/stacking_ensemble_submission.csv'
OUTPUT_PATH = 'models/final_magic_knn_submission.csv'
# 鍚勪釜妯″瀷鎻愬彇鐨勬祴璇曢泦楂樼淮鐗瑰緛锛屼互鍙婂畠浠搴旂殑鏉冮噸锛?
# 蹇呴』涓€涓€瀵瑰簲锛佷綘鍙互鏍规嵁瀹冧滑鍗曟ā鍨嬬殑寰楀垎鏉ュ垎閰嶈繖閲岀殑鏉冮噸
FEATURE_PATHS = [
    'models/test_features_swin.npy',
    'models/test_features_beit.npy',  
    'models/test_features_effb3.npy'  
]
FEATURE_WEIGHTS = [0.4, 0.35, 0.25] # 馃憟 鏂板锛氱壒寰佹潈閲嶏紒鍔犺捣鏉ユ渶濂界瓑浜?
# KNN 瓒呭弬鏁?
K = 25              
ALPHA = 5.0         
CLIP_MIN = 1e-5     
# ==========================================
# 馃搳 2. 鏁版嵁涓庣壒寰佸姞杞?
# ==========================================
print("馃攳 [1/5] 姝ｅ湪鍔犺浇鍩虹棰勬祴姒傜巼...")
sub_df = pd.read_csv(BASE_SUBMISSION_PATH)
cols = [f'c{i}' for i in range(10)]
original_probs = sub_df[cols].values
print("馃З [2/5] 姝ｅ湪鐙珛褰掍竴鍖栧苟鍔犳潈鎷兼帴楂樼淮瑙嗚鐗瑰緛...")
weighted_features_list = []
for path, weight in zip(FEATURE_PATHS, FEATURE_WEIGHTS):
    if os.path.exists(path):
        print(f"   -> 鍔犺浇鐗瑰緛: {path} | 鏉冮噸: {weight}")
        feat = np.load(path)
        
        # 馃幆 榄旀硶 1锛氬厛鐙珛杩涜 L2 褰掍竴鍖栵紝澶у鍥炲埌鍚屼竴璧疯窇绾?
        feat_norm = normalize(feat, norm='l2', axis=1)
        
        # 馃幆 榄旀硶 2锛氫箻涓婁綘璧嬩簣瀹冪殑鏉冮噸
        feat_weighted = feat_norm * weight
        
        weighted_features_list.append(feat_weighted)
    else:
        print(f"   WARN: feature file not found and skipped: {path}")
if not weighted_features_list:
    raise ValueError("馃毃 娌℃湁鎵惧埌浠讳綍鐗瑰緛鏂囦欢锛岃妫€鏌ヨ矾寰勶紒")
# 鎷兼帴鍔犳潈鍚庣殑鐗瑰緛
combined_features = np.hstack(weighted_features_list)
# 馃幆 榄旀硶 3锛氭暣浣撴嫾鎺ュ悗鍐嶅仛涓€娆?L2 褰掍竴鍖栵紝鍠傜粰 KNN
print("鈿栵笍 [3/5] 鎵ц鍏ㄥ眬鐗瑰緛 L2 褰掍竴鍖?..")
combined_features = normalize(combined_features, norm='l2', axis=1)
# ==========================================
# 馃尣 3. 鏋勫缓 KNN 绱㈠紩鏍?
# ==========================================
print(f"馃尣 [4/5] 姝ｅ湪鏋勫缓 KNN 鍥剧粨鏋?(K={K}, Metric=Cosine)...")
# n_jobs=-1 鎷夋弧 CPU 绠楀姏
knn = NearestNeighbors(n_neighbors=K, metric='cosine', n_jobs=-1)
knn.fit(combined_features)
print("馃攷 姝ｅ湪璁＄畻娴嬭瘯闆嗕腑姣忎竴寮犲浘鐨勮繎閭昏竟鐣?..")
# distances 杩斿洖鐨勬槸浣欏鸡璺濈 (1 - cos_sim)
distances, indices = knn.kneighbors(combined_features)
# ==========================================
# 馃 4. 璺濈鍔犳潈鐨勬鐜囧钩婊?(Message Passing)
# ==========================================
print(f"馃 [5/5] 鎵ц闈炵嚎鎬ц窛绂诲姞鏉冨钩婊?(Alpha={ALPHA})...")
smoothed_probs = np.zeros_like(original_probs)
for i in tqdm(range(len(original_probs)), desc="Graph Smoothing"):
    neighbor_idx = indices[i]
    neighbor_dist = distances[i] 
    
    # 灏嗕綑寮﹁窛绂昏浆鎹负浣欏鸡鐩镐技搴?(鍊煎湪 0~1 涔嬮棿锛岃秺澶ц秺鐩镐技)
    # 鍔犱笂 1e-8 闃叉闄や互 0 鐨勮绠楁孩鍑?
    similarities = 1.0 - neighbor_dist 
    similarities = np.clip(similarities, 0.0, 1.0)
    
    # 鏍稿績榄旀硶锛氫娇鐢ㄦ寚鏁?alpha 鏀惧ぇ鏋佸害鐩镐技鐨勫抚鐨勬潈閲?
    weights = similarities ** ALPHA
    
    # 鏉冮噸褰掍竴鍖?(璁╁綋鍓嶈妭鐐瑰懆鍥寸殑閭诲眳鏉冮噸鍜屼负 1)
    weights = weights / (weights.sum() + 1e-8)
    
    # 鍔犳潈鑱氬悎 (鐩稿綋浜?GCN 涓殑 Aggregate 姝ラ)
    smoothed_probs[i] = np.average(original_probs[neighbor_idx], axis=0, weights=weights)
# ==========================================
# 馃洝锔?5. Log Loss 鏋侀檺闃茬垎涓庝繚瀛?
# ==========================================
print(f"馃洝锔?鎵ц姒傜巼瑁佸壀 (Clip [{CLIP_MIN}, {1.0 - CLIP_MIN}])...")
smoothed_probs = np.clip(smoothed_probs, CLIP_MIN, 1.0 - CLIP_MIN)
# 瑁佸壀鍚庯紝姒傜巼鍜屽彲鑳戒笉绛変簬 1锛屽繀椤诲啀娆″綊涓€鍖?
smoothed_probs = smoothed_probs / smoothed_probs.sum(axis=1, keepdims=True)
# 鍐欏叆 DataFrame
sub_df[cols] = smoothed_probs
sub_df.to_csv(OUTPUT_PATH, index=False)
print(f"馃帀 缁堟瀬榄旀硶鐗?CSV 宸茬敓鎴愶紒鏂囦欢: {OUTPUT_PATH}")
print("馃 绁濆埛姒滈『鍒╋紒")

