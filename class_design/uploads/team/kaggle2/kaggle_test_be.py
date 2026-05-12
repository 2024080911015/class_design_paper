# Auto-converted from kaggle_test_be.ipynb.
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
        "Use train_base_route.py/run_transformer_system.py, or set "
        "RUN_LEGACY_NOTEBOOK_EXPORT=1 to run the historical export."
    )


# %% Notebook cell 1
# Local BEiT pretrained weights are required. This legacy export no longer downloads models.
import os
if not os.path.exists("beit_large.safetensors"):
    raise FileNotFoundError("Missing local pretrained weights: beit_large.safetensors")
size_mb = os.path.getsize("beit_large.safetensors") / (1024 * 1024)
print(f"Local BEiT pretrained weights found: {size_mb:.2f} MB")

# %% Notebook cell 2
import os
import gc
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import log_loss, confusion_matrix
import timm
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import load_file
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# ==========================================
# 鈿欙笍 1. 鍏ㄥ眬閰嶇疆鍙傛暟 (BEiT-Large 宸ㄥ吔鐗?
# ==========================================
CSV_PATH = 'dataset/driver_imgs_list.csv'
TRAIN_DIR = 'dataset/imgs/train_cropped_v2' 
MODEL_NAME = 'beit_large_patch16_224.in22k_ft_in22k_in1k'
WEIGHTS_PATH = 'beit_large.safetensors' 
IMG_SIZE = 224      # 鉁?閬电収 PPT: 闄嶇淮鍒?224x224
EPOCHS = 10         
BATCH_SIZE = 32     # 鉁?鍙傛暟閲?307M锛屽嵆浣挎槸 5090 涔熶笉寤鸿寮€澶ぇ锛?2 鎴?64 鏄瀬闄?
ACCUMULATION_STEPS = 1  
NUM_SPLITS = 5
TARGET_FOLDS = [0, 1, 2, 3, 4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
gc.collect()
torch.cuda.empty_cache()
# ==========================================
# 馃搳 2. 鏁版嵁鍒掑垎 (蹇呴』鍧氬畧 GroupKFold)
# ==========================================
def generate_balanced_folds(csv_path, n_splits=5):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    if 'label_int' not in df.columns:
        df['label_int'] = df['classname'].str.extract(r'(\d+)').astype(int)
    driver_counts = df.groupby('subject').size().sort_values(ascending=False)
    fold_totals = np.zeros(n_splits)
    fold_groups = [[] for _ in range(n_splits)]
    for subject, count in driver_counts.items():
        min_fold_idx = np.argmin(fold_totals)
        fold_groups[min_fold_idx].append(subject)
        fold_totals[min_fold_idx] += count
    df['fold'] = -1
    for i, subjects in enumerate(fold_groups):
        df.loc[df['subject'].isin(subjects), 'fold'] = i
    print("\n" + "="*50)
    print("           5-fold data distribution")
    print("="*50)
    for i in range(n_splits):
        print(f"Fold {i}  |  椹鹃┒鍛樻暟閲? {len(fold_groups[i]):2d}  |  鍥剧墖鎬绘暟: {int(fold_totals[i]):4d}")
    print("="*50 + "\n")
    df.to_csv("train_with_folds.csv", index=False)
    return df
# ==========================================
# 馃柤锔?3. 鏁版嵁闆嗕笌澧炲己
# ==========================================
def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.6),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-10, 10), p=0.5),
        A.GaussNoise(p=0.4),
        # Transformer 閫氬父瀵规摝闄ゆ病閭ｄ箞鏁忔劅锛岃繖閲屼繚鐣欓€傚害澧炲己
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 16), hole_width_range=(1, 16), fill=0, p=0.4),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # BEiT 鏍囧噯棰勫鐞嗗潎鍊兼柟宸?
        ToTensorV2()
    ])
def get_valid_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
class DriverDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['classname'], row['img'])
        
        image = cv2.imread(img_path)
        if image is None:
            fallback_path = os.path.join('./dataset/imgs/train', row['classname'], row['img'])
            image = cv2.imread(fallback_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, row['label_int']
# ==========================================
# 馃洜锔?4. 璇勪及宸ュ叿
# ==========================================
def plot_confusion_matrix(y_true, y_pred, fold_idx, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Fold {fold_idx} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'cm_fold_{fold_idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()
# ==========================================
# 馃殌 5. 涓诲共璁粌娴佺▼
# ==========================================
def main():
    base_dir = 'models/beit' # 鉁?BEiT 涓撳睘瀛樺偍璺緞
    os.makedirs(base_dir, exist_ok=True)
    full_df = generate_balanced_folds(CSV_PATH, NUM_SPLITS)
    oof_preds = np.zeros((len(full_df), 10))
    for fold in TARGET_FOLDS:
        print(f"\n{'='*40}\n馃専 寮€濮嬭缁?Fold {fold} 馃専\n{'='*40}")
        train_df = full_df[full_df['fold'] != fold]
        val_df = full_df[full_df['fold'] == fold]
        train_loader = DataLoader(
            DriverDataset(train_df, TRAIN_DIR, transform=get_train_transforms(IMG_SIZE)),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            DriverDataset(val_df, TRAIN_DIR, transform=get_valid_transforms(IMG_SIZE)),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
        )
        # 1. 寤虹珛绌烘ā鍨?(鉁?閬电収 PPT: drop_path=0.1)
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=10, drop_path_rate=0.1)
        
        # 2. 鍔犺浇鏈湴 Safetensors 鏉冮噸 (鍓旈櫎 head)
        print("馃摜 姝ｅ湪鎵嬪伐鍔犺浇鏈湴 BEiT 棰勮缁冩潈閲?..")
        state_dict = load_file(WEIGHTS_PATH)
        for key in list(state_dict.keys()):
            if key.startswith('head.'):
                del state_dict[key]
                
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        # 鉁?宸紓鍖栧涔犵巼 (閫艰繎 LLRD 鏁堟灉)锛氶骞茬綉缁滃彧寰皟锛屽垎绫诲ご鏀惧ぇ姝ラ暱瀛︿範
        head_params = list(model.head.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith('head.')]
        
        optimizer = AdamW([
            {'params': backbone_params, 'lr': 2e-5}, # 楠ㄥ共缃戠粶瀛︿範鐜囧皬
            {'params': head_params, 'lr': 2e-4}      # 鍒嗙被澶村涔犵巼澶?10 鍊?
        ], weight_decay=1e-2)
        class_weights = torch.tensor([1.0]*9 + [1.5], dtype=torch.float).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scaler = torch.amp.GradScaler('cuda')
        total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        best_val_loss = float('inf')
        save_path = os.path.join(base_dir, f"best_model_beit_fold_{fold}.pth")
        EARLY_STOP_PATIENCE = 3
        epochs_no_improve = 0
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            for i, (images, labels) in enumerate(pbar):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels) / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                train_loss += loss.item() * ACCUMULATION_STEPS
                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    # 姊害瑁佸壀 (闃叉澶фā鍨嬭缁冨垵鏈熷穿鎺?
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                pbar.set_postfix({'loss': f"{loss.item()*ACCUMULATION_STEPS:.4f}"})
            model.eval()
            val_loss = 0.0
            fold_preds_list, fold_labels = [], []
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]")
            with torch.no_grad():
                for images, labels in vbar:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    fold_preds_list.append(outputs.softmax(dim=1).cpu().numpy())
                    fold_labels.extend(labels.cpu().numpy())
                    vbar.set_postfix({'v_loss': f"{loss.item():.4f}"})
            avg_val_loss = val_loss / len(val_loader)
            current_fold_preds = np.concatenate(fold_preds_list, axis=0)
            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved: {best_val_loss:.4f} -> {avg_val_loss:.4f}; saving model.")
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                epochs_no_improve = 0 
                
                oof_preds[val_df.index] = current_fold_preds
                best_labels = fold_labels
                best_preds = np.argmax(current_fold_preds, axis=1)
            else:
                epochs_no_improve += 1
                print(f"鈿狅笍 楠岃瘉闆?Loss 鏈檷浣?({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"馃洃 杩炵画 {EARLY_STOP_PATIENCE} 杞湭涓嬮檷锛岃Е鍙戞棭鍋滐紒")
                    break
        print("馃摳 姝ｅ湪鐢熸垚娣锋穯鐭╅樀...")
        plot_confusion_matrix(best_labels, best_preds, fold, base_dir)
        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
    print("\n馃帀 鎵€鏈?Fold 璁粌瀹屾瘯锛佷繚瀛?OOF 棰勬祴缁撴灉...")
    np.save(os.path.join(base_dir, "oof_preds_beit.npy"), oof_preds)
    final_labels = full_df['label_int'].values
    final_log_loss = log_loss(final_labels, oof_preds)
    print(f"馃敟 Final OOF Log Loss (BEiT-Large): {final_log_loss:.4f}")
if __name__ == "__main__":
    main()

# %% Notebook cell 3
import os
import torch
import pandas as pd
import numpy as np
import timm
from tqdm.auto import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
# ==========================================
# 鈿欙笍 1. 鍩虹閰嶇疆 (BEiT-Large 宸ㄥ吔鐗?
# ==========================================
# 馃敟 5090 涓撳睘鍔犻€熼瓟娉曪細寮€鍚?TF32 鏍稿績
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
FOLDS = [0, 1, 2, 3, 4]
SAMPLE_SUBMISSION_PATH = './dataset/sample_submission.csv'
TEST_DIR = 'dataset/imgs/test_cropped_v2'
# 鉁?BEiT 涓撳睘璺緞涓庢ā鍨嬪悕绉?
SAVE_DIR = 'models/beit'
MODEL_NAME = 'beit_large_patch16_224.in22k_ft_in22k_in1k'
WEIGHT_PATH_TEMPLATE = os.path.join(SAVE_DIR, 'best_model_beit_fold_{}.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 鈿狅笍 鏋佸叾鍏抽敭锛氬昂瀵稿繀椤绘槸 224
IMG_SIZE = 224      
# BEiT-Large 鍙傛暟閲忛珮杈?307M锛岀函鎺ㄧ悊鏃?5090 鏄惧瓨寰堝ぇ锛屼絾涓轰簡闃茬垎锛岃涓?64 鎴?128
BATCH_SIZE = 128    
NUM_WORKERS = 8
# ==========================================
# 馃柤锔?2. 鏋勫缓娴嬭瘯闆?DataLoader (鎸?CSV 涓ユ牸淇濆簭)
# ==========================================
class TestDriverDataset(Dataset):
    def __init__(self, csv_path, test_dir):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"馃毃 鎵句笉鍒板畼鏂规彁浜ゆā鏉? {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.test_dir = test_dir
        
        self.image_names = self.df['img'].values
        # 鈿狅笍 鏋佸叾鍏抽敭锛欱EiT 涓撳睘鐨勫綊涓€鍖栧潎鍊煎拰鏂瑰樊锛岀粷瀵逛笉鑳介敊锛?
        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.test_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            fallback_path = os.path.join('./dataset/imgs/test', img_name)
            image = cv2.imread(fallback_path)
            if image is None:
                raise FileNotFoundError(f"鎵句笉鍒板浘鐗? {img_path}")
                
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image=image)['image']
        return image_tensor, img_name
# ==========================================
# 馃殌 3. 鏍稿績棰勬祴涓?5鎶樿瀺鍚?(Ensemble)
# ==========================================
def generate_submission():
    print(f"馃殌 鍚姩 BEiT-Large 棰勬祴娴佺▼锛佷娇鐢ㄨ澶? {DEVICE}")
    test_dataset = TestDriverDataset(SAMPLE_SUBMISSION_PATH, TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Loaded {len(test_dataset)} test images in CSV order.")
    all_fold_preds = []
    for fold in FOLDS:
        weight_path = WEIGHT_PATH_TEMPLATE.format(fold)
        print(f"\n馃懆鈥嶁殩锔?姝ｅ湪鍑嗗绗?{fold} 鍙?BEiT 妯″瀷...")
        if not os.path.exists(weight_path):
            print(f"鈿狅笍 鎵句笉鍒版潈閲嶆枃浠?{weight_path}锛岃烦杩囨鎶橈紒")
            continue
        # 1. 瀹炰緥鍖栧共鍑€鐨勭┖妯″瀷
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=10)
        model.to(DEVICE)
        
        # 2. 鍔犺浇璁粌濂界殑鏉冮噸
        state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        # 3. 缂栬瘧鍔犻€?(BEiT 杩欑澶?Transformer 鏋佸叾鍚冭繖涓姞閫?
        if int(torch.__version__.split('.')[0]) >= 2:
            print("鈿?鍚敤 torch.compile 鍔犻€熷ぇ妯″瀷鎺ㄧ悊...")
            model = torch.compile(model)
        fold_preds = []
        with torch.no_grad():
             # 浣跨敤 AMP 鍗婄簿搴︽帹鐞嗭紝鐪佹樉瀛樻彁閫?
             with torch.amp.autocast('cuda'):
                pbar = tqdm(test_loader, desc=f"Fold {fold} predicting")
                for images, _ in pbar:
                    images = images.to(DEVICE)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    fold_preds.append(probs.cpu().numpy())
        fold_preds = np.concatenate(fold_preds, axis=0)
        all_fold_preds.append(fold_preds)
        
        del model
        torch.cuda.empty_cache()
    if not all_fold_preds:
        print("No valid fold weights were found; inference stopped.")
        return
    # ==========================================
    # 馃 4. 缁堟瀬骞冲潎铻嶅悎鐢熸垚鎻愪氦鏂囦欢
    # ==========================================
    print(f"\n馃 姝ｅ湪瀵规壘鍒扮殑 {len(all_fold_preds)} 鎶?BEiT 妯″瀷杩涜铻嶅悎...")
    all_fold_preds = np.array(all_fold_preds) 
    final_preds = np.mean(all_fold_preds, axis=0) 
    print("馃摑 姝ｅ湪鐢熸垚 BEiT 鐨?CSV 鎻愪氦鏂囦欢...")
    img_filenames = test_dataset.image_names
    df_submit = pd.DataFrame(final_preds, columns=[f'c{i}' for i in range(10)])
    df_submit.insert(0, 'img', img_filenames) 
    submit_filename = os.path.join(SAVE_DIR, 'submission_beit_5fold.csv')
    df_submit.to_csv(submit_filename, index=False)
    print(f"馃帀 澶у姛鍛婃垚锛丅EiT 棰勬祴鏂囦欢宸蹭繚瀛樹负: {submit_filename}")
if __name__ == '__main__':
    generate_submission()

# %% Notebook cell 4
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
IMG_SIZE = 224  # BEiT 灏哄
BATCH_SIZE = 128
FOLDS = [0, 1, 2, 3, 4]  # 馃憟 璁惧畾浣犺鎻愬彇鐨勬姌鏁?
# 鍔犺浇娴嬭瘯闆嗚矾寰?
df_test = pd.read_csv(SAMPLE_SUBMISSION_PATH)
image_names = df_test['img'].values
# BEiT 涓撳睘褰掍竴鍖?
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
print("馃殌 鍑嗗鎻愬彇 BEiT 5鎶橀骞茬壒寰?..")
test_loader = DataLoader(TestFeatureDataset(), batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
# 鐢ㄤ簬瀛樻斁 5 鎶樼壒寰佺殑鍒楄〃
all_fold_features = []
for fold in FOLDS:
    print(f"\n馃懆鈥嶁殩锔?姝ｅ湪鍔犺浇 BEiT Fold {fold} 妯″瀷...")
    
    # 1. 姣忔寰幆鍒涘缓涓€涓共鍑€鐨勬ā鍨?
    model = timm.create_model('beit_large_patch16_224.in22k_ft_in22k_in1k', pretrained=False, num_classes=10)
    
    # 2. 璇诲彇瀵瑰簲鎶樼殑鏉冮噸
    weight_path = f'models/beit/best_model_beit_fold_{fold}.pth'
    if not os.path.exists(weight_path):
        print(f"鈿狅笍 鎵句笉鍒版潈閲?{weight_path}锛岃烦杩囨鎶橈紒")
        continue
        
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    # 3. 鍒囬櫎鍒嗙被澶达紒
    model.reset_classifier(0)
    model.to(DEVICE)
    model.eval()
    # 缂栬瘧鍔犻€?(濡傛灉鏀寔)
    if int(torch.__version__.split('.')[0]) >= 2:
        model = torch.compile(model)
    fold_features = []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for images in tqdm(test_loader, desc=f"馃拵 鎻愬彇 Fold {fold} 鐗瑰緛"):
                features = model(images.to(DEVICE))
                fold_features.append(features.cpu().numpy())
    # 鎷兼帴褰撳墠鎶樼殑鎵€鏈夋壒娆＄壒寰?-> 褰㈢姸 (79726, 1024)
    current_fold_final_features = np.concatenate(fold_features, axis=0)
    all_fold_features.append(current_fold_final_features)
    
    # 娓呯悊鏄惧瓨锛岄槻姝?5 鎶樿窇涓嬫潵鐖嗘樉瀛?
    del model
    torch.cuda.empty_cache()
if not all_fold_features:
    raise ValueError("馃毃 娌℃湁鎴愬姛鎻愬彇鍒颁换浣曠壒寰侊紒")
# 馃幆 榄旀硶锛氬 5 鎶樼殑鐗瑰緛鐭╅樀姹傜畻鏈钩鍧囷紒
print("\n馃 姝ｅ湪瀵?5 鎶樼壒寰佽繘琛岀畻鏈钩鍧囪瀺鍚?..")
all_fold_features = np.array(all_fold_features) # 褰㈢姸: (5, 79726, 1024)
final_avg_features = np.mean(all_fold_features, axis=0) # 褰㈢姸: (79726, 1024)
# 瑕嗙洊淇濆瓨涓哄崟涓€鐨?npy 鏂囦欢
save_path = 'models/test_features_beit.npy'
np.save(save_path, final_avg_features)
print(f"鉁?BEiT 5鎶樼壒寰佽瀺鍚堝畬姣曪紒鐭╅樀褰㈢姸: {final_avg_features.shape}")
print(f"馃搧 宸插畨鍏ㄤ繚瀛樿嚦: {save_path}")

