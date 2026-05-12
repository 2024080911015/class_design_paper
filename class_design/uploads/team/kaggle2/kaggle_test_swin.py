# Auto-converted from kaggle_test_swin.ipynb.
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
# 寮€鍚?AutoDL 鐨勫鏈姞閫燂紙涓撻棬鍔犻€?Github 鍜?Kaggle锛?
run_shell("source /etc/network_turbo \u0026\u0026 kaggle competitions download -c state-farm-distracted-driver-detection")
# 濡傛灉瑕佸叧闂姞閫燂紝鍙互浣跨敤锛?
# !source /etc/network_turbo_disable

# %% Notebook cell 2
# 1. 瀹夎鏍稿績渚濊禆锛堜慨姝ｄ簡 grad-cam 鐨勫寘鍚嶏級
run_shell("pip install -q kaggle ultralytics timm albumentations grad-cam pandas seaborn transformers")
# 2. 閰嶇疆 Kaggle 绉橀挜
run_shell("mkdir -p ~/.kaggle")
run_shell("mv kaggle.json ~/.kaggle/")
run_shell("chmod 600 ~/.kaggle/kaggle.json")
# 3. 鍒涘缓鏁版嵁闆嗙洰褰曞苟涓嬭浇
run_shell("mkdir -p ./dataset")
os.chdir("./dataset")
run_shell("kaggle competitions download -c state-farm-distracted-driver-detection")
# 4. 闈欓粯瑙ｅ帇骞舵竻鐞嗗帇缂╁寘
run_shell("unzip -q state-farm-distracted-driver-detection.zip")
run_shell("rm state-farm-distracted-driver-detection.zip")
os.chdir("..")
print("鉁?鐜閰嶇疆涓庢暟鎹笅杞藉畬姣曪紒")

# %% Notebook cell 3
import torch
print("=========================================")
print("          馃枼锔?娣卞害瀛︿範鐜浣撴鎶ュ憡          ")
print("=========================================")
print(f"馃摝 PyTorch 鐗堟湰: {torch.__version__}")
print(f"馃殌 CUDA 鏄惁鍙敤: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"馃槑 璇嗗埆鍒扮殑鏄惧崱: {torch.cuda.get_device_name(0)}")
    print(f"鈿欙笍 褰撳墠 CUDA 鐗堟湰: {torch.version.cuda}")
    print(f"馃 鏄惧瓨澶у皬: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: PyTorch cannot use GPU. Please check the CUDA driver.")
print("=========================================")

# %% Notebook cell 4
import os
import cv2
from tqdm.auto import tqdm
from ultralytics import YOLO
import numpy as np
SOURCE_DIR = './dataset/imgs/train'
TARGET_DIR = './dataset/imgs/train_cropped_v2'
print("馃殌 姝ｅ湪鍔犺浇 YOLOv8n 妫€娴嬫ā鍨?..")
model = YOLO('yolov8n.pt')
for c in range(10):
    os.makedirs(os.path.join(TARGET_DIR, f'c{c}'), exist_ok=True)
all_images = []
for c in range(10):
    class_dir = os.path.join(SOURCE_DIR, f'c{c}')
    if not os.path.exists(class_dir): continue
    for img_name in os.listdir(class_dir):
        all_images.append((f'c{c}', img_name))
print(f"馃搳 鍏辨壘鍒?{len(all_images)} 寮犲浘鐗囷紝5090 鍑嗗灏辩华锛屽紑濮嬫瀬閫熻鍓?..")
no_detect_count = 0
for class_name, img_name in tqdm(all_images, desc="瑁佸壀杩涘害"):
    img_path = os.path.join(SOURCE_DIR, class_name, img_name)
    save_path = os.path.join(TARGET_DIR, class_name, img_name)
    if os.path.exists(save_path): continue
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    # 璁?GPU 0 (浣犵殑 5090) 鍙備笌鎺ㄧ悊
    results = model.predict(img, classes=[0], conf=0.3, verbose=False, device=0)
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = np.argmax(areas)
        x1, y1, x2, y2 = boxes[best_idx].astype(int)
        person_h = y2 - y1
        new_y2 = y1 + int(person_h * 0.65)
        pad_x = int((x2 - x1) * 0.15) 
        pad_y_top = int(person_h * 0.05)   
        crop_x1 = max(0, x1) 
        crop_y1 = max(0, y1 - pad_y_top)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, new_y2) 
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
             crop_img = img[y1:y2, x1:x2]
        else:
             crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        final_img = cv2.resize(crop_img, (384, 384), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_path, final_img)
    else:
        no_detect_count += 1
        final_img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_path, final_img)
print(f"\nYOLO crop finished.")

# %% Notebook cell 5
import subprocess
import os
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# %% Notebook cell 6
#蹇€熻皟璇曚唬鐮?
import os
import ssl
import requests
# 绂佺敤 SSL 璇佷功楠岃瘉锛堢敤浜庡鏈姞閫熺殑鑷鍚嶈瘉涔︼級
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
requests.packages.urllib3.disable_warnings()
import os
import cv2
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import log_loss, confusion_matrix
import timm
from transformers import get_cosine_schedule_with_warmup
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# ==========================================
# 鈿欙笍 1. 鍏ㄥ眬閰嶇疆鍙傛暟 (A100 浼樺寲鐗?
# ==========================================
CSV_PATH = 'dataset/driver_imgs_list.csv'
TRAIN_DIR = 'dataset/imgs/train_cropped_v2' # 鍘昏儗鏅悗鐨勫浘鐗囬泦
MODEL_NAME = 'swin_base_patch4_window12_384.ms_in22k'
IMG_SIZE = 384
EPOCHS = 10         # 鉁?姝ｅ紡璁粌锛岃涓?10 杞?(閰嶅悎鏃╁仠鏈哄埗)
BATCH_SIZE = 32    # 鉁?A100 鏄惧瓨鏋佸ぇ锛岀墿鐞?Batch Size 鎻愬崌鍒?32
ACCUMULATION_STEPS = 1  # 鉁?涓嶉渶瑕侀绻佹搴︾疮鍔狅紝鐩存帴鐪熷疄鏇存柊
NUM_SPLITS = 5
TARGET_FOLDS = [0,1,2,3,4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_swin_pretrained_weights():
    if not os.path.exists("swin_base.safetensors"):
        raise FileNotFoundError("Missing local pretrained weights: swin_base.safetensors")
# 娓呯悊鍐呭瓨鐜
gc.collect()
torch.cuda.empty_cache()
# ==========================================
# 馃搳 2. 鏁版嵁鍒掑垎 (瀹屽叏澶嶅埢 Eval_Pipeline 璐績闅旂绛栫暐)
# ==========================================
def generate_balanced_folds(csv_path, n_splits=5):
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)
    if 'label_int' not in df.columns:
        df['label_int'] = df['classname'].str.extract(r'(\d+)').astype(int)
    # 璐績绠楁硶閫昏緫锛氭寜姣忎釜椹鹃┒鍛樼殑鍥剧墖鏁颁粠澶у埌灏忔帓搴?
    driver_counts = df.groupby('subject').size().sort_values(ascending=False)
    fold_totals = np.zeros(n_splits)
    fold_groups = [[] for _ in range(n_splits)]
    # 姣忔灏嗘渶澶х殑椹鹃┒鍛樺垎閰嶇粰褰撳墠鍥剧墖鏁版渶灏戠殑 Fold
    for subject, count in driver_counts.items():
        min_fold_idx = np.argmin(fold_totals)
        fold_groups[min_fold_idx].append(subject)
        fold_totals[min_fold_idx] += count
    # 鏄犲皠鍥炲師 DataFrame
    df['fold'] = -1
    for i, subjects in enumerate(fold_groups):
        df.loc[df['subject'].isin(subjects), 'fold'] = i
    # 鎵撳嵃闅旂涓庡钩琛℃€ф鏌ユ姤鍛?
    print("\n" + "="*50)
    print("           5-fold data distribution")
    print("="*50)
    for i in range(n_splits):
        num_driver = len(fold_groups[i])
        num_img = int(fold_totals[i])
        print(f"Fold {i}  |  椹鹃┒鍛樻暟閲? {num_driver:2d}  |  鍥剧墖鎬绘暟: {num_img:4d}")
    print("="*50)
    print(f"鉁?鏈€澶ф牱鏈亸宸? {int(fold_totals.max() - fold_totals.min())} 寮犲浘\n")
    output_path = "train_with_folds.csv"
    df.to_csv(output_path, index=False)
    print("\n鉁?鍒掑垎瀹屾垚: train_with_folds.csv\n")
    return df
# ==========================================
# 馃柤锔?3. 鏁版嵁闆嗕笌澧炲己
# ==========================================
def get_train_transforms(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.6),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-10, 10), p=0.5),
        A.GaussNoise(p=0.4),
        #A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 16), hole_width_range=(1, 16), fill=0, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
def get_valid_transforms(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            # 闃插尽鎬у洖閫€锛氬鏋滄病鎵惧埌鎶犲浘鍚庣殑鏂囦欢锛屽幓鍘熷浘閲屾嬁
            fallback_path = os.path.join('dataset/imgs/train', row['classname'], row['img'])
            image = cv2.imread(fallback_path)
            if image is None:
                raise FileNotFoundError(f"鍥剧墖涓㈠け: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, row['label_int']
# ==========================================
# 馃洜锔?4. 璇勪及宸ュ叿 (GradCAM & 娣锋穯鐭╅樀)
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
def generate_grad_cam(model, val_loader, fold_idx, save_dir):
    model.eval()
    try:
        images, labels = next(iter(val_loader))
    except StopIteration:
        return
    target_layer = model.layers[-1].blocks[-1].norm1
    def reshape_transform(tensor, height=12, width=12):
        if tensor.ndim == 4:
            return tensor.permute(0, 3, 1, 2)
        elif tensor.ndim == 3:
            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
            return result.permute(0, 3, 1, 2)
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    input_tensor = images[0:1].to(DEVICE)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    img_np = images[0].permute(1, 2, 0).cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_np = np.clip(std * img_np + mean, 0, 1)
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    plt.imsave(os.path.join(save_dir, f"grad_cam_fold_{fold_idx}.png"), cam_image)
# ==========================================
# 馃殌 5. 涓诲共璁粌娴佺▼
# ==========================================
def main():
    ensure_swin_pretrained_weights()
    base_dir = 'models'
    os.makedirs(base_dir, exist_ok=True)
    full_df = generate_balanced_folds(CSV_PATH, NUM_SPLITS)
    oof_preds = np.zeros((len(full_df), 10))
    for fold in TARGET_FOLDS:
        print(f"\n{'='*40}\n馃専 寮€濮嬭缁?Fold {fold} 馃専\n{'='*40}")
        train_df = full_df[full_df['fold'] != fold]
        val_df = full_df[full_df['fold'] == fold]
        train_loader = DataLoader(
            DriverDataset(train_df, TRAIN_DIR, transform=get_train_transforms(IMG_SIZE)),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True # 鉁?num_workers=4 鎻愰€?
        )
        val_loader = DataLoader(
            DriverDataset(val_df, TRAIN_DIR, transform=get_valid_transforms(IMG_SIZE)),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True # 鉁?num_workers=4 鎻愰€?
        )
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=10, drop_path_rate=0.3)
    
    # 2. 浠庢湰鍦版墜鍔ㄨ鍙栧垰鍒氫笅杞界殑鏉冮噸
        from safetensors.torch import load_file
        state_dict = load_file("swin_base.safetensors")
        
        # 3. 鍓ョ鍘熸潈閲嶄腑鐨勫垎绫诲ご锛堝師妯″瀷鏄?2涓囩被锛屾垜浠彧瑕?10 绫伙紝闃叉褰㈢姸鍐茬獊鎶ラ敊锛?
        for key in list(state_dict.keys()):
            if key.startswith('head.'):
                del state_dict[key]
                
        # 4. 鎶婄函鍑€鐨勬潈閲嶅杩涙ā鍨嬮噷
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        class_weights=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],dtype=torch.float).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights,label_smoothing=0.1)
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
        scaler = torch.amp.GradScaler('cuda')
        total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        best_val_loss = float('inf')
        save_path = os.path.join(base_dir, f"best_model_swin_fold_{fold}.pth")
        EARLY_STOP_PATIENCE = 2
        epochs_no_improve = 0
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
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
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                pbar.set_postfix({'loss': f"{loss.item()*ACCUMULATION_STEPS:.4f}"})
            model.eval()
            val_loss = 0.0
            fold_preds_list = []
            fold_labels = []
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
                epochs_no_improve = 0  # 閲嶇疆璁℃暟鍣?
                oof_preds[val_df.index] = current_fold_preds
                best_labels = fold_labels
                best_preds = np.argmax(current_fold_preds, axis=1)
            else:
                epochs_no_improve += 1
                print(f"鈿狅笍 楠岃瘉闆?Loss 鏈檷浣?({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"馃洃 杩炵画 {EARLY_STOP_PATIENCE} 杞?Loss 鏈笅闄嶏紝瑙﹀彂鏃╁仠鏈哄埗 (Early Stopping)锛屾彁鍓嶇粨鏉熸湰鎶樿缁冿紒")
                    break
        print("馃摳 姝ｅ湪鐢熸垚娣锋穯鐭╅樀涓?Grad-CAM 娉ㄦ剰鍔涚儹鍔涘浘...")
        plot_confusion_matrix(best_labels, best_preds, fold, base_dir)
        model.load_state_dict(torch.load(save_path))
        generate_grad_cam(model, val_loader, fold, base_dir)
        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()
    print("\n馃帀 鎵€鏈?Fold 璁粌瀹屾瘯锛佷繚瀛?OOF 棰勬祴缁撴灉...")
    np.save(os.path.join(base_dir, "oof_preds_swin.npy"), oof_preds)
    final_labels = full_df['label_int'].values
    final_log_loss = log_loss(final_labels, oof_preds)
    print(f"馃敟 Final OOF Cross-Validation Log Loss: {final_log_loss:.4f}")
if __name__ == "__main__":
    main()
    # 鉁?璁粌瀹屾垚鍚庤嚜鍔ㄦ柇寮€瀹炰緥浠ヨ妭鐪佺畻鍔?
    print("\n馃洃 璁粌鍏ㄩ儴缁撴潫锛屾ā鍨嬪凡淇濆瓨鑷?Google Drive銆傛鍦ㄨ嚜鍔ㄦ柇寮€瀹炰緥...")
   

# %% Notebook cell 7
import os
import timm
ensure_swin_pretrained_weights()

# %% Notebook cell 8
import subprocess
import os
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# %% Notebook cell 9
# --no-check-certificate 杩欎釜鍙傛暟鏄牳姝﹀櫒锛屽己琛屾棤瑙嗕换浣曡瘉涔︽嫤鎴紒
ensure_swin_pretrained_weights()
print("Local Swin pretrained weights found.")

# %% Notebook cell 10
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
# 鈿欙笍 1. 鍩虹閰嶇疆 (RTX 5090 鏈湴璺緞鐗?
# ==========================================
# 馃敟 5090 涓撳睘鍔犻€熼瓟娉曪細寮€鍚?TF32 鏍稿績
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
folds=[0,1,2,3,4]
# 涓ユ牸鎸夌収浣犳湰鍦扮殑鐩稿璺緞璁剧疆
SAMPLE_SUBMISSION_PATH = './dataset/sample_submission.csv'
TEST_DIR = 'dataset/imgs/test_cropped_v2' 
SAVE_DIR = './models'
# 妯″瀷涓庢潈閲嶉厤缃?
MODEL_NAME = 'swin_base_patch4_window12_384.ms_in22k'
WEIGHT_PATH_TEMPLATE = os.path.join(SAVE_DIR, 'best_model_swin_fold_{}.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
BATCH_SIZE = 128  # 5090 鏄惧瓨鏈?32G锛屾帹鐞嗕笉瀛樻搴︼紝鐩存帴寮€鍒?128 璧烽
NUM_WORKERS = 8
# ==========================================
# 馃柤锔?2. 鏋勫缓娴嬭瘯闆?DataLoader (鎸?CSV 椤哄簭涓ユ牸璇诲彇)
# ==========================================
class TestDriverDataset(Dataset):
    def __init__(self, csv_path, test_dir):
        # 馃専 鍏抽敭淇锛氱洿鎺ヨ鍙栧畼鏂规牱鏈彁浜ゆ枃浠讹紝涓ユ牸淇濊瘉椤哄簭涓€鑷存€?
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"馃毃 鎵句笉鍒板畼鏂规彁浜ゆā鏉? {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.test_dir = test_dir
        
        # 鎻愬彇鎵€鏈夌殑鍥剧墖鏂囦欢鍚?(褰㈠ img_1.jpg)
        self.image_names = self.df['img'].values
        # 娴嬭瘯闆嗙殑棰勫鐞嗗繀椤诲拰楠岃瘉闆嗕竴妯′竴鏍凤紒浠呭仛鏍囧噯鍖栵紝鏃犻殢鏈哄寮?
        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                 raise FileNotFoundError(f"Missing test image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 鈿狅笍 鍒犻櫎浜嗘墍鏈夌殑瑁佸壀閫昏緫锛岀洿鎺ユ妸鍘熷浘鍠傝繘鍘荤缉鏀?
        augmented = self.transform(image=image)
        image_tensor = augmented['image']
        return image_tensor, img_name
# ==========================================
# 馃殌 3. 鏍稿績棰勬祴涓?5鎶樿瀺鍚?(Ensemble) 閫昏緫
# ==========================================
def generate_submission():
    print(f"馃殌 鍚姩棰勬祴娴佺▼锛佷娇鐢ㄨ澶? {DEVICE}")
    # 1. 鍑嗗娴嬭瘯鏁版嵁
    test_dataset = TestDriverDataset(SAMPLE_SUBMISSION_PATH, TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Loaded {len(test_dataset)} test images in CSV order.")
    # 2. 鍑嗗绌哄鍣ㄥ瓨鏀鹃娴嬬粨鏋?
    all_fold_preds = []
    # 3. 瀹炰緥鍖栫┖妯″瀷澹冲瓙
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=10)
    model.to(DEVICE)
    model.eval()
    # 濡傛灉浣犺缁冩椂鐢ㄤ簡 torch.compile锛岃繖閲屾渶濂戒篃寮€鍚紝浠ヤ繚璇佹潈閲嶇粨鏋勭殑瀹岀編鍏煎
    if int(torch.__version__.split('.')[0]) >= 2: 
        print("鈿?鍚敤 torch.compile 鍔犻€熸帹鐞?..")
        model = torch.compile(model)
    # 4. 渚濇璇峰嚭 5 鎶樻ā鍨嬭繘琛岄娴?
    for fold in folds:
        # 1. 姣忔寰幆閲嶆柊鍒涘缓涓€涓共鍑€鐨勭┖妯″瀷
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=10)
        model.to(DEVICE)
        model.eval()
        weight_path = WEIGHT_PATH_TEMPLATE.format(fold)
        print(f"\n馃懆鈥嶁殩锔?姝ｅ湪鍔犺浇绗?{fold} 鍙锋潈閲?({weight_path})...")
        if not os.path.exists(weight_path):
            continue
        # 2. 璇诲彇鏉冮噸锛堢敱浜庤缁冩椂宸茬粡鍘绘帀浜嗗墠缂€锛岃繖閲岀洿鎺ヨ鍑烘潵鐨勫氨鏄共鍑€鐨勶級
        state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
        
        # 3. 馃幆 鍏抽敭淇锛氬厛鍔犺浇鏉冮噸锛屽苟涓斿幓鎺?strict=False 浠ラ槻涓囦竴
        model.load_state_dict(state_dict, strict=True) 
        
        # 4. 馃幆 鍏抽敭淇锛氭潈閲嶅畨鍏ㄨ濉畬姣曞悗锛屽啀杩涜鍥剧紪璇戝姞閫?
        if int(torch.__version__.split('.')[0]) >= 2:
            print("鈿?鍚敤 torch.compile 鍔犻€熸帹鐞?..")
            model = torch.compile(model)
        fold_preds = []
        with torch.no_grad():
             with torch.amp.autocast('cuda'):
                pbar = tqdm(test_loader, desc=f"Fold {fold} predicting")
                for images, _ in pbar:
                    images = images.to(DEVICE)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    fold_preds.append(probs.cpu().numpy())
        fold_preds = np.concatenate(fold_preds, axis=0)
        all_fold_preds.append(fold_preds)
    if not all_fold_preds:
        print("No valid fold weights were found; inference stopped.")
        return
    # 5. 缁堟瀬骞冲潎铻嶅悎
    print(f"\n馃 姝ｅ湪瀵规壘鍒扮殑 {len(all_fold_preds)} 涓ā鍨嬭繘琛屾鐜囧钩鍧囪瀺鍚?..")
    all_fold_preds = np.array(all_fold_preds) 
    final_preds = np.mean(all_fold_preds, axis=0) 
    # 6. 鐢熸垚 Kaggle 瑕佹眰鐨?submission.csv
    print("馃摑 姝ｅ湪鐢熸垚鏈€缁堢殑 CSV 鎻愪氦鏂囦欢...")
    
    # 鐩存帴澶嶇敤 DataLoader 閲岀殑鍑嗙‘椤哄簭
    img_filenames = test_dataset.image_names
    df_submit = pd.DataFrame(final_preds, columns=[f'c{i}' for i in range(10)])
    df_submit.insert(0, 'img', img_filenames) 
    submit_filename = os.path.join(SAVE_DIR, 'submission_swin_5fold_fixed.csv')
    df_submit.to_csv(submit_filename, index=False)
    print(f"馃帀 澶у姛鍛婃垚锛侀娴嬫枃浠跺凡瀹夊叏淇濆瓨涓? {submit_filename}")
if __name__ == '__main__':
    generate_submission()

