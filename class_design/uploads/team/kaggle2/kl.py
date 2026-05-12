# Auto-converted from kl.ipynb.
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
        "Use train_with_pseudo.py/run_transformer_system.py, or set "
        "RUN_LEGACY_NOTEBOOK_EXPORT=1 to run the historical export."
    )


# %% Notebook cell 1
if not os.path.exists("effb0_student.safetensors"):
    raise FileNotFoundError("Missing local pretrained weights: effb0_student.safetensors")

# %% Notebook cell 2
import os
import gc
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from transformers import get_cosine_schedule_with_warmup
# ==========================================
# ⚙️ 1. 全局配置参数 (轻量级学生模型)
# ==========================================
# 老师的最强融合预测文件（软标签来源）
TEACHER_CSV_PATH = 'models/final_magic_knn_submission.csv' 
TEST_DIR = 'dataset/imgs/test_cropped_v2'  # 在 7.9万张测试集上进行蒸馏
# 选择一个身手轻捷的学生模型 (EffB0 跑得极快，泛化极强)
STUDENT_MODEL_NAME = 'tf_efficientnet_b0.ns_jft_in1k'
IMG_SIZE = 224
BATCH_SIZE = 128    # 小模型显存占用低，Batch可以开很大
EPOCHS = 5          # 蒸馏收敛非常快，5轮足够
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ==========================================
# 🧮 2. 定义神仙损失函数 (KL 散度)
# ==========================================
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    def forward(self, student_logits, teacher_probs):
        # 学生输出除以温度后算 LogSoftmax
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # 老师的概率加上温度进行软化 (Smooth)
        teacher_soft = torch.pow(teacher_probs, 1.0 / self.temperature)
        teacher_soft = teacher_soft / teacher_soft.sum(dim=1, keepdim=True)
        
        # KL 散度乘以温度的平方 (梯度对齐)
        return self.kl_loss(student_log_probs, teacher_soft) * (self.temperature ** 2)
# ==========================================
# 🖼️ 3. 数据集提取 (专门吐出软标签)
# ==========================================
def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
class DistillationDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.prob_cols = [f'c{i}' for i in range(10)]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['img'])
        
        image = cv2.imread(img_path)
        if image is None:  # 防御性回退
            image = cv2.imread(os.path.join('dataset/imgs/test', row['img']))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
            
        # 👑 提取 10 维的软标签概率
        soft_label = row[self.prob_cols].values.astype(np.float32)
        return image, torch.tensor(soft_label)
# ==========================================
# 🚀 4. 主干蒸馏流程
# ==========================================
def main():
    print("📚 正在加载老师的软标签数据集...")
    teacher_df = pd.read_csv(TEACHER_CSV_PATH)
    
    # 构建 DataLoader
    train_loader = DataLoader(
        DistillationDataset(teacher_df, TEST_DIR, transform=get_transforms(IMG_SIZE)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )
    print(f"🐣 正在初始化学生模型: {STUDENT_MODEL_NAME} ...")
    # 1. ⚠️ 注意这里要把 pretrained 改为 False，防止它自动联网
    model = timm.create_model(STUDENT_MODEL_NAME, pretrained=False, num_classes=10)
    
    # 2. 引入 safetensors 读取工具
    from safetensors.torch import load_file
    
    print("📥 正在手工加载本地学生模型预训练权重 (effb0_student.safetensors)...")
    state_dict = load_file("effb0_student.safetensors")
    
    # 3. 剥离原权重中的分类头 (防止 1000类 vs 10类 形状冲突)
    for key in list(state_dict.keys()):
        if key.startswith('classifier.'):
            del state_dict[key]
            
    # 4. 安全加载进模型
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    # 损失函数与优化器
    criterion = DistillationLoss(temperature=3.0) # 温度设为 3.0，极致平滑
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    print("\n" + "="*40 + "\n🔥 软标签蒸馏正式开始 🔥\n" + "="*40)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Distilling]")
        
        for images, soft_labels in pbar:
            images, soft_labels = images.to(DEVICE), soft_labels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                student_logits = model(images)
                loss = criterion(student_logits, soft_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            pbar.set_postfix({'KL_Loss': f"{loss.item():.4f}"})
    print("🎉 学生模型蒸馏完毕！开始输出学生的预测概率...")
    
    # ==========================================
    # 🧪 5. 生成学生的预测并与老师调和
    # ==========================================
    model.eval()
    
    # 重新构建不 shuffle 的 DataLoader 用于按顺序输出预测
    test_loader = DataLoader(
        DistillationDataset(teacher_df, TEST_DIR, transform=get_transforms(IMG_SIZE)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=8
    )
    
    student_preds = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Student Inference"):
            images = images.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(images)
                probs = F.softmax(logits, dim=1)
            student_preds.append(probs.cpu().numpy())
            
    student_preds = np.concatenate(student_preds, axis=0)
    
    # 保存纯净版学生预测 (备用)
    student_df = teacher_df.copy()
    cols = [f'c{i}' for i in range(10)]
    student_df[cols] = student_preds
    student_df.to_csv('submission_student_only.csv', index=False)
    
    # 👑 黄金调和：老师 85% + 学生 15% (绝对的防爆盾)
    print("\n✨ 正在执行师生调和魔法...")
    teacher_preds = teacher_df[cols].values
    
    final_probs = (teacher_preds * 0.85) + (student_preds * 0.15)
    
    teacher_df[cols] = final_probs
    output_path = 'models/KAGGLE_GOLD_SUBMISSION.csv'
    teacher_df.to_csv(output_path, index=False)
    
    print(f"\n🏆 全剧终！终极调和文件已保存至: {output_path}")
    print("🥇 拿去提交 Kaggle，准备迎接断崖式提分吧！")
if __name__ == "__main__":
    main()

# %% Notebook cell 3
import pandas as pd
import numpy as np
# 1. 读取你那份最牛的 0.15033 的 CSV 文件
BEST_SUB_PATH = 'models/final_magic_knn_submission.csv' # 替换为你的文件名
sub_df = pd.read_csv(BEST_SUB_PATH)
cols = [f'c{i}' for i in range(10)]
probs = sub_df[cols].values
# 2. 👑 核心魔法：概率锐化 (Temperature Scaling)
# ALPHA > 1.0 会让模型变得更自信（马太效应：富者越富，强者越强）
# 建议尝试 1.1, 1.2 和 1.3 这三个档位
ALPHA = 1.15  
# 进行指数放大
sharpened_probs = probs ** ALPHA
# 重新归一化，保证每行概率和为 1
sharpened_probs = sharpened_probs / sharpened_probs.sum(axis=1, keepdims=True)
# 3. 极其微小的底线防爆 (极其自信的同时，依然防止无限大的 Loss)
CLIP_MIN = 1e-6 # 这里一定要设得很小，比如 1e-6 甚至 1e-7
sharpened_probs = np.clip(sharpened_probs, CLIP_MIN, 1.0 - CLIP_MIN)
sharpened_probs = sharpened_probs / sharpened_probs.sum(axis=1, keepdims=True)
# 保存提交
sub_df[cols] = sharpened_probs
sub_df.to_csv('submission_sharpened_1.15.csv', index=False)
print("🎉 锐化完成！生成 submission_sharpened_1.15.csv")

