 #predict_sample.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# ==== 从 train_clcd.py 中复制的模型定义 (仅复制需要的部分) ====
# --- U-Net 相关组件 ---
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- 模型定义 ---
class UNet(torch.nn.Module):
    def __init__(self, n_channels=6, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class CNN_LSTM_Model_Simple(torch.nn.Module):
    def __init__(self, in_channels=6, n_classes=1, base_channels=32):
        super(CNN_LSTM_Model_Simple, self).__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        self.down3 = Down(base_channels*4, base_channels*8)
        self.down4 = Down(base_channels*8, base_channels*8) # 瓶颈层

        # --- 直接上采样，不经过 LSTM ---
        self.up1 = Up(base_channels*16, base_channels*4) # 注意拼接后的通道数
        self.up2 = Up(base_channels*8, base_channels*2)
        self.up3 = Up(base_channels*4, base_channels)
        self.up4 = Up(base_channels*2, base_channels)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # --- 跳过 LSTM, 直接上采样 ---
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
# ==== 模型定义结束 ====

# ==== 配置 ====
# 数据路径
TEST_T1_DIR = "data/CLCD/test/time1"
TEST_T2_DIR = "data/CLCD/test/time2"
LABEL_DIR   = "data/CLCD/test/label"

# --- 修改这里来选择要测试的模型 ---
# MODEL_TYPE = "UNet"
MODEL_TYPE = "CNN_LSTM_Model_Simple" # <--- 根据你训练的模型修改 ---
# MODEL_TYPE = "CNN_LSTM_Model" # 如果你修复并训练了完整版，也可以测试它
# ----

# 根据 MODEL_TYPE 设置模型路径和实例化模型
if MODEL_TYPE == "UNet":
    MODEL_PATH = "models/clcd_unet_*.pth" # 请替换为实际的 .pth 文件名
    # 实例化模型
    model = UNet(n_channels=6, n_classes=1)
elif MODEL_TYPE == "CNN_LSTM_Model_Simple":
    MODEL_PATH = "models/clcd_cnn_lstm_simple.pth" # 请确保路径正确
    # 实例化模型
    model = CNN_LSTM_Model_Simple(in_channels=6, n_classes=1, base_channels=32)
elif MODEL_TYPE == "CNN_LSTM_Model":
    MODEL_PATH = "models/clcd_cnn_lstm.pth" # 请确保路径正确
    # 实例化模型 (需要复制完整版CNN_LSTM_Model的定义到上面)
    # model = CNN_LSTM_Model(...)
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

print(f"[INFO] Selected Model Type: {MODEL_TYPE}")
print(f"[INFO] Model Path: {MODEL_PATH}")
# ==== 配置结束 ====

# 图像预处理
tf_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_img(path):
    return Image.open(path).convert("RGB")

# 自动随机选择一个样本
all_files = os.listdir(TEST_T1_DIR)
if not all_files:
    raise FileNotFoundError("No files found in the test directory.")
sample_id = random.choice(all_files)
t1_path = os.path.join(TEST_T1_DIR, sample_id)
t2_path = os.path.join(TEST_T2_DIR, sample_id)
label_path = os.path.join(LABEL_DIR, sample_id)

print(f"[INFO] Randomly selected sample: {sample_id}")

# 加载图像
try:
    t1 = tf_img(load_img(t1_path))
    t2 = tf_img(load_img(t2_path))
    label = tf_img(Image.open(label_path).convert("L"))  # 确保标签是单通道
except Exception as e:
    print(f"[ERROR] Failed to load images for {sample_id}: {e}")
    exit(1)

# 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 加载模型权重
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"[ERROR] Model file not found at {MODEL_PATH}. Please check the path.")
    exit(1)
except Exception as e:
    print(f"[ERROR] Failed to load model state dict: {e}")
    exit(1)

model.eval()

# 构造输入 (将t1和t2在通道维度上连接)
x = torch.cat([t1, t2], dim=0).unsqueeze(0).to(device)  # (1, 6, H, W)

with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits)
    # 计算变化区域占比来判断是否有变化
    change_ratio = (prob > 0.5).float().mean().item()
    pred_class = change_ratio > 0.1  # 如果变化区域超过10%，则判断为有变化

print(f"[RESULT] Prediction: {pred_class} (False=No Change, True=Change)")
print(f"[INFO] Change Ratio: {change_ratio:.4f}")

# 可视化
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(t1.permute(1, 2, 0))
axs[0].set_title("Time 1")
axs[1].imshow(t2.permute(1, 2, 0))
axs[1].set_title("Time 2")
axs[2].imshow(label[0], cmap="gray")
axs[2].set_title("Ground Truth")
axs[3].imshow(prob[0, 0].cpu(), cmap="gray") # prob shape is (1, 1, H, W)
axs[3].set_title("Prediction (Prob Map)")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()
