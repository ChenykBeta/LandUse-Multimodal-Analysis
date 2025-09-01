# train_clcd.py (V3: 集成多种模型和调试功能)
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 模型定义直接写在脚本中
from clcd_dataset import CLCDDataset # <--- 使用修改后的 CLCDDataset ---
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ==== 配置 ====
DATA_ROOT   = "data/CLCD"       # 请确保路径正确
IMG_SIZE    = 256               # 训练分辨率
BATCH_SIZE  = 4                 # 根据显存调整
EPOCHS      = 30
LR          = 1e-4              # 可尝试 5e-5, 1e-5
NUM_WORKERS = 0                 # Windows 兼容

# --- 通过修改此变量来切换模型 ---
# MODEL_TYPE = "UNet"
# MODEL_TYPE = "CNN_LSTM_Model"
#MODEL_TYPE = "CNN_LSTM_Model_Simple" # <--- 默认使用简化版进行调试 ---
MODEL_TYPE = "CNN_LSTM_Model"
# ----

if MODEL_TYPE == "UNet":
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_PATH   = f"models/clcd_unet_{timestamp}.pth"
    VIS_DIR     = f"vis_results_unet_{timestamp}"
elif MODEL_TYPE == "CNN_LSTM_Model":
    SAVE_PATH   = "models/clcd_cnn_lstm.pth"
    VIS_DIR     = "vis_results_cnn_lstm"
elif MODEL_TYPE == "CNN_LSTM_Model_Simple":
    SAVE_PATH   = "models/clcd_cnn_lstm_simple.pth"
    VIS_DIR     = "vis_results_cnn_lstm_simple"

os.makedirs("models", exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ==== 自定义 Transform 类 (用于处理图像对和标签) ====
class PairedTransform:
    """对图像对 (t1, t2) 和标签 (label) 应用同步变换"""
    def __init__(self, base_transform, augment_transform=None):
        self.base_transform = base_transform
        self.augment_transform = augment_transform

    def __call__(self, img1, img2, mask):
        # 1. 应用基础变换 (Resize, ToTensor) 到所有图像
        img1 = self.base_transform(img1)
        img2 = self.base_transform(img2)
        mask = self.base_transform(mask) # ToTensor 会自动将 L 转为 [1, H, W] 并归一化到 [0,1]

        # 2. 如果有增强变换，则应用到所有图像
        if self.augment_transform:
            if hasattr(self.augment_transform, 'transforms'):
                 non_geom_transforms = [t for t in self.augment_transform.transforms 
                                        if not isinstance(t, (transforms.RandomHorizontalFlip, 
                                                              transforms.RandomVerticalFlip, 
                                                              transforms.RandomRotation,
                                                              transforms.RandomAffine))]
                 if non_geom_transforms:
                     print("[WARNING] Non-geometric transforms found in augment_transform. Applying only to images.")
                     for t in non_geom_transforms:
                         img1 = t(img1)
                         img2 = t(img2)
                     geom_transforms = [t for t in self.augment_transform.transforms if t not in non_geom_transforms]
                     if geom_transforms:
                        geom_tf = transforms.Compose(geom_transforms)
                        cat_geom = torch.cat([img1, img2, mask], dim=0)
                        cat_geom = geom_tf(cat_geom)
                        c_img = img1.shape[0]
                        img1 = cat_geom[0:c_img, ...]
                        img2 = cat_geom[c_img:2*c_img, ...]
                        mask = cat_geom[2*c_img:, ...]
                 else:
                     cat_img = torch.cat([img1, img2, mask], dim=0)
                     cat_img = self.augment_transform(cat_img)
                     c_img = img1.shape[0]
                     img1 = cat_img[0:c_img, ...]
                     img2 = cat_img[c_img:2*c_img, ...]
                     mask = cat_img[2*c_img:, ...]
            else:
                 cat_img = torch.cat([img1, img2, mask], dim=0)
                 cat_img = self.augment_transform(cat_img)
                 c_img = img1.shape[0]
                 img1 = cat_img[0:c_img, ...]
                 img2 = cat_img[c_img:2*c_img, ...]
                 mask = cat_img[2*c_img:, ...]
            
        return img1, img2, mask

# ==== 数据增强 (训练时) ====
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

val_base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# 创建配对变换
train_transform = PairedTransform(base_transform, train_augment_transform)
val_transform = PairedTransform(val_base_transform, None) # 验证集不增强

# ==== 数据集 ====
train_set = CLCDDataset(DATA_ROOT, split="train", transform=train_transform)
val_set   = CLCDDataset(DATA_ROOT, split="val",   transform=val_transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ==== 模型定义 (全部写在这里) ====
# --- 1. 标准 U-Net ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
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

# --- 2. 原始 CNN_LSTM_Model ---
# --- 修复后的 CNN_LSTM_Model (在 train_clcd.py 中替换旧的 CNN_LSTM_Model 类) ---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels=6, hidden_size=64, num_layers=1, num_classes=1):
        super(CNN_LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN Encoder (类似 U-Net 的下采样部分)
        self.inc = DoubleConv(input_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512) # 输出 (B, 512, 8, 8)
        
        # LSTM
        # 输入图像大小 256x256, 经过5次下采样后为 8x8 (H_out = W_out = 8)
        # 我们将 (B, 512, 8, 8) 展平为 (B, 512, 64)，然后转置为 (B, 64, 512)
        # 这意味着序列长度 seq_len = 64, 每个时间步的输入维度 input_size = 512
        self.lstm_input_size = 512  # <--- 修复点 1: 正确的 input_size ---
        self.lstm_seq_len = 8 * 8   # <--- 明确定义序列长度 ---
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Decoder (上采样部分)
        # 使用 LSTM 最后一个时间步的隐藏状态作为瓶颈特征
        # hn: (num_layers, B, hidden_size) -> 取 hn[-1] -> (B, hidden_size)
        # 将其扩展为空间特征图 (B, hidden_size, 1, 1) 然后上采样
        
        # 调整 LSTM 输出通道数以便与 skip connection 拼接
        self.lstm_bottleneck_channels = hidden_size # 64
        # 上采样到 x5 (down4 output) 的大小 (B, 64, 8, 8)，然后调整通道数与 x5 匹配 (256)
        self.lstm_to_bottleneck = nn.Conv2d(self.lstm_bottleneck_channels, 512, kernel_size=1) 
        
        # 修改 Up 模块的输入通道数以匹配拼接后的特征
        # up1: 接收 lstm_bottleneck (512) + skip from down4 (512) = 1024
        self.up1 = Up(1024, 256) # <--- 修复点 2: 正确的输入通道数 ---
        self.up2 = Up(512, 128)  # 256 (from up1) + 128 (skip from down3) = 384 -> 输出 128
        self.up3 = Up(256, 64)   # 128 (from up2) + 64 (skip from down2) = 192 -> 输出 64
        self.up4 = Up(128, 32)   # 64 (from up3) + 32 (skip from down1) = 96 -> 输出 32
        # 最后插值到原图大小
        self.final_upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.outc = OutConv(32, num_classes)

    def forward(self, x):
        # CNN Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Shape: (B, 512, 8, 8)
        
        # Prepare for LSTM
        batch_size = x5.size(0)
        # Flatten spatial dimensions: (B, 512, 8, 8) -> (B, 512, 64)
        x5_flat = x5.view(batch_size, 512, -1) # (B, 512, 64)
        # Transpose to (B, seq_len, input_size): (B, 64, 512)
        x5_seq = x5_flat.permute(0, 2, 1) # (B, 64, 512)
        
        # LSTM
        # x5_seq: (B, seq_len=64, input_size=512)
        lstm_out, (hn, cn) = self.lstm(x5_seq) # lstm_out: (B, 64, hidden_size)
        
        # Decoder
        # 使用最后一个时间步的隐藏状态作为特征
        hn_last = hn[-1] # (B, hidden_size)
        # 扩展为空间特征图 (B, hidden_size, 1, 1)
        bottleneck = hn_last.unsqueeze(-1).unsqueeze(-1) # (B, hidden_size, 1, 1)
        # 上采样到 x5 的大小 (B, hidden_size, 8, 8)
        bottleneck_up = torch.nn.functional.interpolate(bottleneck, size=(8, 8), mode='nearest') # (B, 64, 8, 8)
        # 调整通道数以匹配 x5 (512)
        bottleneck_up = self.lstm_to_bottleneck(bottleneck_up) # (B, 512, 8, 8)
        
        # Upsample with skip connections
        x = self.up1(bottleneck_up, x5) # (B, 256, 16, 16) - 注意: up1 输入通道现在是 1024
        x = self.up2(x, x4)             # (B, 128, 32, 32)
        x = self.up3(x, x3)             # (B, 64, 64, 64)
        x = self.up4(x, x2)             # (B, 32, 128, 128)
        x = self.final_upsample(x)      # (B, 32, 256, 256)
        logits = self.outc(x)           # (B, 1, 256, 256)
        return logits

# --- 3. 简化版 CNN_LSTM_Model (无 LSTM) ---
class CNN_LSTM_Model_Simple(nn.Module):
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

# --- 实例化模型 ---
if MODEL_TYPE == "UNet":
    model = UNet(n_channels=6, n_classes=1).to(device)
elif MODEL_TYPE == "CNN_LSTM_Model":
    model = CNN_LSTM_Model(input_channels=6, hidden_size=64, num_layers=1, num_classes=1).to(device)
elif MODEL_TYPE == "CNN_LSTM_Model_Simple":
    model = CNN_LSTM_Model_Simple(in_channels=6, n_classes=1, base_channels=32).to(device)

# ==== 损失函数 ====
class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        prob = torch.sigmoid(pred)
        intersection = (prob * target).sum(dim=(1,2,3))
        union = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()
        loss = self.weight_bce * bce_loss + self.weight_dice * dice_loss
        return loss

criterion = DiceBCELoss(weight_bce=1.0, weight_dice=1.0, smooth=1e-6)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5) # 每15个epoch减半

def iou_score(pred, target, eps=1e-7):
    """计算 IoU 分数"""
    with torch.no_grad():
        prob = torch.sigmoid(pred)
        pred_bin = (prob > 0.5).float()
        inter = (pred_bin * target).sum(dim=(1,2,3))
        union = (pred_bin + target - pred_bin * target).sum(dim=(1,2,3)) + eps
        iou = (inter / union).mean().item()
    return iou

def save_prediction(epoch, pred, target, fname_prefix="pred"):
    """保存预测结果和真实标签用于可视化诊断"""
    with torch.no_grad():
        pred_img = (torch.sigmoid(pred[0]) > 0.5).float().cpu().numpy().squeeze() * 255
        target_img = target[0].cpu().numpy().squeeze() * 255
        pred_img = Image.fromarray(pred_img.astype(np.uint8), mode='L')
        target_img = Image.fromarray(target_img.astype(np.uint8), mode='L')
        pred_img.save(os.path.join(VIS_DIR, f"{fname_prefix}_epoch{epoch:02d}.png"))
        target_img.save(os.path.join(VIS_DIR, f"label_epoch{epoch:02d}.png"))

# --- 计算模型参数总数 ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"[INFO] Model '{MODEL_TYPE}' has {count_parameters(model)} trainable parameters.")
best_val_iou = 0.0
print(f"[INFO] Starting training for {EPOCHS} epochs with {MODEL_TYPE} model...")
print(f"[INFO] Batch Size: {BATCH_SIZE}, Initial LR: {LR}")

for epoch in range(1, EPOCHS+1):
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    train_iou = 0.0
    num_train_batches = len(train_loader)
    
    # --- 用于累积梯度范数 ---
    total_grad_norm = 0.0
    
    for batch_idx, (x, y, name) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        
        # --- 计算并累积梯度范数 ---
        param_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_grad_norm += p.grad.data.norm(2).item() ** 2
        param_grad_norm = param_grad_norm ** (1. / 2)
        total_grad_norm += param_grad_norm
        
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        train_iou += iou_score(pred, y) * x.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_iou /= len(train_loader.dataset)
    avg_grad_norm = total_grad_norm / num_train_batches

    # --- 验证阶段 ---
    model.eval()
    val_loss = 0.0
    val_iou  = 0.0
    with torch.no_grad():
        for batch_idx, (x, y, name) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item() * x.size(0)
            val_iou  += iou_score(pred, y) * x.size(0)
            if batch_idx == 0 and epoch % 5 == 0:
                 save_prediction(epoch, pred, y, fname_prefix=f"val_pred_{MODEL_TYPE}")
                
    val_loss /= len(val_loader.dataset)
    val_iou  /= len(val_loader.dataset)

    scheduler.step()

    # --- 打印日志 (包含梯度范数) ---
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train_loss={train_loss:.4f} | train_iou={train_iou:.4f} | "
          f"val_loss={val_loss:.4f} | val_iou={val_iou:.4f} | "
          f"lr={current_lr:.2e} | avg_grad_norm={avg_grad_norm:.6f}")

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"[INFO] New best model saved to {SAVE_PATH} (val_iou={best_val_iou:.4f})")

print(f"[INFO] Training completed. Best val_iou for {MODEL_TYPE}: {best_val_iou:.4f}")




