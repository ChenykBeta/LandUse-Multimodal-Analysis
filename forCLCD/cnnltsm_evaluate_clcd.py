# evaluate_clcd.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T

# ========================================
#  配置参数（请按需修改）
# ========================================
MODEL_TYPE = "CNN_LSTM_Model"  # ✅ 必须和训练时一致
CHECKPOINT_PATH = "models/clcd_cnn_lstm.pth"  # ✅ 对应的 .pth 文件
DATA_ROOT = "data/CLCD"
SPLIT = "test"  # 或 "val"
OUTPUT_DIR = f"results_{MODEL_TYPE}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_WORKERS = 2

# ========================================
#  模型组件定义（完全复制 train_clcd.py 中的定义）
# ========================================

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
    def __init__(self, in_channels, out_channels, bilinear=True):  # 注意：bilinear=True
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

# --- 核心：CNN_LSTM_Model（完全复制）---
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels=6, hidden_size=64, num_layers=1, num_classes=1):
        super(CNN_LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN Encoder
        self.inc = DoubleConv(input_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)  # 输出 (B, 512, 8, 8)
        
        # LSTM
        self.lstm_input_size = 512
        self.lstm_seq_len = 8 * 8
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Decoder
        self.lstm_bottleneck_channels = hidden_size
        self.lstm_to_bottleneck = nn.Conv2d(self.lstm_bottleneck_channels, 512, kernel_size=1) 
        
        self.up1 = Up(1024, 256)  # 拼接后 512+512=1024
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 32)
        self.final_upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.outc = OutConv(32, num_classes)

    def forward(self, x):
        # CNN Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # (B, 512, 8, 8)
        
        # Prepare for LSTM
        batch_size = x5.size(0)
        x5_flat = x5.view(batch_size, 512, -1)  # (B, 512, 64)
        x5_seq = x5_flat.permute(0, 2, 1)       # (B, 64, 512)
        
        # LSTM
        lstm_out, (hn, cn) = self.lstm(x5_seq)  # lstm_out: (B, 64, 64)
        
        # Decoder
        hn_last = hn[-1]  # (B, 64)
        bottleneck = hn_last.unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        bottleneck_up = torch.nn.functional.interpolate(bottleneck, size=(8, 8), mode='nearest')  # (B, 64, 8, 8)
        bottleneck_up = self.lstm_to_bottleneck(bottleneck_up)  # (B, 512, 8, 8)
        
        # Upsample with skip connections
        x = self.up1(bottleneck_up, x5)  # (B, 256, 16, 16)
        x = self.up2(x, x4)              # (B, 128, 32, 32)
        x = self.up3(x, x3)              # (B, 64, 64, 64)
        x = self.up4(x, x2)              # (B, 32, 128, 128)
        x = self.final_upsample(x)       # (B, 32, 256, 256)
        logits = self.outc(x)            # (B, 1, 256, 256)
        return logits

# ========================================
#  数据集定义
# ========================================
class CLCDDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="test", transform=None):
        self.t1_dir = os.path.join(root_dir, split, "time1")
        self.t2_dir = os.path.join(root_dir, split, "time2")
        self.filenames = [f for f in os.listdir(self.t1_dir) if f.endswith(".png") or f.endswith(".jpg")]
        self.transform = transform or T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        t1 = Image.open(os.path.join(self.t1_dir, fname)).convert("RGB")
        t2 = Image.open(os.path.join(self.t2_dir, fname)).convert("RGB")
        t1 = self.transform(t1)
        t2 = self.transform(t2)
        x = torch.cat([t1, t2], dim=0)  # (6, H, W)
        return x, fname  # 测试时不需要 label

# ========================================
#  主函数
# ========================================
def main():
    print(f"🚀 使用模型: {MODEL_TYPE}")
    print(f"📁 加载权重: {CHECKPOINT_PATH}")

    # 1. 构建模型
    model = CNN_LSTM_Model(input_channels=6, hidden_size=64, num_layers=1, num_classes=1).to(DEVICE)

    # 2. 加载权重
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 模型权重加载成功！")

    # 3. 加载数据集
    dataset = CLCDDataset(root_dir=DATA_ROOT, split=SPLIT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"📊 数据集大小: {len(dataset)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. 推理
    with torch.no_grad():
        for x, fnames in tqdm(dataloader, desc="推理中"):
            x = x.to(DEVICE)
            pred = model(x)  # (B, 1, 256, 256)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float() * 255
            pred = pred.cpu().numpy()

            for i, fname in enumerate(fnames):
                img = pred[i, 0]
                img = Image.fromarray(img.astype(np.uint8), mode='L')
                save_path = os.path.join(OUTPUT_DIR, fname)
                img.save(save_path)

    print(f"🎉 推理完成！结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()