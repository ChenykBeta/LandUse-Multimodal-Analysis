# predict_vis.py 
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib.pyplot as plt
from model import CNN_LSTM

# 配置路径
DATA_PREPARED = "data_prepared"
MODEL_PATH = "models/cnn_lstm.pth"
BATCH_SIZE = 16  #
OUT_DIR = "predict_results"
os.makedirs(OUT_DIR, exist_ok=True)

# === 加载数据 ===
X = np.load(os.path.join(DATA_PREPARED, "X_seq.npy"))
Y = np.load(os.path.join(DATA_PREPARED, "Y.npy"))
classes = np.load(os.path.join(DATA_PREPARED, "classes.npy"))

print(f"[INFO] 模型输入形状: SEQ_LEN={X.shape[1]}, C={X.shape[2]}, H={X.shape[3]}, W={X.shape[4]}")
print(f"[INFO] 类别数: {len(classes)}, 类别: {classes}")

# 转为 Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# === 定义模型 ===
SEQ_LEN, C, H, W = X.shape[1:]
model = CNN_LSTM(
    in_channels=C,
    cnn_feat=128,
    lstm_hidden=128,
    lstm_layers=1,
    num_classes=len(classes)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === 批量预测 ===
all_preds = []
with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH_SIZE):
        batch_x = X_tensor[i:i+BATCH_SIZE]
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.numpy())

all_preds = np.array(all_preds)

# === 可视化部分预测结果 ===
num_show = min(10, len(X))  # 只显示前10个样本
for idx in range(num_show):
    true_label = classes[Y[idx]]
    pred_label = classes[all_preds[idx]]

    img = np.transpose(X[idx, 0], (1, 2, 0))  # 取时间序列第1帧，C,H,W -> H,W,C
    plt.imshow(img)
    plt.title(f"True: {true_label}, Pred: {pred_label}")
    plt.axis("off")
    plt.savefig(os.path.join(OUT_DIR, f"sample_{idx}.png"), dpi=150)
    plt.close()

print(f"[INFO] 已保存 {num_show} 个预测可视化结果到 {OUT_DIR}")
