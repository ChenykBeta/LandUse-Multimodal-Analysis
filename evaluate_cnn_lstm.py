import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

from model import CNN_LSTM

# 配置路径
DATA_PREPARED = "data_prepared"
MODEL_PATH = "models/cnn_lstm.pth"
BATCH_SIZE = 16  # 小批量处理，防止卡死

# === 加载数据 ===
X = np.load(os.path.join(DATA_PREPARED, "X_seq.npy"))
Y = np.load(os.path.join(DATA_PREPARED, "Y.npy"))
classes = np.load(os.path.join(DATA_PREPARED, "classes.npy"))

print(f"[INFO] 数据集形状: X={X.shape}, Y={Y.shape}")
print(f"[INFO] 类别数: {len(classes)}, 类别: {classes}")

# 转为 Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.long)

# === 定义模型（保持与训练时一致）===
SEQ_LEN, C, H, W = X.shape[1:]
model = CNN_LSTM(
    in_channels=C,
    cnn_feat=128,
    lstm_hidden=128,
    lstm_layers=1,
    num_classes=len(classes)
)

# === 加载模型参数 ===
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === 分批预测 ===
all_preds = []
with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH_SIZE):
        batch_x = X_tensor[i:i+BATCH_SIZE]
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.numpy())

all_preds = np.array(all_preds)

# === 计算准确率 ===
acc = accuracy_score(Y, all_preds)
print(f"[INFO] 模型在全部数据上的准确率: {acc:.4f}")

# === 混淆矩阵可视化 ===
cm = confusion_matrix(Y, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("[INFO] 混淆矩阵已保存为 confusion_matrix.png")
