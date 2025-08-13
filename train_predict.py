# train_predict.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import create_model

# 假设已有预处理好的数据集 X (samples×seq_len×features), Y (samples×output_len)
# 这里仅示例构造随机数据
X = np.random.rand(1000, 7, 1)   # 1000 个样本，每个样本 7 年 NDVI
Y = np.random.rand(1000, 1)      # 对应每个样本未来 1 年 NDVI（可扩展为多步预测）

# 转为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 创建模型、损失函数和优化器
model = create_model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
epochs = 50
for epoch in range(epochs):
    epoch_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

# 训练完成后保存模型
torch.save(model.state_dict(), "models/landuse_lstm.pth")
print("模型训练完成并已保存。")

# 预测（以训练集最后一个样本为例）
with torch.no_grad():
    sample = X_tensor[-1].unsqueeze(0)  # 最后一条样本
    predicted = model(sample).numpy().flatten()
    print("未来一年 NDVI 预测值（示例）：", predicted)
