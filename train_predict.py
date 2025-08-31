import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import CNN_LSTM  # 导入模型

# 假设已有预处理好的数据集 X (samples×seq_len×features), Y (samples×output_len)
X = np.random.rand(1000, 5, 3, 10, 10)  # 1000 个样本，每个样本 5 年 NDVI，3 个通道，10x10 的图像
Y = np.random.randint(0, 10, 1000)  # 对应每个样本未来 1 年 NDVI 分类标签（10 类）

# 转为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.long)  # 分类任务标签应为 long 类型

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 创建模型、损失函数和优化器
model = CNN_LSTM(in_channels=3, cnn_feat=128, lstm_hidden=64, lstm_layers=1, num_classes=10)
criterion = torch.nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
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
torch.save(model.state_dict(), "models/cnn_lstm.pth")
print("模型训练完成并已保存。")

# 预测（以训练集最后一个样本为例）
# 预测（以训练集最后一个样本为例）
with torch.no_grad():
    sample = X_tensor[-1].unsqueeze(0)  # 最后一条样本
    predicted = model(sample).cpu().numpy().flatten()

    # 找到预测类别编号
    pred_idx = np.argmax(predicted)

    # 读取类别名称
    classes = np.load("data_prepared/classes.npy")
    pred_name = classes[pred_idx]

    print("预测结果（类别编号）：", pred_idx)
    print("预测结果（类别名称）：", pred_name)
    print("预测结果的原始分数向量：", predicted)


