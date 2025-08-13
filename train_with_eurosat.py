import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 从你的项目 model.py 中导入 LSTM
from model import LandUseLSTM

data = torch.load("data/processed_eurosat.pt", weights_only=False)
# /data = torch.load("data/processed_eurosat.pt")
X = torch.tensor(data["X"], dtype=torch.float32)
Y = torch.tensor(data["Y"], dtype=torch.long)
classes = data["classes"]

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LandUseLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

print("Training finished. Classes:", classes)
