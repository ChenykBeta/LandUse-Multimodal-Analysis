# train_cnn_lstm.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import CNN_LSTM

DATA_DIR = "data_prepared"
X_path = os.path.join(DATA_DIR, "X_seq.npy")
Y_path = os.path.join(DATA_DIR, "Y.npy")
classes_path = os.path.join(DATA_DIR, "classes.npy")

# 1. load
X = np.load(X_path)  # N, T, C, H, W
Y = np.load(Y_path)
classes = np.load(classes_path, allow_pickle=True)
num_classes = len(classes)
print("Loaded", X.shape, "labels:", num_classes)

# 2. tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.long)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)


# 3. model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM(in_channels=X.shape[2], cnn_feat=128, lstm_hidden=128, lstm_layers=1, num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. train
EPOCHS = 8
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} acc={acc:.4f}")

# 5. save
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cnn_lstm.pth")
print("Saved model to models/cnn_lstm.pth")
