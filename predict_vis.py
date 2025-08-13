# predict_vis.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import CNN_LSTM

DATA_DIR = "data_prepared"
X = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
Y = np.load(os.path.join(DATA_DIR, "Y.npy"))
classes = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = CNN_LSTM(in_channels=X.shape[2], cnn_feat=128, lstm_hidden=128, lstm_layers=1, num_classes=len(classes))
model.load_state_dict(torch.load("models/cnn_lstm.pth", map_location=device))
model.to(device)
model.eval()

# pick some samples and visualize predicted vs true
os.makedirs("results", exist_ok=True)
n_samples = min(12, X.shape[0])
indices = np.random.choice(X.shape[0], n_samples, replace=False)

fig, axes = plt.subplots(n_samples//4, 4, figsize=(12, 3*(n_samples//4)))
axes = axes.flatten()
for ax, idx in zip(axes, indices):
    seq = X[idx]  # T,C,H,W
    # show middle frame RGB
    mid = seq[seq.shape[0]//2]  # C,H,W
    img = np.transpose(mid, (1,2,0))
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    xb = torch.tensor(X[idx:idx+1], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(xb)
        pred = logits.argmax(dim=1).item()
    ax.set_title(f"pred:{classes[pred]}\ntrue:{classes[Y[idx]]}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("results/sample_predictions.png", dpi=200)
print("Saved results/sample_predictions.png")
