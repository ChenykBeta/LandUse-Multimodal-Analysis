import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """简单 CNN 编码器：输入 (B, C, H, W) -> 输出 (B, feature_dim)"""
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x H/2 x W/2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x H/4 x W/4
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),  # 128 x 1 x 1
        )
        self.out_dim = feature_dim
        self.fc = nn.Linear(128, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, C, H, W)
        f = self.features(x)             # (B, 128, 1, 1)
        f = f.view(f.size(0), -1)        # (B, 128)
        f = self.relu(self.fc(f))        # (B, feature_dim)
        return f

class CNN_LSTM(nn.Module):
    """CNN encoder + LSTM for sequence classification"""
    def __init__(self, in_channels=3, cnn_feat=128, lstm_hidden=64, lstm_layers=1, num_classes=10):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, feature_dim=cnn_feat)
        self.lstm = nn.LSTM(input_size=cnn_feat, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden//2),
            nn.ReLU(),
            nn.Linear(lstm_hidden//2, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        # collapse time and batch to encode in batch
        x_flat = x.view(B * T, C, H, W)
        feats = self.encoder(x_flat)               # (B*T, feat)
        feats = feats.view(B, T, -1)               # (B, T, feat)
        out, _ = self.lstm(feats)                  # (B, T, hidden)
        last = out[:, -1, :]                       # (B, hidden)
        logits = self.classifier(last)             # (B, num_classes)
        return logits
