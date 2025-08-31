# preprocess_seq.py
import os
import numpy as np
import rasterio
from tqdm import tqdm

# 配置路径：改成你 EuroSAT 解压后的 tif 文件路径
EURO_TIF_DIR = r"D:\PYCode\landuse_prediction_project\data\eurosat\ds\images\remote_sensing\otherDatasets\sentinel_2\tif"
OUT_DIR = "data_prepared"
SEQ_LEN = 5           # 每个样本的时间步数（模拟）
USE_CHANNELS = [1, 2, 3]  # 使用 RGB 三个波段

os.makedirs(OUT_DIR, exist_ok=True)

# 获取类别
classes = sorted([d for d in os.listdir(EURO_TIF_DIR) if os.path.isdir(os.path.join(EURO_TIF_DIR, d))])
class_to_idx = {c: i for i, c in enumerate(classes)}
print("检测到的类别:", class_to_idx)

X_list, Y_list = [], []

for cls in classes:
    folder = os.path.join(EURO_TIF_DIR, cls)
    files = [f for f in os.listdir(folder) if f.lower().endswith(".tif")]
    for fname in tqdm(files, desc=f"处理类别: {cls}"):
        path = os.path.join(folder, fname)
        try:
            with rasterio.open(path) as src:
                # 读取指定波段
                bands = []
                for b in USE_CHANNELS:
                    band = src.read(b).astype(np.float32)
                    bands.append(band)
                img = np.stack(bands, axis=2)  # H, W, C

                # 归一化
                img = img - img.min()
                if img.max() > 0:
                    img = img / img.max()

                # 转为 C,H,W
                img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32)

                # 模拟时间序列
                seq = []
                for t in range(SEQ_LEN):
                    noise = np.random.normal(0, 0.01, img_chw.shape).astype(np.float32)
                    frame = np.clip(img_chw + noise, 0.0, 1.0)
                    seq.append(frame)
                seq = np.stack(seq, axis=0)  # T, C, H, W

                X_list.append(seq)
                Y_list.append(class_to_idx[cls])
        except Exception as e:
            print(f"跳过文件 {path}, 错误: {e}")

# 保存
X = np.array(X_list)  # N, T, C, H, W
Y = np.array(Y_list)
np.save(os.path.join(OUT_DIR, "X_seq.npy"), X)
np.save(os.path.join(OUT_DIR, "Y.npy"), Y)
np.save(os.path.join(OUT_DIR, "classes.npy"), np.array(classes))

print(f"保存完成: {OUT_DIR}")
print("X 形状:", X.shape)
print("Y 形状:", Y.shape)
