# preprocess_seq.py
import os
import numpy as np
import rasterio
from tqdm import tqdm

# 配置：修改为你项目里的目录（相对或绝对）
EURO_TIF_DIR = r"data/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif"
OUT_DIR = "data_prepared"
SEQ_LEN = 5           # 模拟时间步数（可改）
USE_CHANNELS = [1,2,3]  # rasterio 1-based band indices to use as channels (e.g., 1:R,2:G,3:B) —— 调整若必要

os.makedirs(OUT_DIR, exist_ok=True)

classes = sorted([d for d in os.listdir(EURO_TIF_DIR) if os.path.isdir(os.path.join(EURO_TIF_DIR, d))])
class_to_idx = {c:i for i,c in enumerate(classes)}
print("Classes:", class_to_idx)

X_list = []
Y_list = []

for cls in classes:
    folder = os.path.join(EURO_TIF_DIR, cls)
    files = [f for f in os.listdir(folder) if f.lower().endswith(".tif")]
    for fname in tqdm(files, desc=f"Processing {cls}"):
        path = os.path.join(folder, fname)
        try:
            with rasterio.open(path) as src:
                # read desired bands; if file has fewer bands, rasterio will raise
                bands = []
                for b in USE_CHANNELS:
                    try:
                        band = src.read(b).astype(np.float32)
                    except Exception:
                        # fallback: if band not present, read band 1
                        band = src.read(1).astype(np.float32)
                    bands.append(band)
                img = np.stack(bands, axis=2)  # H x W x C

                # normalize to 0-1
                img = img - img.min()
                if img.max() > 0:
                    img = img / img.max()
                # convert to C,H,W and float32
                img_chw = np.transpose(img, (2,0,1)).astype(np.float32)

                # simulate a temporal sequence by repeating with tiny noise
                seq = []
                for t in range(SEQ_LEN):
                    noise = np.random.normal(0, 0.01, img_chw.shape).astype(np.float32)
                    frame = img_chw + noise
                    frame = np.clip(frame, 0.0, 1.0)
                    seq.append(frame)
                seq = np.stack(seq, axis=0)  # T, C, H, W

                X_list.append(seq)
                Y_list.append(class_to_idx[cls])

        except Exception as e:
            print(f"Skipped {path}, err: {e}")

X = np.array(X_list)  # N, T, C, H, W
Y = np.array(Y_list)
print("Final shapes:", X.shape, Y.shape)

np.save(os.path.join(OUT_DIR, "X_seq.npy"), X)
np.save(os.path.join(OUT_DIR, "Y.npy"), Y)
np.save(os.path.join(OUT_DIR, "classes.npy"), np.array(classes))
print("Saved to", OUT_DIR)
