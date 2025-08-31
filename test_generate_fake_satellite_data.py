# generate_fake_satellite_data.py
import os
import numpy as np
from torchgeo.datasets import EuroSAT
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import geopandas as gpd

# 数据目录
DATA_DIR = "data"
BOUNDARY_DIR = os.path.join(DATA_DIR, "boundary")
RAW_IMG_DIR = os.path.join(DATA_DIR, "raw_images")

os.makedirs(BOUNDARY_DIR, exist_ok=True)
os.makedirs(RAW_IMG_DIR, exist_ok=True)

# 1. 下载或读取 EuroSAT 数据
print("Loading EuroSAT...")
# 使用RGB波段以确保我们得到的是3通道图像
dataset = EuroSAT(root=os.path.join(DATA_DIR, "eurosat"), download=True, bands=["B04", "B03", "B02"])

# 2. 随机取 16 张图像（4x4 拼接）
idxs = np.random.choice(len(dataset), 16, replace=False)
patches = []
for i in idxs:
    sample = dataset[i]
    # EuroSAT返回的是字典格式，包含'image'和'label'键
    img_tensor = sample['image']  # 这是一个tensor，形状为(C, H, W)
    print(f"Original image tensor shape: {img_tensor.shape}")
    
    # 将tensor转换为numpy数组，并调整维度顺序为(H, W, C)
    img = img_tensor.numpy()
    if len(img.shape) == 3:
        # EuroSAT的图像格式是(C, H, W)，需要转换为(H, W, C)
        img = np.transpose(img, (1, 2, 0))
    
    print(f"Processed image shape: {img.shape}")
    
    # 确保数据类型正确（0-255范围的uint8）
    if img.max() <= 1.0:
        # 如果值在0-1范围内，放大到0-255范围
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        # 归一化到0-255范围
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
    patches.append(img)

# 3. 拼接成大图（256x256）
print("拼接图像...")
rows = []
for r in range(4):
    start_idx = r * 4
    end_idx = min((r + 1) * 4, len(patches))
    row_imgs = patches[start_idx:end_idx]
    if len(row_imgs) > 0:
        row_combined = np.hstack(row_imgs)
        rows.append(row_combined)

if len(rows) > 0:
    big_img = np.vstack(rows)
    print(f"Big image shape: {big_img.shape}")

    # 4. 保存为 GeoTIFF（假设左上角经纬度是 0,0，每像素 0.01 度）
    transform = from_origin(0, 0, 0.01, 0.01)  # 设置坐标系
    tif_path = os.path.join(RAW_IMG_DIR, "fake_image.tif")
    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=big_img.shape[0],
        width=big_img.shape[1],
        count=big_img.shape[2] if len(big_img.shape) > 2 else 1,
        dtype='uint8',
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        if len(big_img.shape) > 2 and big_img.shape[2] >= 3:
            # 彩色图像 (H, W, C) 格式
            for b in range(min(3, big_img.shape[2])):  # 对 RGB 三个通道逐个写入
                dst.write(big_img[:,:,b], b+1)
        else:
            # 灰度图像或单通道图像
            channel_data = big_img[:, :, 0] if len(big_img.shape) > 2 else big_img
            dst.write(channel_data, 1)
    print(f"Saved fake satellite image to {tif_path}")

    # 5. 创建 ROI shapefile（覆盖中间一块区域）
    roi_geom = box(0.05, -0.15, 0.15, -0.05)  # 假设坐标系 EPSG:4326
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[roi_geom], crs="EPSG:4326")
    roi_path = os.path.join(BOUNDARY_DIR, "roi.shp")
    gdf.to_file(roi_path)
    print(f"Saved ROI shapefile to {roi_path}")
else:
    print("Error: No valid images to process.")