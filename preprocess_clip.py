# preprocess_clip.py
import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# ===== 配置 =====
BOUNDARY_FILE = "data/boundary/roi.shp"  # 研究区边界文件（.shp 或 .geojson）
RAW_IMAGE_DIR = "data/raw_images"        # 原始影像目录（.tif）
CLIPPED_DIR = "data_prepared/clipped"    # 裁剪后保存目录

os.makedirs(CLIPPED_DIR, exist_ok=True)

def clip_image_to_roi(image_path, shapes, output_path):
    with rasterio.open(image_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def main():
    # 1. 读取研究区边界
    print(f"Loading ROI boundary from: {BOUNDARY_FILE}")
    gdf = gpd.read_file(BOUNDARY_FILE)
    shapes = gdf.geometry

    # 2. 遍历原始影像文件
    for fname in os.listdir(RAW_IMAGE_DIR):
        if fname.lower().endswith(".tif"):
            in_path = os.path.join(RAW_IMAGE_DIR, fname)
            out_path = os.path.join(CLIPPED_DIR, fname)

            print(f"Clipping: {fname}")
            clip_image_to_roi(in_path, shapes, out_path)

    print(f"All clipped images saved to {CLIPPED_DIR}")

if __name__ == "__main__":
    main()
