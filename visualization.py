# visualization.py

import numpy as np
import matplotlib.pyplot as plt
import rasterio

# 示例：生成并保存一个分类地图（虚构数据）
landuse_map = np.random.randint(1, 4, size=(100, 100))  # 1:耕地, 2:林地, 3:城市
plt.figure(figsize=(4,4))
plt.imshow(landuse_map, cmap='viridis')
plt.title("预测地类图")
plt.colorbar(ticks=[1,2,3], label='类标号')
plt.savefig("results/landuse_map.png")
plt.close()

# 示例：绘制趋势图（假设耕地与城市面积随年变化）
years = np.arange(2018, 2029)
area_farmland = np.linspace(60, 30, len(years))  # 耕地面积减少
area_urban = np.linspace(10, 40, len(years))     # 城市用地增加
plt.figure(figsize=(6,4))
plt.plot(years, area_farmland, marker='o', label='耕地面积 (%)')
plt.plot(years, area_urban, marker='s', label='城市用地 (%)')
plt.xlabel("年份")
plt.ylabel("面积占比")
plt.title("土地利用变化趋势")
plt.legend()
plt.grid(True)
plt.savefig("results/trend_chart.png")
plt.close()
