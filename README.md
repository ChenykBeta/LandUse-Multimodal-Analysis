# LandUse-Multimodal-Analysis
基于多模态融合的土地利用分析与预测系统

# 🗂 目录结构（建议使用 VSCode + GitHub 同步）：

# LandUse-DeepPredictor/
# ├── data/                # 存放原始数据（遥感、矢量图）
# ├── preprocess/          # 数据处理与标签生成脚本
# ├── classify_model/      # 土地利用分类模型（CNN）
# ├── forecast_model/      # 趋势预测模型（LSTM）
# ├── analysis/            # 统计变化、转移矩阵分析
# ├── visualize/           # 可视化绘图与地图渲染
# ├── output/              # 输出结果图像、模型、图表
# ├── report/              # Word报告草稿、管理建议文档
# └── README.md            # 项目说明入口

# ✅ 初始化说明文件（README.md）模板：

readme_template = """
# LandUse-DeepPredictor

本项目旨在通过多源遥感影像、深度学习模型与地理信息融合，完成区域土地利用的识别、时序预测与趋势可视化。

## 项目目标
1. 提取 2017-2023 年多时相遥感影像，进行土地利用分类（耕地、林地、草地等）
2. 构建 CNN 分类模型与 LSTM 时间序列预测模型，输出未来趋势
3. 提供可视化图表与管理建议文档

## 主要目录说明
- `data/`：存放遥感数据（建议使用 GeoTIFF 格式）与 shapefile 矢量图层
- `preprocess/`：图像裁剪、掩膜处理、样本标签生成等预处理脚本
- `classify_model/`：CNN 卷积模型训练与精度评估
- `forecast_model/`：基于 LSTM 的土地利用面积预测
- `analysis/`：生成转移矩阵、变化趋势统计图
- `visualize/`：渲染地图、绘制面积变化折线图等
- `output/`：保存模型结果、图像、图表
- `report/`：撰写项目报告、政策建议草案

## 环境配置建议
```bash
conda create -n landuse python=3.9
conda activate landuse
pip install -r requirements.txt
```

## 快速开始
```bash
# 1. 裁剪遥感图像到研究区域
python preprocess/crop_image.py

# 2. 训练 CNN 分类模型
python classify_model/train_cnn.py

# 3. 构建 LSTM 预测模型
python forecast_model/train_lstm.py

# 4. 统计与可视化输出
python analysis/land_change_analysis.py
```

## 项目成员
陈云开、邢万鑫、方昶朔、王仕钰、李浩东、魏含希、周晓宁、胡雅昕
