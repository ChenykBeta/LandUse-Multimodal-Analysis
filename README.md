# LandUse-DeepPredictor

基于多模态遥感数据的土地利用识别与趋势预测系统  
项目目标是构建一个融合遥感影像、地理信息、统计数据的深度学习框架，实现区域土地利用分类、时空变化分析与未来趋势预测，并形成可视化成果与管理建议。

## 项目背景

土地利用变化是城乡发展、生态演变与资源管理的重要指标。以玛纳斯地区为例，耕地扩张、城市用地增长和生态退化问题日益突出。传统人工调查成本高、更新慢，亟需基于遥感与深度学习的自动化分析手段。

## 项目目标

1. 构建多时相遥感数据的预处理与融合流程
2. 训练 CNN 分类模型，实现地类识别（耕地、林地、水体等）
3. 利用 LSTM 时序模型预测未来土地利用格局
4. 输出变化趋势图、预测图与管理建议报告

## 项目结构

LandUse-DeepPredictor/
├── data/                原始遥感影像、shp、统计数据  
├── preprocess/          数据裁剪、标签生成、样本构建  
├── classify_model/      CNN 分类模型训练与测试  
├── forecast_model/      LSTM 预测模型训练与趋势图  
├── analysis/            土地变化统计、转移矩阵计算  
├── visualize/           可视化图像绘制（地图、曲线）  
├── output/              输出结果（图像、模型、图表）  
├── report/              项目报告、政策建议、成员总结  
└── README.md            项目说明文档  

## 环境配置

建议使用 Python 3.9 + conda 环境：

conda create -n landuse python=3.9  
conda activate landuse  
pip install -r requirements.txt  

主要依赖库包括：  
- rasterio, geopandas, shapely （遥感与矢量处理）  
- pytorch, scikit-learn （深度学习与传统模型）  
- matplotlib, seaborn （图表绘制）  
- streamlit（可视化交互平台，可选）  

## 快速开始

# 步骤1：裁剪图像至研究区范围  
python preprocess/crop_image.py  

# 步骤2：训练 CNN 分类模型  
python classify_model/train_cnn.py  

# 步骤3：分析土地利用变化  
python analysis/land_change_analysis.py  

# 步骤4：训练 LSTM 模型并预测趋势  
python forecast_model/train_lstm.py  

# 步骤5：绘制分类图与预测图  
python visualize/draw_map.py  

## 输出成果

- 土地利用分类图（2017~2023）  
- 土地利用变化趋势图（面积统计）  
- 土地利用预测图（2025~2027）  
- 转移矩阵与空间变化分析  
- 管理建议报告（Word/PDF）  
- 可复现代码仓库（GitHub）  

## 项目成员

陈云开（项目总控）  
邢万鑫、方昶朔、王仕钰、李浩东、魏含希、周晓宁、胡雅昕（按成果模块协作分工）  

## 许可证

本项目采用 MIT License 开源协议。欢迎学习与二次开发，引用请注明来源。
