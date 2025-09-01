# LandUse-Multimodal-Analysis

**基于多模态融合的深度学习土地利用变化检测与预测系统**

本项目旨在利用深度学习技术，融合多时相遥感影像，实现高精度的土地利用变化检测与动态预测。系统以 `dataset.py`, `train.py`, `evaluate.py` 三大核心脚本为驱动，构建了从数据加载、模型训练到结果评估的完整自动化流程，为土地资源管理提供智能化分析工具。


## 项目亮点

- **三大核心脚本**：`dataset.py`, `train.py`, `evaluate.py` 构成简洁高效的 pipeline。
- **先进模型架构**：支持 UNet、CNN-LSTM 等多种深度学习模型，适用于变化检测与时空预测任务。
- **模块化设计**：代码结构清晰，易于理解、复现和二次开发。
- **端到端流程**：覆盖数据准备、模型训练、推理预测全阶段。



### 项目说明（背景补充）

本项目原计划基于真实区域（的高分辨率遥感数据进行土地利用分析与预测。由于与合作单位的数据对接流程尚在推进中，部分真实数据暂未完成获取与处理。

为确保项目技术路线的可行性验证和开发进度的持续推进，再开始测试时我们使用公开数据集 **EuroSAT** 作为替代方案，用于：

- 验证数据预处理流程
- 测试分类与预测模型的代码实现
- 调试训练与评估 pipeline。



## 基于EuroSAT数据的项目结构

```
LandUse-DeepPredictor/
├── data/                # EuroSAT 数据
│   └── EuroSAT/         # 原始数据目录
├── preprocess/          # 数据预处理脚本
│   └── crop_image.py    # 图像裁剪（示例）
├── classify_model/      # 分类模型代码
│   └── train_cnn.py     # CNN 模型训练
├── forecast_model/      # 预测模型代码
│   └── train_lstm.py    # LSTM 模型训练
├── analysis/            # 变化分析脚本
│   └── land_change_analysis.py
├── visualize/           # 可视化脚本
│   └── draw_map.py      # 结果绘图
├── output/              # 输出结果（图像、模型等）
└── report/              # 项目报告与总结（待填充）
```

环境配置将在下文的正式版本中提及，这里不再赘述。


## 使用流程

# 1. 裁剪图像
python preprocess/crop_image.py

# 2. 训练 CNN 分类模型
python classify_model/train_cnn.py

# 3. 分析土地变化
python analysis/land_change_analysis.py

# 4. 训练 LSTM 预测模型
python forecast_model/train_lstm.py

# 5. 绘制结果图
python visualize/draw_map.py

当前代码架构已完全适配真实业务场景，**一旦真实数据到位，仅需替换数据路径与调整少量参数，即可无缝切换至实际应用**。本测试版本的所有技术成果将作为正式版的核心基础。

当了解到在暑期已经很难拿到可用的数据集的时候，我们倍感压力；但幸好经过我们的不懈搜索，找到了相对高度可用的CLCD数据集（将在下面着重介绍）经过不懈努力，我们将上述模型重新编写、适配，形成了最终可用的、适配CLCD数据集的新模型：



### 项目结构（基于CLCD）
```
LandUse-Multimodal-Analysis/
├── data/                          # 数据根目录
│   └── CLCD/                      # 采用CLCD数据集格式
│       ├── train/                 # 训练集
│       │   ├── time1/             # t1 时相影像 (RGB)
│       │   ├── time2/             # t2 时相影像 (RGB)
│       │   └── label/             # 变化标签 (二值图)
│       ├── val/                   # 验证集 (结构同 train)
│       └── test/                  # 测试集 (结构同 train)
├── models/                        # 训练好的模型权重文件
│   ├── clcd_unet_*.pth            # UNet 模型权重
│   ├── clcd_cnn_lstm.pth          # CNN-LSTM 模型权重
│   └── clcd_cnn_lstm_simple.pth   # CNN-LSTM-Simple 模型权重
├── results_UNet/                  # UNet 模型的预测结果
├── results_CNN_LSTM_Model/        # CNN-LSTM 模型的预测结果
├── results_CNN_LSTM_Model_Simple/ # CNN-LSTM-Simple 模型的预测结果
├── dataset.py                     #  核心：数据集定义与加载
├── train.py                       #  核心：模型训练与验证
├── evaluate.py                    #  核心：模型评估与结果生成
├── requirements.txt               # Python 依赖包
└── README.md                      # 项目文档
```
###CropLand Change Dection （CLCD） 数据集
CLCD数据集由600对耕地变化样本图像组成，其中360对用于训练，120对用于验证，120对用于测试。 CLCD中的双时叶图像分别由中国广东省高分二号于2017年和2019年采集，空间分辨率范围为0.5-2 m。每组样本由512×512两张图像和相应的耕地变化二进制标签组成。

下载 CLCD 数据集：[OneDrive](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/liumx23_mail2_sysu_edu_cn/Ejm7aufQREdIhYf5yxSZDIkBr68p2AUQf_7BAEq4vmV0pg?e=ZWI3oy) |https://pan.baidu.com/s/1Un-bVxUm1N9IHiDOXLLHlg?pwd=miu2


数据灵感来源于：@ARTICLE{9780164,
  author={Liu, Mengxi and Chai, Zhuoqun and Deng, Haojun and Liu, Rong},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A CNN-Transformer Network With Multiscale Context Aggregation for Fine-Grained Cropland Change Detection}, 
  year={2022},
  volume={15},
  number={},
  pages={4297-4306},
  doi={10.1109/JSTARS.2022.3177235}}
在此表示忠心的感谢！

## 核心功能与脚本使用

本项目的核心由三个 Python 脚本构成，它们共同完成了土地利用分析的全流程。

### 1. `dataset.py` - 数据集加载器

该脚本定义了 `CLCDDataset` 类，负责加载和预处理 CLCD 格式的数据集。

- **功能**：
  - 自动从 `data/CLCD/{split}` 目录下读取 `time1` 和 `time2` 的遥感影像。
  - 将两张 RGB 影像拼接为 6 通道输入张量。
  - 支持图像尺寸调整和标准化。
- **使用**：
  ```python
  from dataset import CLCDDataset
  dataset = CLCDDataset(root_dir="data/CLCD", split="train")
  ```

### 2. `train.py` - 模型训练引擎

该脚本是模型的训练中心，支持多种模型架构。

- **功能**：
  - 支持训练 **UNet**, **CNN_LSTM_Model**, **CNN_LSTM_Model_Simple** 三种模型。
  - 自动选择 GPU/CPU，支持断点续训。
  - 训练过程中实时监控损失和评估指标（如 IoU），并在 `models/` 目录下保存最佳模型。
  - 训练日志清晰，便于调试。
- **使用**：
  ```bash
  # 训练 UNet 模型
  python train.py --model_type UNet --epochs 100 --lr 0.001

  # 训练 CNN-LSTM 模型
  python train.py --model_type CNN_LSTM_Model --epochs 100 --lr 0.001
  ```
  > **参数说明**：`--model_type` (UNet/CNN_LSTM_Model/CNN_LSTM_Model_Simple), `--epochs`, `--lr`, `--batch_size`。

### 3. `evaluate.py` - 模型评估与预测

该脚本用于加载训练好的模型并对测试集进行推理，生成最终的预测结果。

- **功能**：
  - 根据 `MODEL_TYPE` 自动加载对应的模型结构和权重。
  - 对测试集进行前向推理，生成二值变化图。
  - 将预测结果（0/1）转换为 0/255 的灰度图，保存在 `results_{MODEL_TYPE}/` 目录下。
  - 代码高度通用，只需修改 `MODEL_TYPE` 即可切换模型。
- **使用**：
  ```bash
  # 评估 UNet 模型
  python evaluate.py

  # 评估 CNN-LSTM 模型 (需修改 evaluate.py 中的 MODEL_TYPE)
  # MODEL_TYPE = "CNN_LSTM_Model"
  python evaluate.py
  ```
  > **注意**：在运行前，请确保 `evaluate.py` 中的 `MODEL_TYPE` 与您要评估的模型一致。

---

##  快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n landuse python=3.9
conda activate landuse
或者可以直接使用enviroment.yml一键配置

# 安装依赖
pip install torch torchvision pillow numpy tqdm matplotlib
```

### 2. 数据准备

请将您的数据集按照 `data/CLCD/` 的结构组织，确保包含 `train`, `val`, `test` 三个子集，每个子集下包含 `time1`, `time2`, 和 `label` (仅训练/验证集需要) 文件夹。

### 3. 训练模型

```bash
# 以训练 UNet 为例
python train.py --model_type UNet --epochs 50 --batch_size 4 --lr 0.001
```

### 4. 生成预测结果

```bash
# 确保 evaluate.py 中 MODEL_TYPE 设置正确
python evaluate.py
```

预测结果将自动保存在对应的 `results_*/` 文件夹中。

---

## 项目成果

- **模型权重**：在 `models/` 目录下生成的 `.pth` 文件。
- **预测结果**：在 `results_*/` 目录下生成的 `.png` 格式变化图。
- **可复现性**：提供完整的代码、数据格式说明和训练脚本，确保研究可复现。

---

## 项目成员

**项目负责人**：陈云开  
**团队成员**：邢万鑫, 方昶朔, 王仕钰, 李浩东, 魏含希, 周晓宁, 胡雅昕  
**指导老师**：王琼, 孟文娟

---

##  许可证

本项目采用 [MIT License](LICENSE) 开源协议。欢迎学习、使用和二次开发。引用本项目时，请注明来源。

