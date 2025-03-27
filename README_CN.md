# 人脸质量增强的重复数据筛选系统

[English](./README.md)

## 目录

- [人脸质量增强的重复数据筛选系统](#人脸质量增强的重复数据筛选系统)
  - [目录](#目录)
  - [项目概述](#项目概述)
  - [技术创新点](#技术创新点)
    - [1. K 折交叉验证优选人脸组合](#1-k-折交叉验证优选人脸组合)
    - [2. 基于 eDifFIQA 的增强表征](#2-基于-ediffiqa-的增强表征)
    - [3. 先进的质量评估方法](#3-先进的质量评估方法)
  - [参考文献](#参考文献)
  - [运行环境](#运行环境)
  - [环境配置与使用说明](#环境配置与使用说明)
    - [方法一：完整包安装](#方法一完整包安装)
    - [方法二：Conda 安装](#方法二conda-安装)
  - [开源协议](#开源协议)

## 项目概述

本项目基于 [eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA) 模型，结合其人脸质量评估能力，开发了一套人脸质量增强的重复数据筛选系统，用于识别和删除重复人脸图像。

人脸识别技术长期以来一直是前沿学术研究的热门领域。随着 AI 技术的进步，各种人脸识别方法都高度依赖大规模训练数据集。然而这些数据集中常存在大量冗余数据，若不加以筛选可能成为高频噪声，影响模型训练效果——不仅导致训练结果不佳，还可能使收敛速度变慢，甚至引发训练梯度的异常变化。

我们的系统通过精心训练的人脸质量评估模型，能有效提取最具区分度的人脸数据，同时过滤低区分度的冗余数据。这一过程既提高了模型训练效率，也显著加速了模型收敛，从而提升整体系统性能。

[此处插入程序流程图]

## 技术创新点

### 1. K 折交叉验证优选人脸组合

- 采用 k 折交叉验证获取最优人脸组合
- 以各人脸间距离作为评判标准
- 特别适合小批量多重复的应用场景

### 2. 基于 eDifFIQA 的增强表征

- 相比传统平均距离法，引入 eDifFIQA 模型进行质量评估
- 使用质量加权平均值作为人脸表征
- 综合考虑人脸角度、噪声、亮度、摄像头畸变等因素

### 3. 先进的质量评估方法

**核心流程**：

1. 扩散过程：使用定制 UNet 模型生成噪声和重建图像
2. 水平翻转图像重复过程以捕捉姿态影响
3. 通过嵌入比较计算质量评分

**优化方案**：

- 知识蒸馏与标签优化：
  - 利用 FR 模型嵌入空间的相对位置信息优化质量标签
  - 采用表示一致性损失(Lrc)和质量损失(Lq)提升预测能力

## 参考文献

```bibtex
@article{babnikTBIOM2024,
  title={{eDifFIQA: 基于去噪扩散概率模型的高效人脸图像质量评估}},
  author={Babnik, {\v{Z}}iga and Peer, Peter and {\v{S}}truc, Vitomir},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM)},
  year={2024},
  publisher={IEEE}
}
```

## 运行环境

- Windows 环境（支持 Python 嵌入包）
- Linux 环境（理论上支持，需自己配置 Python）

## 环境配置与使用说明

### 方法一：完整包安装

1. 通过 HuggingFace 链接下载完整包[下载链接](https://huggingface.co/scolenchris/FaceDuplicationFilter/blob/main/DJ_folder_main1.zip)
2. 解压后直接运行 Windows 下的`start.bat`

### 方法二：Conda 安装

1. 创建 Python 3.10 环境
2. 安装项目根目录`requirements.txt`中的依赖包
3. 根据[eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA)说明下载模型权重：
   - 推荐下载：`r100.pth`和`ediffiqaL.pth`
   - 放入`weights`文件夹
4. 运行`allmain.py`

## 开源协议

遵循原项目[eDifFIQA](https://github.com/LSIbabnikz/eDifFIQA)的开源协议。
