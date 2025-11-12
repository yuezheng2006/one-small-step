---
title: 视频领域文档配图需求清单
description: 统一的配图规范与待补图片列表
pageType: doc
date: 20251113
---
# 视频领域文档配图需求清单

本目录包含9个视频领域的核心技术文档，以下是配图需求说明。

## 图片存放位置

所有图片统一存放在：`/assets/images/video/`

## 配图需求列表

### 1. 生成模型基础（generation/）

#### 1.1 Diffusion Model（扩散模型）
- **文件路径**：`generation/what-is-diffusion-model.md`
- **图片名称**：`diffusion-model-process.svg`
- **内容需求**：
  - 展示前向扩散过程（清晰图像 → 噪声）
  - 展示反向去噪过程（噪声 → 清晰图像）
  - 标注时间步 t=0, t=500, t=1000
  - 建议使用左右对比或上下流程图
- **参考**：DDPM论文图1、Stable Diffusion官网示意图
 - **外部参考链接（可重绘为统一风格SVG）**：
   - Wikipedia Diffusion model 词条流程示意（https://en.wikipedia.org/wiki/Diffusion_model）
   - HuggingFace Annotated Diffusion 流程图（https://huggingface.co/blog/annotated-diffusion）

#### 1.2 VAE（变分自编码器）
- **文件路径**：`generation/what-is-VAE.md`
- **图片名称**：`vae-architecture.svg`
- **内容需求**：
  - Encoder（输入图像 → μ, σ）
  - Latent Space（潜在向量z，采样过程）
  - Decoder（z → 重建图像）
  - 标注维度（如512×512×3 → 64×64×4 → 512×512×3）
- **参考**：VAE原论文图1、Stable Diffusion技术报告
 - **外部参考链接（可重绘为统一风格SVG）**：
   - ML Glossary VAE/AE架构示意（https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html）
   - 教程类文章示意图（用于提取布局，不直接使用版权图）（例如：https://towardsdatascience.com/vae-variational-autoencoders-how-to-employ-neural-networks-to-generate-new-images-bdeb216ed2c0/）

#### 1.3 DiT（Diffusion Transformer）
- **文件路径**：`generation/what-is-DiT.md`
- **图片名称**：`dit-architecture.svg`
- **内容需求**：
  - 左侧：传统U-Net架构（下采样-上采样）
  - 右侧：DiT架构（Patch Embedding + Transformer Blocks）
  - 对比参数量、计算复杂度
  - 标注AdaLN、Cross-Attention等关键组件
- **参考**：DiT论文图2、Sora技术报告
 - **外部参考链接（可重绘为统一风格SVG）**：
   - DiT 论文图示（https://arxiv.org/abs/2212.09748）

#### 1.4 GAN（生成对抗网络）
- **文件路径**：`generation/what-is-GAN.md`
- **图片名称**：`gan-adversarial-training.svg`
- **内容需求**：
  - Generator（噪声z → 假图像）
  - Discriminator（真图像/假图像 → 真假判别）
  - 对抗箭头（Generator试图欺骗Discriminator）
  - 训练循环示意
- **参考**：GAN原论文图1、DCGAN架构图
 - **外部参考链接（可重绘为统一风格SVG）**：
   - DCGAN/StyleGAN 博文与论文示意图（便于抽取生成/判别模块关系）

#### 1.5 Latent Diffusion（潜在扩散）
- **文件路径**：`generation/what-is-latent-diffusion.md`
- **图片名称**：`latent-diffusion-pipeline.svg`
- **内容需求**：
  - 完整流程：图像 → VAE Encoder → 潜在空间扩散 → VAE Decoder → 生成图像
  - 标注维度变化（512×512×3 → 64×64×4 → 64×64×4 → 512×512×3）
  - 标注文本条件注入位置（CLIP Text Encoder → U-Net）
  - Stable Diffusion完整pipeline
- **参考**：Latent Diffusion论文图3、Stable Diffusion架构图
 - **外部参考链接（可重绘为统一风格SVG）**：
   - LDM 论文图示（https://arxiv.org/abs/2112.10752）
   - HuggingFace Annotated Diffusion（流程元素可复用）（https://huggingface.co/blog/annotated-diffusion）

---

### 2. 视频理解（understanding/）

#### 2.1 ViT（Vision Transformer）
- **文件路径**：`understanding/what-is-ViT.md`
- **图片名称**：`vit-patch-embedding.svg`
- **内容需求**：
  - 图像分块过程（224×224图像 → 196个16×16 Patch）
  - Patch Embedding（线性投影）
  - 添加[CLS] Token和位置编码
  - Patch序列示意
- **参考**：ViT原论文图1、An Image is Worth 16x16 Words
 - **外部参考链接（可重绘为统一风格SVG）**：
   - ViT 论文示意（https://arxiv.org/abs/2010.11929）

#### 2.2 Temporal Modeling（时序建模）
- **文件路径**：`understanding/what-is-temporal-modeling.md`
- **图片名称**：`temporal-modeling-methods.svg`
- **内容需求**：
  - 四种方法对比：
    1. 光流（Optical Flow）：显式运动向量
    2. 3D卷积：时空联合卷积核
    3. RNN/LSTM：时序循环处理
    4. Temporal Attention：帧间注意力
  - 每种方法的感受野、复杂度标注
- **参考**：时序建模综述图
 - **外部参考链接（可重绘为统一风格SVG）**：
   - TimeSformer（https://arxiv.org/abs/2102.05095）
   - SlowFast（https://arxiv.org/abs/1812.03982）

#### 2.3 3D Convolution（3D卷积）
- **文件路径**：`understanding/what-is-3D-convolution.md`
- **图片名称**：`3d-convolution-kernel.svg`
- **内容需求**：
  - 左侧：2D卷积核（3×3，单帧）
  - 右侧：3D卷积核（3×3×3，多帧）
  - 立方体卷积核滑动示意
  - 时空感受野对比
- **参考**：C3D论文图1、I3D架构图
 - **外部参考链接（可重绘为统一风格SVG）**：
   - C3D（https://arxiv.org/abs/1412.0767）
   - I3D（https://arxiv.org/abs/1705.07750）

---

### 3. 多模态（multimodal/）

#### 3.1 CLIP（对比语言-图像预训练）
- **文件路径**：`multimodal/what-is-CLIP.md`
- **图片名称**：`clip-contrastive-learning.svg`
- **内容需求**：
  - 图像编码器（ViT）和文本编码器（Transformer）并行
  - 相似度矩阵（N×N，对角线高亮）
  - 对比学习目标（匹配对相似度高，非匹配对低）
  - 零样本分类流程
- **参考**：CLIP原论文图1、OpenAI CLIP博客图
 - **外部参考链接（可重绘为统一风格SVG）**：
   - CLIP 论文（https://arxiv.org/abs/2103.00020）
   - OpenAI CLIP Blog（https://openai.com/research/clip）

---

## 图片规格建议

### 尺寸
- 宽度：800-1200px（适合网页展示）
- 高度：根据内容自适应，建议不超过800px
- 格式：PNG（支持透明背景）或 WebP（更小体积）

### 风格
- 简洁清晰，避免过多细节
- 使用统一配色方案
- 关键部分用高亮色标注
- 添加必要的文字标注（中文）
- 背景：白色或浅色

### 工具推荐
- **绘图**：Figma、Draw.io、Excalidraw
- **截图**：论文原图（需标注来源）
- **AI生成**：可以用Stable Diffusion生成概念图

---

## 临时方案

在正式配图完成前，文档可以正常使用，图片会显示为占位符或无法加载。建议：

1. **优先级1（核心概念）**：Diffusion Model、VAE、DiT
2. **优先级2（应用广泛）**：ViT、CLIP、Latent Diffusion
3. **优先级3（补充理解）**：GAN、Temporal Modeling、3D Convolution

---

## 更新日志

- **2025-11-12**：初始化图片需求清单，创建9个文档的占位符
