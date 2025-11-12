---
title: 视频领域技术文档
description: 本目录结构、学习路径与技术对比说明
pageType: doc
date: 20251113
---
# 视频领域技术文档

本目录包含视频生成、理解和多模态相关的核心技术文档。

## 📁 目录结构

```
video/
├── generation/           # 视频生成技术
│   ├── what-is-diffusion-model.md
│   ├── what-is-VAE.md
│   ├── what-is-DiT.md
│   ├── what-is-GAN.md
│   └── what-is-latent-diffusion.md
├── understanding/        # 视频理解技术
│   ├── what-is-ViT.md
│   ├── what-is-temporal-modeling.md
│   └── what-is-3D-convolution.md
└── multimodal/          # 多模态技术
    └── what-is-CLIP.md
```

## 🎯 学习路径

### 路径1：视频生成（文生视频/图像生成）
```
1. VAE → 2. Diffusion Model → 3. Latent Diffusion → 4. DiT
                                                        ↓
                                                    Sora理解
```

**推荐顺序：**
1. 先学VAE（理解压缩和潜在空间）
2. 再学Diffusion Model（理解扩散去噪原理）
3. 结合两者学Latent Diffusion（Stable Diffusion原理）
4. 最后学DiT（理解Sora架构）

**关联知识：**
- CLIP（文本条件）
- ViT（Transformer视觉应用）

---

### 路径2：视频理解（动作识别/分类）
```
1. Temporal Modeling → 2. 3D Convolution → 3. ViT → 4. Video Transformer
```

**推荐顺序：**
1. 先学Temporal Modeling（理解时序建模的核心问题）
2. 再学3D Convolution（经典方法）
3. 学ViT（Transformer在视觉的应用）
4. 扩展到Video Transformer（时空建模）

---

### 路径3：多模态应用（图文理解/检索）
```
1. ViT → 2. CLIP → 3. 文生图/文生视频应用
```

**推荐顺序：**
1. 先学ViT（图像编码基础）
2. 再学CLIP（图文对齐）
3. 应用到Stable Diffusion、DALL-E等

---

## 📊 技术对比

### 生成模型对比

| 模型 | 生成质量 | 速度 | 训练难度 | 主要应用 |
|------|---------|------|---------|---------|
| **GAN** | 高 | 快（1步） | 难（不稳定） | 图像翻译、人脸生成 |
| **VAE** | 中 | 快 | 简单 | 数据压缩、特征学习 |
| **Diffusion** | 最高 | 慢（50-1000步） | 中 | 文生图、文生视频 |
| **Latent Diffusion** | 最高 | 中（快64倍） | 中 | Stable Diffusion |

### 视频理解方法对比

| 方法 | 时空建模 | 计算复杂度 | 主要应用 |
|------|---------|-----------|---------|
| **光流 + 2D CNN** | 分离 | 高（光流计算） | Two-Stream |
| **3D CNN** | 联合 | 高 | C3D、I3D |
| **RNN/LSTM** | 串行 | 中 | 序列处理 |
| **Transformer** | 并行 | 高 | TimeSformer、ViViT |

---

## 🔗 文档间关系图

```
[Transformer基础] (已有 guide/ai/)
         ↓
    ┌────┴────┐
    ↓         ↓
  [ViT]    [CLIP]
    ↓         ↓
  [DiT]  [Latent Diffusion]
    ↓         ↓
  [Sora]  [Stable Diffusion]

[VAE] → [Latent Diffusion]
         ↓
    [Stable Diffusion]

[Diffusion Model]
    ↓
┌───┴───┐
↓       ↓
[DiT] [Latent Diffusion]

[Temporal Modeling]
    ↓
┌───┴───┐
↓       ↓
[3D Conv] [Video Transformer]
```

---

## 🎓 前置知识

建议先学习以下基础知识（在 `guide/ai/` 目录）：

- **必需**：
  - [什么是Transformer](../ai/what-is-transformer.md)
  - [什么是Multi-Head Attention](../ai/what-is-multi-head-attention.md)
  - [什么是Encoder-Decoder架构](../ai/what-is-encoder-decoder.md)

- **推荐**：
  - [什么是向量嵌入](../ai/what-is-vector-embedding.md)
  - [什么是表示空间](../ai/what-is-representation-space.md)

---

## 📝 配图说明

每篇文档顶部都有配图占位符（`<!-- TODO: ... -->`），详细的配图需求见 [IMAGE_TODO.md](./IMAGE_TODO.md)。

图片存放位置：`/assets/images/video/`

---

## 🚀 实际应用

### 文生图/文生视频
- **Stable Diffusion**：VAE + Latent Diffusion + CLIP
- **Sora**：DiT + Spacetime Patch + VAE
- **DALL-E 2**：CLIP + Diffusion

### 视频理解
- **动作识别**：3D CNN、SlowFast、TimeSformer
- **视频分类**：I3D、R(2+1)D

### 图像-文本任务
- **图文检索**：CLIP
- **图像描述**：CLIP + GPT
- **视觉问答**：ViT + Transformer

---

## 📚 扩展阅读

### 经典论文
1. **Attention Is All You Need** (2017) - Transformer原论文
2. **An Image is Worth 16x16 Words** (2020) - ViT
3. **Denoising Diffusion Probabilistic Models** (2020) - DDPM
4. **High-Resolution Image Synthesis with Latent Diffusion Models** (2021) - Stable Diffusion
5. **Learning Transferable Visual Models From Natural Language Supervision** (2021) - CLIP
6. **Scalable Diffusion Models with Transformers** (2022) - DiT

### 开源项目
- **Stable Diffusion**：https://github.com/CompVis/stable-diffusion
- **CLIP**：https://github.com/openai/CLIP
- **TimeSformer**：https://github.com/facebookresearch/TimeSformer

---

## ⚠️ 注意事项

1. 所有文档都包含"大白话解释"和"正文"两部分
2. 作者标注为"AI收集"，表示AI辅助创作
3. 文档持续更新中，欢迎贡献改进

---

**最后更新**：2025-11-13
