---
title: 什么是 Diffusion Model（扩散模型）
description: 什么是 Diffusion Model（扩散模型）
pageType: doc
date: 20251112
author: AI收集
plainLanguage: |
  把噪声当画布，一步步“擦干净”生成清晰内容。
  稳、细、可控，适合文生图/视频等多场景。
---

  想象你有一张照片，你故意把它弄脏：

  **前向扩散过程（加噪声）：**
  ```
  原始清晰图片
    ↓ 加一点噪声（步骤1）
  稍微模糊的图片
    ↓ 再加噪声（步骤2）
  更模糊的图片
    ↓ 继续加噪声...（步骤3-1000）
  完全是雪花噪声（啥都看不出来）
  ```

  **反向去噪过程（生成）：**
  ```
  纯噪声（雪花图）
    ↓ AI去噪（步骤1）："这里好像有个轮廓"
  隐约看出形状
    ↓ 继续去噪（步骤2）："这是一只猫"
  看出大致轮廓
    ↓ 精细去噪...（步骤3-1000）
  高清猫咪图片
  ```

  ---

  **核心流程（以图像生成为例）：**

  ```
  训练阶段：
  1. 拿一张猫的照片
  2. 加噪声1000次 → 变成雪花
  3. 训练AI学会"每一步应该去掉多少噪声"
  4. 重复百万张图片，AI学会去噪

  生成阶段：
  1. 从纯噪声开始
  2. AI预测："这步应该去掉这些噪声"
  3. 去掉一点噪声 → 图片清晰一点
  4. 重复1000次
  5. 最终生成全新的猫咪图片
  ```

  ---

  **与GAN的对比：**

  **GAN（生成对抗网络）：**
  ```
  Generator（生成器）："我画了只猫"
  Discriminator（鉴别器）："假的！猫耳朵不对"
  Generator："我改"
  Discriminator："还是假的！毛色不对"
  （反复对抗，直到鉴别器分不出真假）
  ```
  - 优点：生成速度快
  - 缺点：训练不稳定、容易模式崩塌

  **Diffusion Model（扩散模型）：**
  ```
  训练："学会如何一步步去噪"
  生成："从噪声开始，逐步去噪1000次"
  ```
  - 优点：训练稳定、生成质量高、多样性好
  - 缺点：生成慢（需要1000步）

  ---

  **关键概念：**

  **1. 前向扩散（Forward Diffusion）**
  - 逐步加噪声的过程
  - 有固定公式（马尔可夫链）
  - 最终变成纯高斯噪声

  **2. 反向去噪（Reverse Denoising）**
  - 逐步去噪的过程
  - 需要AI学习（神经网络预测噪声）
  - 从噪声恢复出清晰图像

  **3. 噪声调度（Noise Schedule）**
  - 控制每一步加多少噪声
  - 常见：线性调度、余弦调度
  - 影响生成质量

  ---

  **实际应用：**

  **1. 图像生成**
  - Stable Diffusion：文生图
  - DALL-E 2：文生图
  - Midjourney：文生图

  **2. 视频生成**
  - Sora：文生视频（OpenAI）
  - Runway Gen-2：文生视频
  - Pika：视频编辑

  **3. 音频生成**
  - AudioLDM：文生音频
  - DiffWave：语音合成

  **4. 3D生成**
  - DreamFusion：文生3D模型
  - Point-E：点云生成

  ---

  **加速技巧：**

  **问题：1000步太慢了！**

  **解决方案：**

  **1. DDIM（去马尔可夫假设）**
  - 原本1000步 → 只需50步
  - 质量几乎不损失

  **2. Latent Diffusion（潜在空间扩散）**
  - 不在像素空间做扩散
  - 先用VAE压缩到潜在空间（小64倍）
  - 在小空间做扩散 → 速度快64倍
  - Stable Diffusion就是这个原理

  **3. Consistency Models**
  - 一步生成（最新研究）
  - 牺牲一点质量换速度

  ---

  **文本控制（Text-to-Image）：**

  **如何让AI按文本生成图片？**

  ```
  用户输入："一只戴墨镜的猫"
    ↓
  文本编码器（CLIP）：
    "一只戴墨镜的猫" → [文本向量]
    ↓
  Diffusion模型去噪时：
    每一步都"看"文本向量
    通过Cross-Attention机制
    "墨镜"引导生成墨镜
    "猫"引导生成猫
    ↓
  生成：一只戴墨镜的猫
  ```

  ---

  **数学本质（简化版）：**

  **前向扩散：**
  
  ```math
  q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \cdot I)
  ```

  意思：下一步 = 上一步 * 缩放 + 噪声

  **反向去噪：**
  
  ```math
  p(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
  ```

  意思：AI预测"去噪后应该是什么样"

  ---

  **常见误区：**

  **误区1："Diffusion就是加噪声去噪声"**
  - 不完全对！关键是"学会预测噪声"
  - 预测噪声 = 知道如何去噪

  **误区2："步数越多越好"**
  - 不对！1000步后已经收敛
  - 再多步也没用，反而浪费时间

  **误区3："Diffusion比GAN好"**
  - 看任务：
    - 质量优先 → Diffusion
    - 速度优先 → GAN
    - 实时生成 → GAN

  说白了，Diffusion Model就是"去噪大师"——通过学习如何一步步去噪，最终能从纯噪声生成高质量图片/视频。虽然慢，但质量和稳定性远超GAN，是当前生成模型的主流选择。
---


<!-- TODO: 添加扩散模型示意图，展示前向扩散（加噪）和反向去噪过程 -->
![diffusion-model-process](/assets/images/video/diffusion-model-process.svg)

扩散模型：把“噪声”当画布，一步步擦掉它，直到世界显形。稳、细、可控，是它的性格。

## 为什么它重要（≤5min）
- 生成要稳、细节要真：对比GAN更稳定、对比VAE更清晰。
- 易于加条件：文本/类别/控制信号都能优雅接入。

## 怎么做（一图一流程）
- 加噪（固定）：`x_0 → x_T`（逐步加高斯噪声）
- 去噪（学习）：`x_T → x_0`（网络预测噪声，逐步还原）
- 加速：DDIM（50步左右），或在潜在空间做扩散（LDM）。

![diffusion-model-process](/assets/images/video/diffusion-model-process.svg)

## 优缺点（一句到位）
- 优点：训练稳定、细节好、可控性强。
- 缺点：推理需多步，算力成本高。

## 易错点（别踩坑）
- 把“去噪”当目标：本质是“预测噪声”，去噪是随之而来。
- 盲目拉步数：超过收敛点不涨质量还浪费时间。

## 适用场景（马上用得上）
- 文生图/视频：Stable Diffusion、Sora 等
- 音频/语音合成：DiffWave 系列
- 图像编辑：inpainting / image2image

## 参考图源（供重绘或嵌入）
- Wikipedia Diffusion model词条概述图与流程（适合抽取关键框架）
- HuggingFace Annotated Diffusion流程图（条理清晰，便于转为SVG）

## 一页总结
- 两条链：加噪固定、去噪学习；关键在“噪声预测”。
- 速度策略：DDIM跳步、潜在扩散降维。
- 文本条件：用CFG 3–8间调优，平衡符合度与多样性。

## 核心建模（DDPM视角）

- 前向加噪（固定）：
```math
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t; \sqrt{1-\beta_t}\,x_{t-1},\; \beta_t\,I\big)
```

- 训练目标（预测噪声）：
```math
L = \big\|\varepsilon - \varepsilon_\theta(x_t, t)\big\|^2
```

- 采样初始化（从纯噪声出发）：
```math
x_T \sim \mathcal{N}(0, I)
```

- 迭代更新（反向去噪）：
```math
x_{t-1} = \frac{x_t - \beta_t/\sqrt{1-\bar{\alpha}_t}\cdot\varepsilon_{pred}}{\sqrt{1-\beta_t}} + \sigma_t\cdot z
```

- 调度与加速：
  - 噪声调度：线性/余弦等（影响质量与稳定性）
  - DDIM：跳步与确定性采样，20–50步可达较好效果（与原DDPM一致的训练目标，采样公式改造）

## 核心概念

### 扩散模型的设计思想

**问题：** 如何生成复杂的高维数据（图像、视频）？

**传统方法的问题：**
- **GAN**：训练不稳定，模式崩塌
- **VAE**：生成质量一般，图像模糊
- **Flow-based**：架构限制，表达能力弱

**Diffusion的解决方案：**
```
训练：学习如何逐步去噪
生成：从噪声开始，逐步去噪T步
```

### 两个核心过程

**前向扩散过程（Forward Diffusion Process）：**
- 功能：逐步向数据添加高斯噪声
- 目标：将数据分布转换为标准高斯分布
- 特点：过程固定，有解析解

**反向去噪过程（Reverse Denoising Process）：**
- 功能：逐步从噪声中恢复数据
- 目标：学习数据分布
- 特点：需要神经网络学习

## 工作流程

### 完整流程示意（图像生成）

**训练阶段：**

```
步骤1：前向扩散（数据 → 噪声）

原始图像 x_0（猫的照片）
  ↓ t=1，加少量噪声
x_1（稍微模糊的猫）
  ↓ t=2，继续加噪声
x_2（更模糊的猫）
  ↓ ...
  ↓ t=1000，加很多噪声
x_1000（纯高斯噪声，完全看不出猫）

```

数学表示：

```math
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \cdot I)
```

其中：
- $\beta_t$：噪声调度参数（控制加多少噪声）
- $\mathcal{N}$：高斯分布
- $I$：单位矩阵

---

步骤2：训练去噪网络

对于每个时间步t：
  输入：x_t（有噪声的图像）+ t（时间步）
  目标：预测噪声 ε

  损失函数：
  
  ```math
  L = \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2
  ```

  意思：预测的噪声要和真实噪声接近


**生成阶段**


步骤1：从纯噪声开始

```math
x_T \sim \mathcal{N}(0, I)
```

（随机采样高斯噪声）



步骤2：迭代去噪
```
for t = T 到 1:
  # 预测噪声
  ε_pred = ε_θ(x_t, t)

  # 去噪（计算x_{t-1}）见下式
```

```math
x_{t-1} = \frac{x_t - \beta_t/\sqrt{1-\bar{\alpha}_t} \cdot \varepsilon_{pred}}{\sqrt{1-\beta_t}} + \sigma_t \cdot z
```

其中：
- $z \sim \mathcal{N}(0, I)$：随机性（保证多样性）
- $\sigma_t$：噪声方差

---

步骤3：输出

x_0 → 生成的清晰图像


### Diffusion 时序图（精简）

- 流程：加噪固定 → 学习去噪 → 迭代更新 → 输出图像
- 详见上文“核心建模（DDPM视角）”与“怎么做（一图一流程）”

### 关键机制：噪声调度（Noise Schedule）

**什么是噪声调度？**

控制每一步加多少噪声的策略。

**线性调度（Linear Schedule）：**
```
β_1 = 0.0001
β_2 = 0.0002
...
β_T = 0.02

均匀增加
```

**余弦调度（Cosine Schedule）：**

```math
\alpha_t = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)
```

前期加噪慢，后期加噪快

**为什么重要？**
- 影响生成质量
- 影响训练稳定性
- 余弦调度通常效果更好

## DDPM vs DDIM

### DDPM（Denoising Diffusion Probabilistic Models）

**特点：**
- 原始Diffusion模型
- 需要T步（通常1000步）
- 每步都有随机性
- 生成质量高但慢

**生成公式：**

```math
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \cdot z
```

其中 $z \sim \mathcal{N}(0, I)$（随机项）

### DDIM（Denoising Diffusion Implicit Models）

**创新点：** 去掉马尔可夫假设，允许跳步

**特点：**
- 可以只用50步（原本1000步）
- 确定性生成（去掉随机项）
- 速度快20倍，质量几乎不降

**生成公式：**

```math
x_{t-1} = \sqrt{\alpha_{t-1}} \cdot \frac{x_t - \sqrt{1-\alpha_t} \cdot \varepsilon_\theta(x_t,t)}{\sqrt{\alpha_t}} + \sqrt{1-\alpha_{t-1}} \cdot \varepsilon_\theta(x_t,t)
```

无随机项 $z$

**对比：**

| 特性 | DDPM | DDIM |
|------|------|------|
| **步数** | 1000 | 50-100 |
| **速度** | 慢 | 快20倍 |
| **随机性** | 有 | 可选 |
| **质量** | 高 | 几乎一样 |
| **应用** | 研究 | 实际产品 |

## 文本条件控制（精简）

```math
\varepsilon_{guided} = \varepsilon_\theta(x_t, t, \emptyset) + w\cdot\big(\varepsilon_\theta(x_t, t, c) - \varepsilon_\theta(x_t, t, \emptyset)\big)
```

- 训练：同一个模型学习有条件/无条件两种预测
- 采样：用 `w≈3–8` 平衡文本符合度与多样性（常用 7.5）

## Latent Diffusion（潜在扩散）

### 核心思想

**问题：** 在像素空间做Diffusion太慢（512×512图像 = 262,144维）

**解决方案：** 在压缩的潜在空间做Diffusion

### 架构

```
原始图像（512×512×3）
  ↓
VAE Encoder
  ↓
潜在表示（64×64×4）← 缩小64倍！
  ↓
Diffusion过程（在这里做）
  ↓
去噪后的潜在表示
  ↓
VAE Decoder
  ↓
生成图像（512×512×3）
```

**优势：**
- 速度快64倍（空间缩小8倍，维度缩小64倍）
- 内存省64倍
- 质量几乎不降

**代表模型：**
- Stable Diffusion
- DALL-E 2（部分使用）

## 实际应用（精简）
- Stable Diffusion：CLIP 文本条件 + U-Net 去噪（潜在空间）+ VAE 解码
- Sora：DiT 骨干 + 时空 Patch，视频更连贯（详见 DiT 文档）
- DALL-E 2：CLIP 对齐 + 先验/解码两阶段 Diffusion

## 加速技巧（精简）
- DDIM：跳步/确定性采样（常用 20–50 步）
- Latent Diffusion：潜在空间扩散（速度/显存友好）
- Distillation/Consistency：进一步压步或一/少步生成（质量/实现难度需权衡）

## FAQ 速答（精简）
- 与 GAN 相比：更稳、更细节；但采样多步、速度慢
- 预测噪声而非 `x_0`：更易训练、效果更好（与 score matching 等价）
- 实时方向：Consistency/蒸馏/专用硬件（取舍质量与速度）

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM原论文
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) - DDIM论文
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) - Stable Diffusion论文
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [什么是VAE](/guide/video/generation/what-is-VAE) - 本站相关文章
- [什么是DiT](/guide/video/generation/what-is-DiT) - 本站相关文章
