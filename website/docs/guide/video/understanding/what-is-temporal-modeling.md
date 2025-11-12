---
title: 什么是时序建模（Temporal Modeling）
description: 什么是时序建模（Temporal Modeling）在视频理解和生成中的应用
pageType: doc
date: 20251112
author: AI收集
plainLanguage: |
  把“时间”这条线看懂：视频不只是多张图，还要理解变化。
  选型贴场景：实时/高精度/小数据用不同方法。
---

  **图像理解 vs 视频理解：**
  ```
  图像（静态）：
    看一张猫的照片
    AI：这是一只猫

  视频（动态）：
    看10帧猫的视频：
      帧1：猫站着
      帧2：猫弯腿
      帧3：猫跳起
      帧4：猫在空中
      帧5：猫落地
      ...

    时序建模：
      - 识别动作：猫在跳跃
      - 理解运动轨迹：从地面→空中→地面
      - 预测下一帧：猫会落地后站稳
  ```

  ---

  **核心问题：视频 vs 图像的区别**

  ```
  图像：
    维度：H × W × C（高×宽×通道）
    例如：224×224×3 = 150,528个值

  视频：
    维度：T × H × W × C（时间×高×宽×通道）
    例如：10帧 × 224×224×3 = 1,505,280个值

  新增挑战：
    - 时间维度T（帧序列）
    - 帧间关系（运动、变化）
    - 长期依赖（远距离帧的关系）
  ```

  ---

  **时序建模的三大问题：**

  **1. 运动建模（Motion Modeling）**
  ```
  问题：如何理解物体的运动？

  例子：球从左飞到右
    帧1：球在左边
    帧2：球在中间
    帧3：球在右边

  时序建模目标：
    - 提取运动信息（向右移动）
    - 理解速度（每帧移动多少）
    - 预测轨迹（下一帧球在哪）
  ```

  **2. 时间依赖（Temporal Dependency）**
  ```
  问题：如何关联不同时刻的帧？

  例子：打篮球动作序列
    帧1-5：起跳准备
    帧6-10：空中投篮
    帧11-15：落地

  时序建模目标：
    - 短期依赖：连续帧（帧1和帧2）
    - 长期依赖：远距离帧（帧1和帧15）
    - 理解动作的完整性（整个投篮动作）
  ```

  **3. 时序一致性（Temporal Consistency）**
  ```
  问题：生成的视频如何保持连贯？

  例子：生成"猫走路"视频
    错误（无时序建模）：
      帧1：猫左脚抬起
      帧2：猫突然消失
      帧3：猫右脚抬起
      → 闪烁、跳跃、不连贯

    正确（有时序建模）：
      帧1：猫左脚抬起
      帧2：猫左脚落下
      帧3：猫右脚抬起
      → 平滑、自然
  ```

  ---

  **时序建模的方法：**

  **方法1：光流（Optical Flow）**
  ```
  思想：显式计算像素的运动

  流程：
    帧t和帧t+1
      ↓
    光流算法（如FlowNet）
      ↓
    运动向量场：每个像素的移动方向和距离
      例如：像素(100, 100)向右移动5像素

  应用：
    - 动作识别（Two-Stream网络）
    - 视频插帧（中间帧生成）
    - 目标跟踪
  ```

  **方法2：3D卷积（3D Convolution）**
  ```
  思想：空间和时间一起卷积

  2D卷积（图像）：
    卷积核：3×3（空间）
    看：一个位置的邻域

  3D卷积（视频）：
    卷积核：3×3×3（时间×高×宽）
    看：一个位置在时间和空间的邻域

  例子：
    输入：5帧视频（5×224×224×3）
    3D卷积核：3×3×3
    同时看：3帧×3×3区域

  优势：
    - 同时建模时空信息
    - 自动学习运动模式

  代表：C3D、I3D
  ```

  **方法3：循环神经网络（RNN/LSTM）**
  ```
  思想：逐帧处理，保持记忆

  流程：
    帧1 → CNN提取特征 → f₁
      ↓
    LSTM（记忆f₁）
      ↓
    帧2 → CNN提取特征 → f₂
      ↓
    LSTM（记忆f₁+f₂）
      ↓
    帧3 → CNN提取特征 → f₃
      ↓
    LSTM（记忆f₁+f₂+f₃）
      ↓
    最终输出：综合所有帧的理解

  优势：
    - 处理可变长度视频
    - 捕捉长期依赖

  缺点：
    - 训练慢（串行处理）
    - 梯度消失/爆炸
  ```

  **方法4：时序注意力（Temporal Attention）**
  ```
  思想：让模型"注意"重要的帧

  流程：
    所有帧特征：[f₁, f₂, ..., f_T]
      ↓
    Self-Attention：
      每帧看所有其他帧
      计算注意力权重

    例如，识别"投篮"动作：
      帧5（起跳）：高权重
      帧10（投篮瞬间）：最高权重
      帧2（准备）：中权重
      帧15（落地）：低权重
      ↓
    加权求和：重要帧贡献大

  优势：
    - 并行处理（快）
    - 自动学习重要时刻
    - 可视化注意力权重

  代表：Video Transformer、TimeSformer
  ```

  **方法5：因果卷积（Causal Convolution）**
  ```
  思想：只看过去，不看未来（用于生成）

  普通卷积：
    帧t-1、帧t、帧t+1 → 预测帧t
    （看未来，不能用于生成）

  因果卷积：
    帧t-2、帧t-1、帧t → 预测帧t+1
    （只看过去，适合生成）

  应用：
    - 视频生成（逐帧生成）
    - 实时预测
  ```

  ---

  **时序建模的架构：**

  **1. Two-Stream（双流网络）**
  ```
  并行处理空间和时间：

  空间流（Spatial Stream）：
    输入：单帧RGB图像
    网络：2D CNN
    学习："这是什么"（猫、球、车）

  时间流（Temporal Stream）：
    输入：光流图（多帧运动）
    网络：2D CNN
    学习："在做什么"（跑、跳、转）

  融合：
    空间特征 + 时间特征 → 最终预测

  应用：
    - 动作识别
    - UCF-101、HMDB-51数据集
  ```

  **2. 3D CNN（C3D、I3D）**
  ```
  统一的时空卷积：

  C3D：
    输入：16帧 × 112×112×3
    架构：8层3D卷积
    卷积核：3×3×3（全部3D）

  I3D（Inflated 3D）：
    思想："膨胀"2D卷积为3D
    方法：
      2D卷积核：3×3
      → "膨胀"为：3×3×3
      → 用ImageNet预训练权重初始化

    优势：
      - 利用2D预训练（ImageNet）
      - 效果比从头训练好
  ```

  **3. Transformer（TimeSformer、ViViT）**
  ```
  纯Attention机制：

  TimeSformer：
    输入：视频切成Patch（16×16像素×2帧）
    架构：
      1. Patch Embedding
      2. 时序Attention（帧间关系）
      3. 空间Attention（帧内关系）
      4. 交替堆叠

  优势：
    - 全局感受野（第1层就能看全视频）
    - 并行计算
    - 扩展性好
  ```

  ---

  **视频生成的时序建模：**

  **1. 帧预测（Frame Prediction）**
  ```
  任务：给定前N帧，预测第N+1帧

  方法：
    ConvLSTM：
      帧1, 帧2, ..., 帧N
        ↓
      LSTM逐帧处理
        ↓
      预测帧N+1

  应用：
    - 视频插帧
    - 异常检测（预测偏差大→异常）
  ```

  **2. 视频扩散（Video Diffusion）**
  ```
  任务：生成连贯视频

  挑战：
    - 逐帧生成：容易闪烁
    - 需要时序一致性

  解决方案：
    Spacetime Patch（时空块）：
      不是分别处理每帧
      而是把多帧一起处理

      例如（Sora）：
        Patch：16×16像素 × 2帧
        → 同时建模空间和时间

    3D U-Net：
      把2D U-Net扩展为3D
      同时在时间和空间去噪
  ```

  **3. 自回归生成（Autoregressive）**
  ```
  任务：逐帧生成视频

  方法：
    已有：帧1, 帧2, ..., 帧t
      ↓
    生成模型（如Transformer）
      ↓
    预测：帧t+1
      ↓
    重复：帧t+1 → 帧t+2 → ...

  挑战：
    - 误差累积（前期错误影响后期）
    - 长视频生成慢

  代表：
    - VideoGPT
    - CogVideo
  ```

  ---

  **常见误区：**

  **误区1："3D卷积比2D卷积好"**
  - 不一定！
    - 3D卷积：参数多，需要大数据
    - 2D卷积+时序建模：参数少，训练快
    - 看任务和数据量

  **误区2："Transformer解决了所有问题"**
  - 不完全对！
    - Transformer：需要大数据、大算力
    - 小数据、小模型：3D CNN可能更好
    - 归纳偏置有帮助

  **误区3："视频就是多个图像"**
  - 大错特错！
    - 视频有时序关系（运动、因果）
    - 不建模时序，损失关键信息
    - 必须考虑时间维度

  说白了，时序建模就是"理解时间变化"——通过光流、3D卷积、RNN、Attention等方法，让AI理解视频中的运动、变化和因果关系，从而实现动作识别、视频生成等任务。是视频理解和生成的核心技术。
---


<!-- TODO: 添加时序建模方法对比图，展示2D vs 3D vs RNN vs Attention -->
![temporal-modeling-methods](/assets/images/video/temporal-modeling-methods.svg)

时序建模：把时间看懂，视频才算被理解；方法选得准，连贯自然来。

## 为什么它重要（≤5min）
- 图像回答“是什么”，视频要回答“怎么变化”。

## 常用路线（选型速览）
- 光流：显式度量“往哪儿动”，配2D CNN。
- 3D卷积：时空一起学，端到端。
- RNN/LSTM：逐帧记忆，适合长序列。
- Temporal Attention：并行看全时序，精度高。
- 因果卷积：只看过去，用于生成/实时。

![temporal-modeling-methods](/assets/images/video/temporal-modeling-methods.svg)

## 易错点（别踩坑）
- 只用2D：忽略时间会丢关键语义。
- 一刀切：任务不同，方法要换；实时 vs 高精度优先级不同。

## 适用场景（马上用得上）
- 动作识别、视频生成、异常检测、插帧等。

## 参考图源（供重绘或嵌入）
- TimeSformer/SlowFast 论文图（可提炼为方法对比SVG）

## 一页总结
- 时间是第二维；方法要贴场景；一致性是关键。

## 核心概念

### 时序建模的重要性

**视频 vs 图像的本质区别：**
```
图像：
  - 维度：H × W × C
  - 信息：空间语义（what, where）
  - 任务：分类、检测、分割

视频：
  - 维度：T × H × W × C
  - 信息：空间语义 + 时间动态（when, how）
  - 任务：动作识别、视频生成、预测

关键差异：
  - 时间轴T引入运动信息
  - 帧间关系（短期、长期依赖）
  - 时序一致性要求
```

### 时序建模的三大挑战

**1. 运动表示（Motion Representation）：**
```
问题：如何编码像素/物体的运动？

方法：
  - 显式：光流（Optical Flow）
  - 隐式：时序卷积、Attention
```

**2. 时间依赖（Temporal Dependency）：**
```
问题：如何建模不同时间尺度的依赖？

类型：
  - 短期：连续帧（1-5帧）
  - 中期：局部片段（10-30帧）
  - 长期：全视频（100+帧）
```

**3. 计算效率（Computational Efficiency）：**
```
问题：视频数据量大（T×H×W×C）

挑战：
  - 内存：存储所有帧
  - 计算：处理时空信息
  - 训练：视频数据标注成本高
```

## 时序建模方法

### 方法1：光流（Optical Flow）

**原理：** 显式计算像素级运动

**光流定义：**
```
假设：
  - 亮度恒定：I(x,y,t) = I(x+dx, y+dy, t+dt)
  - 小位移：dx, dy很小

光流：
  v = (u, v)  # 每个像素的运动向量

计算：
  传统方法：Farneback算法、Lucas-Kanade
  深度学习：FlowNet、PWC-Net、RAFT
```

**Two-Stream架构：**
```
空间流（RGB）：
  输入：单帧图像
  网络：2D CNN
  输出：空间特征（物体外观）

时间流（光流）：
  输入：光流图（10帧堆叠）
  网络：2D CNN
  输出：时间特征（运动模式）

融合：
  final = αfeatures_spatial + (1-α)*features_temporal
```

**优缺点：**
```
优点：
  - 显式运动表示
  - 可解释性强
  - 适合动作识别

缺点：
  - 计算光流成本高
  - 光流估计误差累积
  - 两个流需要分别训练
```

### 方法2：3D卷积（3D Convolution）

**原理：** 时空联合卷积

**2D vs 3D卷积：**
```
2D卷积（图像）：
  卷积核：k_h × k_w
  输入：H × W × C_in
  输出：H' × W' × C_out
  感受野：空间邻域

3D卷积（视频）：
  卷积核：k_t × k_h × k_w
  输入：T × H × W × C_in
  输出：T' × H' × W' × C_out
  感受野：时空邻域
```

**C3D架构：**
```
输入：16帧 × 112×112 × 3

Conv1：3×3×3, 64 filters
  → 16×112×112×64

Pool1：1×2×2
  → 16×56×56×64

Conv2：3×3×3, 128 filters
  → 16×56×56×128

Pool2：2×2×2
  → 8×28×28×128

Conv3a, Conv3b：3×3×3, 256 filters
  → 8×28×28×256

Pool3：2×2×2
  → 4×14×14×256

Conv4a, Conv4b：3×3×3, 512 filters
  → 4×14×14×512

Pool4：2×2×2
  → 2×7×7×512

Conv5a, Conv5b：3×3×3, 512 filters
  → 2×7×7×512

Pool5：2×2×2
  → 1×4×4×512

FC：4096 → 4096 → 动作类别
```

**I3D（Inflated 3D）：**
```
核心思想：从2D预训练模型"膨胀"为3D

膨胀方法：
  2D卷积核：k_h × k_w × C_in × C_out
    ↓
  复制k_t次：
    k_t × k_h × k_w × C_in × C_out
    ↓
  除以k_t（保持输出尺度）

优势：
  - 利用ImageNet预训练
  - 收敛更快
  - 效果比随机初始化好10%+
```

**优缺点：**
```
优点：
  - 统一时空建模
  - 端到端训练
  - 不需要光流

缺点：
  - 参数量大（3D卷积核）
  - 计算复杂度高
  - 需要大量数据
```

### 方法3：循环神经网络（RNN/LSTM）

**原理：** 顺序处理，维护隐状态

**LSTM for Video：**
```
架构：
  for t = 1 to T:
    # 提取帧特征
    f_t = CNN(Frame_t)

    # LSTM更新
    h_t, c_t = LSTM(f_t, h_{t-1}, c_{t-1})

  # 最终输出
  output = MLP(h_T)
```

**ConvLSTM：**
```
改进：空间结构感知的LSTM

传统LSTM：
  输入：向量（展平的特征）
  门控：全连接层

ConvLSTM：
  输入：特征图（$H \times W \times C$）
  门控：卷积层

  $i_t = \sigma(\text{Conv}(x_t) + \text{Conv}(h_{t-1}))$ （输入门）
  
  $f_t = \sigma(\text{Conv}(x_t) + \text{Conv}(h_{t-1}))$ （遗忘门）
  
  $o_t = \sigma(\text{Conv}(x_t) + \text{Conv}(h_{t-1}))$ （输出门）
  
  $c_t = f_t \cdot c_{t-1} + i_t \cdot \tanh(\text{Conv}(x_t) + \text{Conv}(h_{t-1}))$
  
  $h_t = o_t \cdot \tanh(c_t)$

**优缺点：**
```
优点：
  - 处理可变长度序列
  - 捕捉长期依赖
  - 参数共享（效率高）

缺点：
  - 串行计算（慢）
  - 梯度消失/爆炸
  - 难以并行化
```

### 方法4：时序Attention

**Temporal Self-Attention：**

输入：
  帧特征序列：$[f_1, f_2, \ldots, f_T]$

Self-Attention（沿时间轴）：

  $Q, K, V = \text{Linear}(f)$ （每个 $f$ 投影）

  ```math
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
  ```

  含义：
    - 每一帧关注其他所有帧
    - 学习哪些帧重要

**Divided Space-Time Attention（TimeSformer）：**

问题：
  视频Patch数量：$N = T \times (H/p) \times (W/p)$
  Self-Attention复杂度：$O(N^2)$
  → 太大！

解决方案：分离时空Attention

  **Time Attention：**
    - 只在时间轴做Attention
    - 固定空间位置，看不同时间

  **Space Attention：**
    - 只在空间做Attention
    - 固定时间，看同一帧的不同位置

架构：
  ```
  for 每层：
    x = Time_Attention(x) + x
    x = Space_Attention(x) + x
    x = MLP(x) + x
```

**优缺点：**
```
优点：
  - 全局感受野
  - 并行计算
  - 灵活性高

缺点：
  - 计算复杂度O(T²)（时间轴）
  - 需要大数据量
  - 位置编码设计复杂
```

### 方法5：因果卷积（Causal Convolution）

**原理：** 只看历史，不看未来

**普通卷积 vs 因果卷积：**
```
普通卷积（3×3时间核）：
  输出t = f(帧t-1, 帧t, 帧t+1)
  → 用到未来信息

因果卷积：
  输出t = f(帧t-2, 帧t-1, 帧t)
  → 只用历史信息

实现：
  - 时间维度padding在左侧
  - 或者mask掉未来位置
```

**应用：视频生成**
```
自回归生成：
  生成帧1
    ↓
  基于帧1，生成帧2
    ↓
  基于帧1-2，生成帧3
    ↓
  ...

因果卷积确保：
  生成帧t时，只依赖帧1到t-1
```

## 时序一致性（Temporal Consistency）

### 问题

**视频生成的挑战：**
```
逐帧生成问题：
  - 帧1：猫在左边
  - 帧2：猫突然跳到右边
  - 帧3：猫消失
  - 帧4：猫又出现

结果：闪烁、跳跃、不自然
```

### 解决方案

**1. 3D卷积/Attention：**
```
思想：同时处理多帧

3D U-Net：
  不是逐帧去噪
  而是同时对多帧去噪

  输入：噪声视频片段（8帧）
  输出：去噪后的8帧
  → 帧间自动一致
```

**2. Temporal Loss：**

训练时添加时序一致性损失

```math
L_{temp} = \|f_t - f_{t+1}\| + \|\text{warp}(f_t, \text{flow}_{t \to t+1}) - f_{t+1}\|
```

第一项：
  相邻帧特征相似

第二项：
  根据光流对齐后，相邻帧像素相似

**3. Latent Video Diffusion：**
```
Stable Video Diffusion方法：

  视频 → 3D VAE Encoder → 时空潜在表示
    ↓
  在潜在空间做扩散（3D U-Net）
    ↓
  3D VAE Decoder → 视频

优势：
  - 3D VAE自动保证时序连贯性
  - 潜在空间更平滑
```

## 代表模型

### 动作识别

**TSN（Temporal Segment Networks）：**
```
思想：稀疏采样 + 融合

方法：
  1. 将视频分成K段
  2. 每段随机采样1帧
  3. 每帧过2D CNN
  4. 融合K个特征

优势：
  - 覆盖全视频（长时依赖）
  - 计算效率高（稀疏采样）
```

**SlowFast Networks：**
```
思想：双路径（慢/快）

Slow pathway：
  - 低帧率（4fps）
  - 高通道数
  - 捕捉语义（what）

Fast pathway：
  - 高帧率（32fps）
  - 低通道数
  - 捕捉运动（how）

融合：
  Lateral connections连接两路径
```

### 视频生成

**Video Diffusion Models：**
```
架构：
  - 3D U-Net（时空去噪）
  - 因果Attention（自回归生成）
  - 3D VAE（压缩）

生成：
  噪声视频 → 3D U-Net去噪 → 清晰视频
```

**Sora（推测架构）：**
```
时序建模：
  - Spacetime Patch：16×16×2（空间×时间）
  - Transformer：Joint Space-Time Attention
  - 因果Mask：生成时只看历史帧
```

## 常见问题

**Q: 为什么不直接用3D CNN处理全视频？**

A:
- 计算量爆炸（T×H×W×C太大）
- 显存不够（无法加载全视频）
- 通常采用：
  - 稀疏采样（TSN）
  - 滑动窗口（处理片段）
  - 分层处理（先局部后全局）

**Q: Transformer比3D CNN好吗？**

A:
- 大数据：Transformer更好（扩展性）
- 小数据：3D CNN更好（归纳偏置）
- 实时应用：3D CNN更快

**Q: 如何选择时序建模方法？**

A:
| 任务 | 推荐方法 | 理由 |
|------|---------|------|
| 动作识别 | 3D CNN / SlowFast | 平衡精度和速度 |
| 视频生成 | 3D U-Net / Transformer | 时序一致性 |
| 实时预测 | 因果卷积 / LSTM | 低延迟 |
| 长视频理解 | Transformer / TSN | 长期依赖 |

**Q: 如何处理超长视频（1000+帧）？**

A:
- 分段处理（Temporal Segment）
- 层级方法（先局部后全局）
- Memory机制（保存关键帧）

## 参考资料

- [Two-Stream Convolutional Networks for Action Recognition](https://arxiv.org/abs/1406.2199)
- [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767) - C3D
- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750) - I3D
- [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) - TimeSformer
- [什么是3D卷积](/guide/video/understanding/what-is-3D-convolution) - 本站相关文章
- [什么是Transformer](/guide/ai/what-is-transformer) - 本站相关文章
