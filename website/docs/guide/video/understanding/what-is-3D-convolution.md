---
title: 什么是 3D 卷积（3D Convolution）
description: 什么是 3D 卷积（3D Convolution）在视频理解中的应用
pageType: doc
date: 20251112
author: AI收集
plainLanguage: |
  时空一起卷：空间+时间同时感受，直接捕捉运动与变化。
  比逐帧的2D更懂“发生了什么”和“何时发生”。
---

  **2D卷积（看图片）：**
  ```
  卷积核：3×3的小窗口
  滑动：在图片上横向、纵向滑动
  看：一个位置周围9个像素

  例如：
    检测边缘：
      左边暗、右边亮 → 竖直边缘
  ```

  **3D卷积（看视频）：**
  ```
  卷积核：3×3×3的立方体窗口
  滑动：在视频上横向、纵向、时间方向滑动
  看：一个位置在3帧中的3×3区域（共27个值）

  例如：
    检测运动：
      帧1：球在左
      帧2：球在中
      帧3：球在右
      → 向右运动
  ```

  ---

  **直观对比：**

  ```
  2D卷积（图像）：
    输入：一张图片（224×224×3）
    卷积核：3×3（高×宽）
    输出：特征图（224×224×64）

    滑动方式：
      ┌─────┐
      │ 3×3 │ → 向右
      └─────┘
              ↓ 向下

  3D卷积（视频）：
    输入：一段视频（16帧×224×224×3）
    卷积核：3×3×3（帧×高×宽）
    输出：特征体（16×224×224×64）

    滑动方式：
        ╔═══╗
       ╔╝3×3╚╗ → 向右
       ║ ×3  ║
       ╚═════╝
         ↓ 向下
         ↓ 时间方向
  ```

  ---

  **3D卷积如何工作：**

  ```
  输入视频：
    帧1：[猫站着的图像]
    帧2：[猫弯腿的图像]
    帧3：[猫跳起的图像]

  3D卷积核（3×3×3）：
    看3帧×3×3区域 = 27个值

  计算（以某个位置为例）：
    帧1的3×3区域：
      [1, 2, 1]
      [0, 5, 3]
      [2, 1, 0]

    帧2的3×3区域：
      [1, 3, 2]
      [1, 6, 4]
      [2, 2, 1]

    帧3的3×3区域：
      [2, 4, 3]
      [2, 7, 5]
      [3, 2, 1]

  卷积核权重（假设）：
    W[帧1] = [...], W[帧2] = [...], W[帧3] = [...]

  输出（单个值）：
    sum(输入 * 权重) = 数字（表示某种时空特征）

  含义（学到的模式）：
    - 低层：简单运动（向上、向下、向左、向右）
    - 高层：复杂动作（跳跃、挥手、转身）
  ```

  ---

  **3D卷积的优势：**

  **1. 自动学习时空特征**
  ```
  不需要手工设计运动特征
  网络自动学到：
    - 边缘随时间的移动
    - 物体的变形
    - 场景的变化
  ```

  **2. 端到端训练**
  ```
  不需要光流等预处理：
    原始视频 → 3D CNN → 动作识别
    （一步到位）

  vs 传统方法：
    原始视频 → 计算光流 → 2D CNN → 动作识别
    （两步，光流计算慢）
  ```

  **3. 统一时空建模**
  ```
  2D CNN + 时序建模（如LSTM）：
    先空间、后时间（分离）

  3D CNN：
    时空联合建模（统一）
  ```

  ---

  **3D卷积的挑战：**

  **1. 参数量大**
  ```
  2D卷积核：
    3×3 = 9个参数

  3D卷积核：
    3×3×3 = 27个参数（3倍）

  一层3D卷积（64个卷积核）：
    27 × 64 × C_in = 1728·C_in个参数

  → 容易过拟合
  ```

  **2. 计算量大**
  ```
  2D卷积：
    输入H×W，卷积核k×k
    计算量：O(H·W·k²)

  3D卷积：
    输入T×H×W，卷积核t×k×k
    计算量：O(T·H·W·t·k²)

  → 训练慢，推理慢
  ```

  **3. 需要大量数据**
  ```
  参数多 → 需要更多数据防止过拟合

  ImageNet：100万图像
  Kinetics-400：24万视频（但每个视频10秒，相当于更多数据）
  ```

  ---

  **经典3D CNN模型：**

  **1. C3D（3D卷积网络）**
  ```
  架构：
    输入：16帧 × 112×112×3

    Conv1：3×3×3, 64 filters
    Pool1：1×2×2（时间不压缩）

    Conv2：3×3×3, 128 filters
    Pool2：2×2×2（开始压缩时间）

    Conv3a, 3b：3×3×3, 256 filters
    Pool3：2×2×2

    Conv4a, 4b：3×3×3, 512 filters
    Pool4：2×2×2

    Conv5a, 5b：3×3×3, 512 filters
    Pool5：2×2×2

    FC6, FC7, FC8

  特点：
    - 所有卷积核：3×3×3（统一）
    - 早期保留时间维度（Pool1不压时间）
    - 后期压缩时间（逐渐减少帧数）
  ```

  **2. I3D（Inflated 3D）**
  ```
  核心思想："膨胀"2D卷积为3D

  步骤：
    1. 从ImageNet预训练的Inception模型开始
    2. "膨胀"所有2D卷积为3D：
       - 2D：3×3 → 3D：3×3×3
       - 权重重复t次，除以t

    3. 在视频数据上微调

  优势：
    - 利用ImageNet预训练（大规模图像数据）
    - 收敛更快
    - 比随机初始化的3D CNN好10%+

  膨胀示例：
    2D卷积核：
      W[h, w, c_in, c_out]

    膨胀为3D：
      for t in range(temporal_kernel_size):
        W_3d[t, h, w, c_in, c_out] = W[h, w, c_in, c_out] / temporal_kernel_size

    意思：时间维度每个位置的权重相同（初始化）
  ```

  **3. R(2+1)D（分解3D卷积）**
  ```
  核心思想：3D卷积 = 2D空间卷积 + 1D时间卷积

  传统3D卷积（t×k×k）：
    一次性时空建模
    参数量：t·k²·C_in·C_out

  R(2+1)D：
    步骤1：2D空间卷积（1×k×k）
      只在空间建模
      参数量：k²·C_in·M

    步骤2：1D时间卷积（t×1×1）
      只在时间建模
      参数量：t·M·C_out

  好处：
    - 参数量减少
    - 增加非线性（两次激活）
    - 效果更好
  ```

  ---

  **3D卷积 vs 其他方法：**

  | 方法 | 优势 | 劣势 | 代表模型 |
  |------|------|------|---------|
  | **2D CNN + LSTM** | 参数少，灵活 | 串行慢，长期依赖弱 | LRCN |
  | **Two-Stream** | 利用光流 | 需要预计算光流 | TSN |
  | **3D CNN** | 端到端，时空统一 | 参数多，计算量大 | C3D, I3D |
  | **Transformer** | 全局感受野 | 需要大数据 | TimeSformer |

  ---

  **实际应用：**

  **1. 动作识别**
  ```
  任务：识别视频中的动作（跑、跳、挥手）

  方法：
    视频 → 3D CNN → 动作类别

  数据集：
    - UCF-101：101类动作
    - Kinetics-400：400类动作
    - Something-Something：174类物体交互
  ```

  **2. 视频分类**
  ```
  任务：识别视频类别（体育、新闻、娱乐）

  方法：
    视频 → 3D CNN → 类别概率
  ```

  **3. 时空检测**
  ```
  任务：检测视频中的人和动作（时空边界框）

  方法：
    3D CNN + 检测头 → (x, y, t, w, h, duration, class)
  ```

  **4. 视频生成**
  ```
  任务：生成连贯视频

  方法：
    3D U-Net：
      噪声视频 → 3D卷积去噪 → 清晰视频
  ```

  ---

  **实现细节：**

  **PyTorch伪代码：**
  ```python
  import torch.nn as nn

  class Conv3DBlock(nn.Module):
      def __init__(self, in_channels, out_channels):
          super().__init__()
          self.conv = nn.Conv3d(
              in_channels,
              out_channels,
              kernel_size=(3, 3, 3),  # (时间, 高, 宽)
              padding=(1, 1, 1)       # 保持尺寸
          )
          self.bn = nn.BatchNorm3d(out_channels)
          self.relu = nn.ReLU()

      def forward(self, x):
          # x: (batch, channels, time, height, width)
          x = self.conv(x)
          x = self.bn(x)
          x = self.relu(x)
          return x

  # 使用
  block = Conv3DBlock(in_channels=3, out_channels=64)
  video = torch.randn(1, 3, 16, 112, 112)  # 1个视频，3通道，16帧，112×112
  output = block(video)  # (1, 64, 16, 112, 112)
  ```

  ---

  **常见误区：**

  **误区1："3D卷积一定比2D卷积好"**
  - 不一定！
    - 小数据集：2D+预训练更好
    - 大数据集：3D CNN更好
    - 看具体任务和资源

  **误区2："3D卷积核越大越好"**
  - 不对！
    - 过大：参数爆炸，过拟合
    - 过小：感受野不够
    - 常用：3×3×3（平衡）

  **误区3："3D CNN已经过时了"**
  - 不对！
    - Transformer需要大数据
    - 3D CNN在中小数据上仍有优势
    - 很多实际应用还在用3D CNN

  说白了，3D卷积就是"时空联合建模"——通过在时间和空间同时滑动卷积核，自动学习视频中的运动模式和时空特征。虽然参数多、计算量大，但端到端训练、无需预处理，是视频理解的经典方法。
---


<!-- TODO: 添加3D卷积示意图，展示2D vs 3D卷积核的区别 -->
![3d-convolution-kernel](/assets/images/video/3d-convolution-kernel.svg)

3D 卷积：空间加时间一起卷，捕捉动作的来龙去脉。

## 为什么它重要（≤5min）
- 视频不是多张图：时间维必须被建模。

## 怎么做（核心要点）
- 卷积核：`3×3×3` 同时看三帧的空间邻域。
- 特征层：低层看“动没动”，高层看“动了什么”。

![3d-convolution-kernel](/assets/images/video/3d-convolution-kernel.svg)

## 优缺点（一句到位）
- 优点：端到端学时空，不依赖光流。
- 缺点：参数/算量都涨，数据要跟上。

## 易错点（别踩坑）
- 大核崇拜：`3×3×3` 通常是甜点位。
- 3D一定赢？小数据或长序列未必。

## 适用场景（马上用得上）
- 动作识别/视频分类/时空检测/视频生成基座。

## 参考图源（供重绘或嵌入）
- C3D/I3D 架构图（可抽取卷积与池化模块重绘SVG）

## 一页总结
- 时空合一更聪明；算力预算要清楚；任务适配再选择。

## 核心概念

### 2D卷积 vs 3D卷积

**2D卷积（图像）：**
```
输入：H × W × C_in
卷积核：k_h × k_w × C_in × C_out
输出：H' × W' × C_out

操作：
  在空间维度（高、宽）滑动卷积核
  每个位置计算：
    output[i, j, c] = Σ input[i+Δi, j+Δj, :] * kernel[Δi, Δj, :, c]

特点：
  - 只处理空间信息
  - 适合图像
```

**3D卷积（视频）：**
```
输入：T × H × W × C_in
卷积核：k_t × k_h × k_w × C_in × C_out
输出：T' × H' × W' × C_out

操作：
  在时空维度（时间、高、宽）滑动卷积核
  每个位置计算：
    output[t, i, j, c] = Σ input[t+Δt, i+Δi, j+Δj, :] * kernel[Δt, Δi, Δj, :, c]

特点：
  - 同时处理时空信息
  - 适合视频
```

### 时空感受野

**2D卷积的局限：**
```
逐帧处理视频：
  Frame1 → 2D CNN → Feature1
  Frame2 → 2D CNN → Feature2
  Frame3 → 2D CNN → Feature3
  ...

问题：
  - 每帧独立处理
  - 无法捕捉帧间运动
  - 需要额外的时序建模（LSTM、Attention）
```

**3D卷积的优势：**
```
联合处理视频片段：
  [Frame1, Frame2, Frame3] → 3D CNN → Feature

优势：
  - 卷积核同时看多帧
  - 自动捕捉运动模式
  - 端到端学习时空特征
```

## 3D卷积的数学原理

### 前向传播

**输入：**
```math
X \in \mathbb{R}^{T \times H \times W \times C_{in}}
```

例如：
  16帧 × 112×112 × 3通道

**卷积核：**
```math
W \in \mathbb{R}^{k_t \times k_h \times k_w \times C_{in} \times C_{out}}
```

例如：
 3×3×3卷积核，64个filter
```math
W \in \mathbb{R}^{3 \times 3 \times 3 \times 3 \times 64}
```

**输出：**
```math
Y[t, i, j, c_out] = \sum_{\Delta t, \Delta i, \Delta j, c_{in}} X[t+\Delta t, i+\Delta i, j+\Delta j, c_{in}] \cdot W[\Delta t, \Delta i, \Delta j, c_{in}, c_{out}] + b[c_{out}]
```

其中：
- Δt ∈ [0, k_t-1]
- Δi ∈ [0, k_h-1]
- Δj ∈ [0, k_w-1]
- c_in ∈ [0, C_in-1]

### 参数量对比

**2D卷积：**
```math
\text{参数量} = k_h \times k_w \times C_{in} \times C_{out}
```

例如（3×3卷积，3→64通道）：
  3 × 3 × 3 × 64 = 1,728

**3D卷积：**
```math
\text{参数量} = k_t \times k_h \times k_w \times C_{in} \times C_{out}
```

例如（3×3×3卷积，3→64通道）：
  3 × 3 × 3 × 3 × 64 = 5,184

增加：3倍（时间维度引入）

### 计算复杂度

**2D卷积：**
```math
\text{FLOPs} \approx H \times W \times k_h \times k_w \times C_{in} \times C_{out}
```

对于H=W=112, k_h=k_w=3, C_in=3, C_out=64：
  FLOPs ≈ 112 × 112 × 9 × 3 × 64 ≈ 21.8M

**3D卷积：**
```math
\text{FLOPs} \approx T \times H \times W \times k_t \times k_h \times k_w \times C_{in} \times C_{out}
```

对于T=16, H=W=112, k_t=k_h=k_w=3, C_in=3, C_out=64：
  FLOPs ≈ 16 × 112 × 112 × 27 × 3 × 64 ≈ 349M

增加：16倍（时间维度引入）

## 经典3D CNN架构

### C3D

**架构设计：**
```
设计原则：
  - 所有3D卷积核：3×3×3（统一）
  - 所有池化核：2×2×2（除了Pool1）
  - Pool1：1×2×2（保留时间维度）

网络结构：
  输入：16帧 × 112×112 × 3

  Conv1a：3×3×3, 64 filters, stride=1, pad=1
    → 16×112×112×64
  Pool1：1×2×2, stride=1×2×2
    → 16×56×56×64

  Conv2a：3×3×3, 128 filters
    → 16×56×56×128
  Pool2：2×2×2
    → 8×28×28×128

  Conv3a：3×3×3, 256 filters
  Conv3b：3×3×3, 256 filters
    → 8×28×28×256
  Pool3：2×2×2
    → 4×14×14×256

  Conv4a：3×3×3, 512 filters
  Conv4b：3×3×3, 512 filters
    → 4×14×14×512
  Pool4：2×2×2
    → 2×7×7×512

  Conv5a：3×3×3, 512 filters
  Conv5b：3×3×3, 512 filters
    → 2×7×7×512
  Pool5：2×2×2
    → 1×4×4×512

  Flatten + FC6：4096
  FC7：4096
  FC8：K classes
```

**训练细节：**
```
数据集：Sports-1M（100万视频）
优化器：SGD
学习率：0.003（每4个epoch除以2）
Batch size：30个视频片段
Clip长度：16帧（每2帧采样一次，覆盖32帧）
```

### I3D（Inflated 3D ConvNets）

**核心思想：** 从2D预训练模型"膨胀"为3D

**膨胀方法：**
```math
W_{2d} \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}
```

膨胀为3D：
```math
W_{3d}[t, h, w, c_{in}, c_{out}] = \frac{W_{2d}[h, w, c_{in}, c_{out}]}{N}
```

其中：
- N = k_t（时间维度大小）
- 每个时间步的权重相同
- 除以N保持输出尺度

例子：
  2D: 3×3 → 3D: 3×3×3
  权重：W_2d[i, j, :, :] 复制3次

**两个流（Two Streams）：**
```
RGB流：
  输入：原始RGB视频
  处理：I3D网络
  输出：空间特征

光流流：
  输入：光流图（预计算）
  处理：I3D网络
  输出：运动特征

融合：
  final = Average(RGB_logits, Flow_logits)
```

**优势：**
```
1. ImageNet预训练：
   - 利用大规模图像数据（130万）
   - 初始化更好

2. 收敛更快：
   - 比随机初始化快约2倍

3. 效果更好：
   - Kinetics-400：74.2%（比C3D高约18%）
```

### R(2+1)D（分解3D卷积）

**动机：** 减少参数，增加非线性

**分解方法：**
```
原始3D卷积（t×k×k）：
  一次操作，参数量：t·k²·C_in·C_out

R(2+1)D分解：
  步骤1：2D空间卷积（1×k×k）
    参数量：k²·C_in·M

  步骤2：1D时间卷积（t×1×1）
    参数量：t·M·C_out

  总参数：k²·C_in·M + t·M·C_out

选择M使得总参数≈原3D卷积：
  M = (t·k²·C_in·C_out) / (k²·C_in + t·C_out)
```

**优势：**
```
1. 参数减少：
   约等同参数量（通过调M）

2. 非线性增加：
   两次ReLU激活（2D后+1D后）
   vs 一次（3D后）

3. 效果提升：
   Kinetics：75.4%（比I3D高1.2%）
```

**架构示例：**
```
R(2+1)D Block：
  输入：T×H×W×C_in

  2D空间卷积：
    Conv2D(1×3×3, C_in → M)
    BatchNorm + ReLU
    → T×H×W×M

  1D时间卷积：
    Conv1D(3×1×1, M → C_out)
    BatchNorm + ReLU
    → T×H×W×C_out
```

## 实现细节

### PyTorch实现

**基础3D卷积：**
```python
import torch
import torch.nn as nn

class BasicConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 使用
conv = BasicConv3D(3, 64, kernel_size=3, padding=1)
video = torch.randn(2, 3, 16, 112, 112)  # (batch, C, T, H, W)
output = conv(video)  # (2, 64, 16, 112, 112)
```

**R(2+1)D Block：**
```python
class R2Plus1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        # 2D空间卷积
        self.conv_2d = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, 3, 3),  # (T=1, H=3, W=3)
            stride=1,
            padding=(0, 1, 1)
        )
        self.bn_2d = nn.BatchNorm3d(mid_channels)
        self.relu_2d = nn.ReLU(inplace=True)

        # 1D时间卷积
        self.conv_1d = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(3, 1, 1),  # (T=3, H=1, W=1)
            stride=1,
            padding=(1, 0, 0)
        )
        self.bn_1d = nn.BatchNorm3d(out_channels)
        self.relu_1d = nn.ReLU(inplace=True)

    def forward(self, x):
        # 2D空间
        x = self.conv_2d(x)
        x = self.bn_2d(x)
        x = self.relu_2d(x)

        # 1D时间
        x = self.conv_1d(x)
        x = self.bn_1d(x)
        x = self.relu_1d(x)

        return x
```

### 训练技巧

**数据增强：**
```python
from torchvision import transforms

video_transform = transforms.Compose([
    # 空间增强
    transforms.RandomResizedCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),

    # 时间增强
    RandomTemporalCrop(16),  # 随机采样16帧
    TemporalRandomCrop(scale=(0.8, 1.0)),  # 时间尺度变化

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

**采样策略：**
```
密集采样（训练）：
  视频长度：T帧
  采样：连续16帧
  步长：1-2帧
  覆盖：整个视频

稀疏采样（推理）：
  视频长度：T帧
  采样：均匀采样10个片段，每个16帧
  融合：平均10个片段的预测
```

## 性能对比

### Kinetics-400数据集

| 模型 | Top-1准确率 | 参数量 | FLOPs |
|------|-----------|--------|-------|
| C3D | 56.1% | 79M | 39G |
| I3D (RGB) | 71.1% | 12M | 53G |
| I3D (Two-Stream) | 74.2% | 25M | 107G |
| R(2+1)D | 75.4% | 64M | 153G |
| SlowFast 8×8 | 79.8% | 34M | 65G |

### 计算效率

**推理速度（V100 GPU）：**
```
C3D：
  - 输入：16帧×112×112
  - 速度：~1200 clips/s
  - 内存：~2GB

I3D：
  - 输入：64帧×224×224
  - 速度：~150 clips/s
  - 内存：~8GB

R(2+1)D：
  - 输入：16帧×112×112
  - 速度：~800 clips/s
  - 内存：~4GB
```

## 常见问题

**Q: 为什么C3D的Pool1是1×2×2而不是2×2×2？**

A:
- 早期保留时间分辨率
- 让网络在早期学习细粒度时间特征
- 后期再压缩时间（Pool2-5都是2×2×2）

**Q: 3D卷积vs (2D+LSTM)哪个好？**

A:
| 维度 | 3D卷积 | 2D+LSTM |
|------|--------|---------|
| 参数量 | 大 | 中 |
| 并行性 | 好（卷积并行） | 差（LSTM串行） |
| 长期依赖 | 弱（感受野有限） | 强 |
| 预训练 | 需要视频数据 | 可用ImageNet |
| 适用场景 | 短期运动 | 长序列 |

**Q: 如何选择3D卷积核大小？**

A:
- **时间维度（k_t）：**
  - 3：标准（C3D、I3D）
  - 1：退化为2D（某些层）
  - 5+：罕见（参数爆炸）

- **空间维度（k_h, k_w）：**
  - 3×3：标准
  - 1×1：降维
  - 7×7：早期层（如I3D的第一层）

**Q: 3D CNN能处理多长的视频？**

A:
- 通常：16-64帧
- 更长：
  - 分段处理
  - 稀疏采样
  - 滑动窗口
- 内存限制：长视频需要更多显存

## 参考资料

- [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767) - C3D论文
- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750) - I3D论文
- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248) - R(2+1)D论文
- [什么是时序建模](/guide/video/understanding/what-is-temporal-modeling) - 本站相关文章
