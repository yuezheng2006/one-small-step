---
title: 什么是 Flash Attention?
description: 什么是 Flash Attention?

date: 20250201
plainLanguage: |
  **Flash Attention 说白了就是：** 让 AI "思考"更快更省内存的优化技巧。

  想象你的大脑有两个"记忆区"：一个是慢速但容量大的"仓库"（HBM），一个是快速但容量小的"工作台"（SRAM）。传统方法就像把所有东西都搬到工作台上，结果工作台放不下，只能来回搬，累死。

  **用菜市场的话说：**
  就像你去买菜，传统方法是把所有菜都搬到收银台，结果台子放不下，只能搬一部分、算一部分、再搬一部分... Flash Attention 就是"分批拿菜"，每次只拿需要的几样，算完再拿下一批，不用把所有菜都堆在收银台。

  **核心技巧：**
  1. **分块处理**：把大任务拆成小块，一块一块处理
  2. **不存中间结果**：算完就扔，需要时再算一遍（虽然多算一次，但省内存）
  3. **优化搬运**：减少在"仓库"和"工作台"之间搬东西的次数

  **效果：**
  - 速度快 3-4 倍：就像从"走路"变成"坐高铁"
  - 内存省 4-20 倍：以前需要 100GB 显存，现在可能只要 20GB
  - 能处理更长的文本：以前最多 2048 个词，现在能到 8192 甚至更长

  说白了，Flash Attention 就是"用聪明的方法偷懒"——不是真的偷懒，而是让 GPU 更高效地工作，就像你整理房间时，不是把所有东西都摊开，而是分类分批处理。
---




![flashattn_banner](/assets/images/flashattn_banner.jpg)

Flash Attention 是一种优化的注意力机制, 旨在提高深度学习模型中注意力计算的效率. 它通优化访存机制来加速训练和推理过程. 

目前的GPU架构中, HBM 容量大但处理速度慢, SRAM 虽然容量小但操作速度快. 

标准的注意力机制使用 HBM 来存储、读取和写入注意力分数矩阵（attention score matrix, 矩阵存储 Q/K/V). 具体步骤为将这些从 HBM 加载到 GPU 的片上 SRAM, 然后执行注意力机制的单个步骤, 然后写回 HBM, 并重复此过程. 

而 Flash Attention 则是采用分块计算（Tiling）技术，将大型注意力矩阵划分为多个块（tile），在 SRAM 中逐块执行计算。通过：
- **分块策略**：将 Q/K/V 矩阵分块后流水线处理，避免存储完整的中间矩阵
- **重计算（Recomputation）**：在反向传播时动态重新计算前向结果，而非存储中间值
- **IO优化**：通过精确的内存访问控制，使数据在 HBM 和 SRAM 间的移动最小化

### 优点

- **计算效率高**：通过分块并行计算和半精度（FP16/BF16）优化，充分利用 GPU Tensor Cores
- **内存使用减少**：重计算技术减少 4-20 倍内存占用，支持更长序列训练
- **训练加速**：反向传播通过延迟重计算优化，实现端到端 2-4 倍加速
- **精度保持**：采用平铺分块策略时仍保持数值稳定性，支持混合精度训练

### 缺点

- **实现复杂**：由于需要对底层计算进行优化, Flash Attention 的实现可能比传统注意力机制更复杂. 
- **硬件依赖**：在某些情况下, 可能需要特定的硬件支持才能充分发挥其性能优势. 
- **调试困难**：优化后的计算过程可能导致调试和故障排查变得更加困难. 

总的来说, Flash Attention 是一种强大的工具, 能够在不牺牲性能的情况下提高模型的效率, 但在实现和使用时需要考虑其复杂性和硬件要求. 

## 性能

![flash2_h100_fwd_bwd_benchmark](/assets/images/flash2_h100_fwd_bwd_benchmark.png)

当使用 H100 显卡且序列长度是512时（数据来自论文测试），PyTorch 的标准处理速度是 62 Tflops，而 Flash Attention 则可以达到 157 Tflops，Flash Attention 2 则可以达到215 Tflops。在 FP16/BF16 精度下，实际加速比可达标准实现的 3-4 倍。

## Refs

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [flash-attention on github](https://github.com/Dao-AILab/flash-attention)
