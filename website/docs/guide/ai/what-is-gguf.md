---
title: 什么是 GGUF
description: 什么是 GGUF

date: 20250113
plainLanguage: |
  **GGUF 说白了就是：** 大模型的"压缩包"格式，但比 ZIP 高级多了。

  想象一下，你要把一个几百 GB 的游戏传给朋友。如果用普通压缩，可能压到 50GB，但解压还得等半天。GGUF 就像是一个"智能压缩包"——不仅体积小，还能直接"边解压边玩"，不用等全部解压完。

  **用菜市场大妈的话说：**
  以前存大模型，就像把一仓库的货分成几千个小箱子，每次用都要把所有箱子搬出来。GGUF 就是把所有货打包成一个超级大箱子，但神奇的是，你只需要打开一个小口，就能拿到想要的任何东西，还不用把整个箱子都打开。

  **核心就三点：**
  1. **省地方**：同样的模型，GGUF 格式能省不少硬盘空间
  2. **开得快**：就像手机秒开 App，不用等加载
  3. **一个文件搞定**：不用配环境、不用装依赖，一个文件就能跑

  说白了，GGUF 就是让"普通人也能在自家电脑上跑大模型"这件事变得可能。以前你得是技术大牛才能搞定，现在就像双击打开 Word 文档一样简单。
---




![gguf-file-structure](/assets/images/gguf-file-structure.png)

[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)（GGML Universal File）是一种专为大型语言模型（LLM）设计的文件格式。它旨在解决大型模型在实际应用中遇到的存储效率、加载速度、兼容性和扩展性等问题，从而简化 LLM 的使用和部署。

## GGUF 的主要特点和优势

*   **高效存储：** GGUF 格式优化了数据的存储方式，减少了存储空间的占用，这对于包含大量参数的大型模型尤为重要。它采用紧凑的二进制编码格式和优化的数据结构来高效地保存模型参数（权重和偏差）。
*   **单文件部署：** 它们可以轻松分发和加载，加载模型所需的所有信息都包含在模型文件中，不需要任何外部文件来获取附加信息。
*   **快速加载：** GGUF 格式支持快速加载模型数据（使用 ```mmap```），这对于需要即时响应的应用场景非常有用，例如在线聊天机器人或实时翻译系统。
*   **跨平台兼容性：** GGUF 兼容多种编程语言，例如 Python 和 R，非常方便在不同平台和环境中使用。大部分语言都可以使用少量代码轻松加载和保存模型，无需外部库。
*   **支持微调：** GGUF 支持微调，允许用户根据特定的应用场景调整 LLM，并存储跨应用部署模型的提示模板。
*   **取代 GGML：** GGUF 是 GGML 的替代者。GGML 由于在灵活性和扩展性方面存在一些限制，已被弃用，由 GGUF 取代。

## GGUF 的应用

GGUF 格式的模型文件可以用于各种应用场景，例如：

*   **本地部署 LLM：** GGUF 格式使得在消费级计算机硬件（包括 CPU 和 GPU）上运行 LLM 成为可能。
*   **移动设备上的 LLM 推理：** 由于其高效的存储和加载特性，GGUF 也适用于在移动设备上进行 LLM 推理。
*   **快速原型开发：** GGUF 使得开发者可以更快速地加载和测试不同的 LLM 模型。

总而言之，GGUF 是一种重要的 LLM 文件格式，它通过提高存储效率、加载速度和兼容性，简化了 LLM 的使用和部署，并有望成为未来大模型文件标准格式之一。

## 那些框架支持 GGUF

- [ggml](https://github.com/ggerganov/ggml)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [InfniLM](https://github.com/InfiniTensor/InfiniLM)
- [crabml](https://github.com/crabml/crabml)

## Reference

- https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
