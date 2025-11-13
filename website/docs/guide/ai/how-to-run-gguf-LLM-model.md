---
title: 如何本地运行 GGUF 格式的 LLM 模型?
description: 如何本地运行 GGUF 格式的 LLM 模型?

date: 20250122
plainLanguage: |
  **本地运行 GGUF 模型说白了就是：** 三步搞定——下载模型、编译工具、运行。

  就像你要在家里看电影，需要：1) 下载电影文件，2) 安装播放器，3) 双击播放。运行 GGUF 模型也是这个思路。

  **用大白话说：**
  想在家做川菜，你需要：1) 买食材（下载模型），2) 准备锅碗瓢盆（编译 llama.cpp），3) 开火做菜（运行模型）。就这么简单。

  **三步流程：**

  **第一步：下模型**
  - 去 Hugging Face 找你想要的模型（比如 DeepSeek-R1）
  - 选一个量化版本（推荐 Q4 或 Q5）
  - 下载，注意文件可能几十 GB

  **第二步：编译 llama.cpp**
  - 下载 llama.cpp 源码
  - 运行几行命令编译（就像安装软件）
  - 如果模型很新，确保用最新版 llama.cpp

  **第三步：运行**
  - 一行命令启动：`llama-server -m 模型文件路径`
  - 浏览器打开 `localhost:端口号`
  - 开始聊天！

  **注意事项：**
  - 确保电脑有足够显存/内存（Q4 的 32B 模型大概需要 20GB）
  - 新模型可能需要新版 llama.cpp 才能支持
  - 可以用 `--host 0.0.0.0` 让局域网其他设备也能访问

  说白了，运行 GGUF 模型就是"下载-编译-运行"三步曲，比想象中简单多了。

podcastUrl: https://assets.listenhub.ai/listenhub-public-prod/podcast/69158f1123647d391e2e6463.mp3
podcastTitle: GGUF模型本地运行：三步搞定，普通电脑也能玩转大模型
---





本篇内容教大家如何本地运行 GGUF 格式的 LLM 模型. 这里以最新的 DeepSeek-R1-Distill-Qwen-32B-GGUF 模型为例. 


## 模型下载

我比较喜欢 unsloth 团队的量化版本, 所以这里下载的是 unsloth 团队的量化版本. 

下载地址 [https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF/tree/main](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF/tree/main)

![](/assets/images/download-gguf-file.png)

注意这个是个合集包, 里面有 Q2-Q8 的量化版本, 选中你喜欢的量化版本, 点击这个下载按钮即可. 不需要全部下载.


## 编译 llama.cpp

由于这个模型比较新 (2025-01-21发布的), 所以需要编译最新的 llama.cpp 才能支持这个模型.

首先下载 llama.cpp 的源码, 然后编译.

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j 
```

如果你的电脑没有 cmake, 请自己搜索如何安装 cmake. 问问 LLM 是好方法.


## 模型运行

直接运行编译好的 llama.cpp 即可.

```bash
./build/bin/llama-server -m /data/unslouth/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf --host 0.0.0.0 --port 9990
```

## ok 啦~!

在浏览器访问你指定的 IP 和端口, 例如 http://192.168.1.2:9990 即可.

![](/assets/images/run-llama.cpp-server.png)



