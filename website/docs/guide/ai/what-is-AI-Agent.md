---
title: 什么是 AI Agent
description: 什么是 AI Agent

date: 20250220
plainLanguage: |
  **AI Agent 说白了就是：** 给 AI 配了"手脚"，让它不仅能说，还能干。

  就像你雇了个助理，他不仅能听懂你的话（大语言模型），还能帮你查资料（工具调用）、记住之前的事（记忆）、看图片听声音（多模态）。AI Agent 就是这样一个"全能助理"。

  **用大白话说：**
  最简单的 AI Agent 就像你手机里的"翻译助手"——你点一下，输入中文，它自动翻译成英文。复杂一点的，就像你让助理"帮我写篇文章"，他会自己去查资料、整理思路、写出来，全程不用你管。

  **核心组成：**
  1. **大脑（LLM）**：理解你的意图，做决策
  2. **手脚（工具调用）**：能操作其他系统，比如搜索、写代码、发邮件
  3. **记忆**：记住之前的对话和结果
  4. **感知**：能看图片、听声音、读文档

  **简单 vs 复杂：**
  - **简单版**：一个 prompt + 触发器，比如"翻译这段文字"
  - **复杂版**：能自动规划任务、调用多个工具、处理多步骤流程

  说白了，AI Agent 就是让 AI 从"聊天机器人"变成"能干活的小助手"。它不仅能回答你的问题，还能帮你完成任务。

podcastUrl: https://assets.listenhub.ai/listenhub-public-prod/podcast/6915911b23647d391e2e6f3e.mp3
podcastTitle: AI Agent：给AI配上“手脚”，变身全能小助理
---




![ai-agent-architecture](/assets/images/ai-agent.jpg)

(image from generativeai.pub)

AI Agent（人工智能代理, 可不是AI特工哦）目前的定义已经很混乱了。但就实际使用来讲 AI Agent 是旨在增强大语言模型，最终达到可以自动完成任务的系统。

AI Agent 一般由大语言模型 (充当大脑), 调度/编排系统 (充当触发器和任务决策), 工具调用 (充当手脚), 记忆与学习 (充当经验), 多模态感知 (充当眼睛和耳朵) 等组成整。

AI Agent 可以是极其简单的 prompt + LLM + 触发器组成, 比如一个中转英的翻译 Agent:

```
请帮我把下面的文本翻译为英文: {text}
```

（text 是用户输入的文本, 会由触发器自动拼进去）

这样一个简单的 prompt 就是一个AI Agent, 用户只要点击这个翻译 Agent 的图标, 进去后在输入框输入文本后点击翻译即可，整个过程不需要任何人工干预。

AI Agent 也可以很复杂, 下图就是 ComfyUI 中将给定人物照片生成另一张人物照片的姿势的 AI Agent:

![](/assets/images/create-consistent-characters-within-comfyui-comfyui-demo-1129.webp)

(图片来自 www.runcomfy.com)


下面则是 refly.ai 中一个给定命题, 自动搜索并生成文章的 AI Agent 的示例:

![](/assets/images/generate-article.webp)

(图片来自 refly.ai)




