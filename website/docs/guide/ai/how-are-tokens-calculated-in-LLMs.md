---
title: LLM 中的 Token 是如何计算的?
description: LLM 中的 Token 是如何计算的?

date: 20250218
plainLanguage: |
  **Token 计算说白了就是：** AI 把文字切成"单词碎片"，然后按碎片收费。

  就像你发短信，以前按字收费（1毛/字），现在 AI 按 Token 收费。但 Token 不是"字"，而是"词的碎片"——比如"苹果"可能是 1 个 Token，"iPhone"可能是 2 个 Token（i + Phone）。

  **用大白话说：**
  想象你买烤串，老板不按"串"收费，而是按"肉块"收费。一串羊肉可能有 5 块肉，一串鸡翅可能有 3 块肉。Token 就是这些"肉块"——AI 把你的文字切成小块，按小块数收费。

  **切分规则：**
  - **英文**：通常一个单词 = 1-2 个 Token（"apple" = 1 个，"application" 可能 = 2 个）
  - **中文**：通常一个字 = 1-2 个 Token（"苹" 和 "果" 可能各算 1 个）
  - **标点/空格**：也算 Token

  **怎么算费用：**
  1. AI 先把你的话切成 Token（用分词器 tokenizer）
  2. 数一下总共多少个 Token
  3. 按 Token 数收费（比如 GPT-4：输入 $0.03/1K tokens，输出 $0.06/1K tokens）

  **实用技巧：**
  - 想省钱？用短句、少用复杂词汇
  - 中文比英文贵（因为中文 Token 多）
  - 可以用在线工具（如 tiktoken）提前算一下会用多少 Token

  **与向量数据库配合：**
  - 文档分块时，按 Token 数控制每块大小（比如每块 512 Token）
  - 这样既不会太长（超模型上下文），也不会太短（浪费空间）

  说白了，Token 就是 AI 的"计费单位"——就像你打车按公里算钱，用 AI 按 Token 算钱。

podcastUrl: https://assets.listenhub.ai/listenhub-public-prod/podcast/6915872ccf351e750b7bba08.mp3
podcastTitle: AI按“文字碎片”收费：Token计费，中文成本高在哪？
---




![token-visualization](/assets/images/token-visualization.png)

在大型语言模型 (LLM) 中, Token 是文本处理的基本单元. 

如果是开源模型, 可以在模型仓库中找到 tokenizer.json 文件, 里面包含了词汇表和对应的 token 映射关系. 

其结构类似：

```json
{
  "version": "1.0",
  "added_tokens": [
    {
      "id": 151643,
      "content": "<|endoftext|>",
    },
    ...
  ],
  "model": {
    "type": "BPE",
    "vocab": {
      "!": 0,
      "\"": 1,
      "#": 2,
      "$": 3,
      "%": 4,
      "&": 5,
      ...
    }
  }
}
```

其中:
- added_tokens 表示特殊 token (如开始/结束符)
- model.type 表示分词算法, vocab 表示词汇表, key 是 token, value 是 token 的 id. 


## 常见问题

- **模型是怎样计算 token 使用量的?**  
  1. 预处理：将输入文本标准化 (如 Unicode 规范化) 
  2. 分词：使用 tokenizer 的词汇表进行子词切分
  3. 统计：计算切分后的 token 数量
  4. 特殊 token：添加系统要求的特殊 token (如开始/结束符) 
  
  注意：
  - 不同模型的分词器不同 (GPT 用 BPE, BERT 用 WordPiece 等) 
  - 空格/标点/语言都会显著影响token数量

- **如果使用大模型 API 写了一个服务, 该怎样计算 token 用量?**
  1. 首先可以尝试搜索模型的 tokenizer, 或者看看有没有已经封装好的库 (比如 OpenAI 的 tiktoken)
  2. 如果实在找不到, 可以自己测试一下, 估计一个大概的系数来计算, 比如一个汉字算作 2 个 token 等等

- **模型该怎样与向量数据库结合？**  
  1. 文档分块 (使用 token 计数控制块大小)
  2. Tokenization 处理 (保持与模型使用一致的 token)
  3. 向量化存储 (建议同时存储原始文本和token信息)
  4. 检索时通过向量相似度匹配




## Reference

- [Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/en/chapter6/5)
- [tiktoken](https://github.com/openai/tiktoken)
- [Understanding LLM Tokenization](https://christophergs.com/blog/understanding-llm-tokenization)
