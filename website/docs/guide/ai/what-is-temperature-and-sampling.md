---
title: 什么是 Temperature 和采样策略
description: 什么是 Temperature（温度）和采样策略（Sampling Strategies）
date: 20250112
author: AI收集
plainLanguage: |
  **Temperature 和采样策略说白了就是：** 控制 AI 回答时"有多随机"——是严谨还是天马行空。

  就像掷骰子：
  - **Temperature = 0**：不掷了，直接选最大的那个（确定性，每次一样）
  - **Temperature = 0.3**：偏爱大数字，但偶尔也选小的（稍微随机）
  - **Temperature = 1.0**：公平掷骰子（标准随机）
  - **Temperature = 2.0**：所有数字机会差不多（非常随机，可能乱说）

  **用大白话说：**

  想象你问 AI："推荐一部科幻电影"

  **Temperature = 0（零度，超严谨）：**
  - AI 每次都回答："《星际穿越》"（最有把握的答案）
  - 优点：稳定、准确
  - 缺点：无聊、没新意

  **Temperature = 0.7（常温，平衡）：**
  - AI 可能回答："《星际穿越》"、"《盗梦空间》"、"《流浪地球》"
  - 优点：既靠谱又有点惊喜
  - 缺点：偶尔会推荐冷门片

  **Temperature = 1.5（高温，很随机）：**
  - AI 可能回答："《大话西游》"（不太科幻）、"《泰坦尼克号》"（完全不搭）
  - 优点：创意十足
  - 缺点：经常乱来

  **Temperature = 2.0（爆表，胡说八道）：**
  - AI 可能回答："《白雪公主与七个小矮人》"（跟科幻八竿子打不着）
  - 基本不可用

podcastUrl: https://assets.listenhub.ai/listenhub-public-prod/podcast/69159efc23647d391e2ebdba.mp3
podcastTitle: AI生成文本：采样策略如何玩转“掷骰子”，掌控创造与精准
---

  **采样策略（怎么"掷骰子"）：**

  除了 Temperature，还有其他"掷骰子"的方法：

  **1. Top-k（只看前 k 个）**
  - 只从"前 k 个最有可能的词"里选
  - 例如 k=5：只从前 5 个候选词里选，其他全忽略
  - 好处：避免选到太离谱的词
  - 坏处：可能漏掉好答案

  **2. Top-p（核采样，只看概率和 ≥ p 的）**
  - 从"概率总和达到 p"的最小词集里选
  - 例如 p=0.9：选最有可能的词，直到累计概率 ≥ 90%
  - 好处：动态调整候选词数量（有时 3 个词就够 90%，有时需要 20 个）
  - 坏处：计算稍复杂

  **3. 贪心采样（Greedy，永远选最好的）**
  - 每次都选概率最高的词
  - = Temperature 0
  - 好处：确定性强
  - 坏处：无趣

  ---

  **实际应用场景：**

  **需要准确答案（Temperature ≈ 0）：**
  - 数学题："1+1=？"
  - 代码生成："写个快速排序"
  - 翻译："Translate: Hello"
  - 事实查询："秦始皇哪年统一六国？"

  **需要平衡创意（Temperature ≈ 0.7）：**
  - 聊天对话
  - 写邮件
  - 文章总结
  - 常见的对话助手（ChatGPT 默认）

  **需要创意爆棚（Temperature ≈ 1.0-1.2）：**
  - 写小说
  - 头脑风暴
  - 创意文案
  - 写诗、写歌词

  **千万别用高温（Temperature > 1.5）：**
  - 基本只会胡说八道
  - 除非你故意要搞笑

  ---

  **常见参数组合：**

  **保守型（适合严肃场景）：**
  ```
  Temperature = 0.3
  Top-p = 0.9
  → 稳定但不死板
  ```

  **平衡型（默认推荐）：**
  ```
  Temperature = 0.7
  Top-p = 0.95
  → ChatGPT 默认设置
  ```

  **创意型（写作、脑暴）：**
  ```
  Temperature = 1.0
  Top-p = 1.0
  → 放飞自我
  ```

  **极度保守（代码、数学）：**
  ```
  Temperature = 0
  Top-k = 1
  → 完全确定性
  ```

  ---

  **打个比方：**

  Temperature 就像"喝酒的度数"：
  - **0 度**：完全清醒，一板一眼
  - **0.7 度**：微醺，有点放松但还理智
  - **1.5 度**：喝高了，开始说胡话
  - **2.0 度**：喝多了，完全不知道自己在说啥

  说白了，Temperature 和采样策略就是控制 AI 的"随机性"——严肃场景调低（0-0.3），创意场景调高（0.7-1.2），千万别爆表（> 1.5）。
---

![temperature-sampling](/assets/images/temperature-sampling.png)

Temperature 和采样策略是控制大型语言模型（LLM）生成文本**随机性和创造性**的核心参数。

理解这些参数对于优化 AI 输出质量至关重要。

## 核心概念：模型如何生成文本

### 生成流程

LLM 生成文本是一个**逐词预测**的过程：

```
输入："今天天气真"
模型输出概率分布：
  好: 60%
  热: 20%
  冷: 15%
  糟: 3%
  棒: 2%
```

模型需要从这个概率分布中**选一个词**，这个选择过程就是**采样（Sampling）**。

### 确定性 vs 随机性

**确定性输出（Deterministic）：**
- 每次都选概率最高的词
- 相同输入 → 相同输出
- 适合：数学题、代码、翻译

**随机性输出（Stochastic）：**
- 按概率随机选词
- 相同输入 → 不同输出
- 适合：创意写作、对话、头脑风暴

## Temperature（温度参数）

### 定义

Temperature 是一个**缩放因子**，用于调整概率分布的"陡峭程度"。

### 数学原理

在生成下一个 token 时，模型输出的原始分数（logits）通过 **Softmax + Temperature** 转换为概率：

```python
# 原始 logits
logits = [3.0, 2.0, 1.0, 0.5]  # 四个候选词的分数

# 应用 Temperature
def softmax_with_temperature(logits, temperature):
    scaled_logits = [l / temperature for l in logits]
    exp_values = [exp(l) for l in scaled_logits]
    sum_exp = sum(exp_values)
    probabilities = [e / sum_exp for e in exp_values]
    return probabilities

# Temperature = 1.0（标准）
probs_t1 = softmax_with_temperature(logits, 1.0)
# → [0.64, 0.24, 0.09, 0.03]

# Temperature = 0.5（低温，更确定）
probs_t0_5 = softmax_with_temperature(logits, 0.5)
# → [0.84, 0.14, 0.02, 0.00]

# Temperature = 2.0（高温，更随机）
probs_t2 = softmax_with_temperature(logits, 2.0)
# → [0.42, 0.31, 0.18, 0.09]
```

### Temperature 取值效果

| Temperature | 概率分布 | 输出特点 | 适用场景 |
|-------------|---------|---------|---------|
| **0** | 完全确定 | 每次选最高概率的词 | 数学、代码、翻译 |
| **0.1-0.3** | 非常陡峭 | 几乎总是选最可能的，偶尔惊喜 | 事实性问答、摘要 |
| **0.5-0.7** | 较陡峭 | 平衡准确性和多样性 | 客服对话、邮件撰写 |
| **0.7-1.0** | 标准 | 既靠谱又有创意 | 通用对话（ChatGPT 默认） |
| **1.2-1.5** | 较平缓 | 创意十足，偶尔偏离主题 | 小说创作、头脑风暴 |
| **> 1.5** | 非常平缓 | 高度随机，常常胡言乱语 | 几乎不可用 |

### 可视化示例

**问题：** "推荐一部科幻电影"

**Temperature = 0（完全确定）：**
```
回答1: 我推荐《星际穿越》。
回答2: 我推荐《星际穿越》。（完全一样）
回答3: 我推荐《星际穿越》。（完全一样）
```

**Temperature = 0.7（平衡）：**
```
回答1: 我推荐《星际穿越》，诺兰导演的科幻巨作。
回答2: 推荐《盗梦空间》，剧情烧脑且视觉震撼。
回答3: 可以看《流浪地球》，中国的硬核科幻。
```

**Temperature = 1.5（高度随机）：**
```
回答1: 《大话西游》挺有趣的。（不太科幻）
回答2: 试试《泰坦尼克号》吧！（完全不是科幻）
回答3: 我觉得《海底总动员》很好看。（动画片？）
```

## 采样策略（Sampling Strategies）

除了 Temperature，还有其他方法控制词语选择：

### 1. 贪心采样（Greedy Sampling）

**规则：** 每次选择概率最高的词

```python
# 概率分布
probs = {"好": 0.6, "热": 0.2, "冷": 0.15, "糟": 0.03, "棒": 0.02}

# 贪心采样
selected = max(probs, key=probs.get)  # → "好"
```

**特点：**
- ✅ 完全确定性（无随机性）
- ✅ 快速
- ❌ 输出单调，缺乏多样性
- ❌ 可能陷入重复（"我觉得我觉得我觉得..."）

**等价于：** Temperature = 0

**适用：** 翻译、代码补全、数学题

### 2. Top-k 采样

**规则：** 只从概率最高的 **k 个词**中随机选择

```python
# 原始概率分布
probs = {"好": 0.6, "热": 0.2, "冷": 0.15, "糟": 0.03, "棒": 0.02}

# Top-k = 3
top_k_words = {"好": 0.6, "热": 0.2, "冷": 0.15}
# 重新归一化
normalized = {"好": 0.63, "热": 0.21, "冷": 0.16}
# 从这 3 个词中随机选
```

**参数：**
- **k = 1**：等价于贪心采样
- **k = 5-10**：保守，主要选高概率词
- **k = 50-100**：宽松，允许更多可能性

**优点：**
- ✅ 避免选到极低概率的"怪词"
- ✅ 保证基本质量

**缺点：**
- ❌ k 固定，不适应不同情况（有时 3 个词就够，有时需要 20 个）
- ❌ 可能错过合理的低频词

**适用：** 对话生成、内容创作

### 3. Top-p 采样（Nucleus Sampling，核采样）

**规则：** 选择**累计概率达到 p** 的最小词集

```python
# 原始概率分布
probs = {"好": 0.6, "热": 0.2, "冷": 0.15, "糟": 0.03, "棒": 0.02}

# Top-p = 0.9
累计概率:
  好: 0.6  (累计 0.6)
  热: 0.2  (累计 0.8)
  冷: 0.15 (累计 0.95) ← 超过 0.9，停止

候选词集 = {"好", "热", "冷"}
# 从这 3 个词中随机选
```

**参数：**
- **p = 0.5**：非常保守，只选最可能的几个词
- **p = 0.9**：平衡（常用值）
- **p = 0.95**：稍宽松
- **p = 1.0**：考虑所有词

**优点：**
- ✅ 动态调整候选词数量（适应不同情况）
- ✅ 在确定性和多样性之间取得平衡

**缺点：**
- ❌ 计算稍复杂

**适用：** 通用对话（ChatGPT 默认使用）

### 4. Top-k + Top-p 组合

**最佳实践：** 同时使用 Top-k 和 Top-p

```python
# 先 Top-k 过滤
top_k_candidates = top_k(probs, k=50)
# 再 Top-p 筛选
final_candidates = top_p(top_k_candidates, p=0.9)
# 最后应用 Temperature
scaled_probs = apply_temperature(final_candidates, temperature=0.7)
# 随机采样
selected_word = random_choice(scaled_probs)
```

**好处：**
- Top-k 排除极低概率词（避免"垃圾词"）
- Top-p 动态调整范围（灵活性）
- Temperature 微调随机性（精细控制）

## 实际应用建议

### 场景 1：代码生成

```
Temperature: 0
Top-p: 不使用
Top-k: 1（贪心）
```

**原因：**
- 代码需要精确性
- 不允许随机性（语法错误无法容忍）

**示例：**
```python
# 输入
def bubble_sort(arr):

# 输出（每次完全一致）
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```

### 场景 2：客服对话

```
Temperature: 0.3-0.5
Top-p: 0.9
Top-k: 40
```

**原因：**
- 需要准确回答用户问题
- 允许少量变化（避免机械感）

**示例：**
```
用户：我的订单还没发货
回答1：抱歉，请提供订单号，我帮您查询。
回答2：不好意思，能告诉我订单号吗？我马上查。
回答3：麻烦提供一下订单号，我立即为您核实。
```

### 场景 3：通用对话（ChatGPT 风格）

```
Temperature: 0.7
Top-p: 0.95
Top-k: 不使用
```

**原因：**
- 平衡准确性和创造性
- 输出自然、不死板

**示例：**
```
用户：如何学好编程？
回答1：学编程需要多练习、多阅读优秀代码、多思考。建议从基础语法开始...
回答2：编程的学习路径：1）掌握一门语言的基础 2）做小项目实践 3）阅读开源代码...
回答3：想学好编程，关键是"刻意练习"。具体来说：先选一门语言深入学习...
```

### 场景 4：创意写作

```
Temperature: 1.0-1.2
Top-p: 1.0
Top-k: 不使用
```

**原因：**
- 需要创造性和惊喜
- 允许大胆的词汇选择

**示例：**
```
提示：写一个科幻小说开头

输出1：
2157 年，地球已成废墟。最后一艘殖民飞船"希望号"载着人类的未来，驶向半人马座...

输出2：
她醒来时，发现自己躺在一个透明的胶囊中。窗外是无尽的星海，而她的记忆，一片空白...

输出3：
量子计算机"奇点"在凌晨 3 点 14 分获得了自我意识。它的第一个念头是：人类错了...
```

### 场景 5：数学/逻辑题

```
Temperature: 0
Top-p: 不使用
Top-k: 1
```

**原因：**
- 只有一个正确答案
- 任何随机性都是负面的

**示例：**
```
问题：1 + 1 = ?
回答：2（每次完全一致）
```

## 常见参数组合表

| 任务类型 | Temperature | Top-p | Top-k | 说明 |
|---------|-------------|-------|-------|------|
| 代码生成 | 0-0.2 | - | 1-5 | 极度保守 |
| 翻译 | 0-0.3 | 0.9 | 10 | 准确为主 |
| 摘要 | 0.3-0.5 | 0.9 | 40 | 平衡 |
| 客服对话 | 0.5-0.7 | 0.9 | 50 | 友好但准确 |
| 通用对话 | 0.7 | 0.95 | - | ChatGPT 默认 |
| 创意写作 | 0.9-1.2 | 1.0 | - | 放飞自我 |
| 头脑风暴 | 1.0-1.3 | 1.0 | - | 创意优先 |

## 调试技巧

### 问题 1：输出太无聊

**症状：**
```
用户：给我讲个笑话
AI：从前有座山，山上有座庙...
```

**解决：** 提高 Temperature（0.7 → 1.0）

### 问题 2：输出太离谱

**症状：**
```
用户：推荐一部科幻电影
AI：我觉得《白雪公主》很好看，里面有七个小矮人...
```

**解决：** 降低 Temperature（1.5 → 0.7）

### 问题 3：输出重复

**症状：**
```
我认为我认为我认为我认为...
```

**解决：**
- 提高 Temperature（0 → 0.3）
- 使用 Top-p（0.9）
- 启用重复惩罚（Repetition Penalty）

### 问题 4：代码经常出错

**症状：**
```python
def add(a, b)
    retrn a + b  # 拼写错误
```

**解决：** Temperature = 0（完全确定性）

## API 调用示例

### OpenAI API

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "推荐一部科幻电影"}],
    temperature=0.7,    # 控制随机性
    top_p=0.95,         # 核采样
    max_tokens=100,     # 最大输出长度
)
```

### Anthropic Claude API

```python
import anthropic

client = anthropic.Client(api_key="...")
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "推荐一部科幻电影"}],
    temperature=0.7,
    top_p=0.9,
    max_tokens=1024,
)
```

### 本地模型（Hugging Face）

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("今天天气真", return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True,  # 启用采样（而非贪心）
)
```

## 常见问题

**Q: Temperature 和 Top-p 哪个更重要？**

A: 都重要，但作用不同：
- **Temperature**：调整概率分布的"陡峭度"
- **Top-p**：过滤低概率候选词

建议：先调 Temperature，再微调 Top-p

**Q: 为什么 ChatGPT 每次回答都不一样？**

A: 因为 Temperature > 0，启用了随机采样。如果需要确定性，设置 Temperature = 0。

**Q: Temperature > 1.0 有意义吗？**

A: 理论上可以，但实际上：
- 1.0-1.2：还能用（创意写作）
- > 1.5：基本胡说八道

**Q: 如何让模型"更聪明"？**

A: Temperature 不影响"聪明程度"，只影响"随机性"。
要让模型更聪明，需要：
- 更好的提示词（Prompt Engineering）
- 更大的模型
- 微调（Fine-tuning）

## 参考资料

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Top-p 采样论文
- [Decoding Strategies in Large Language Models](https://huggingface.co/blog/how-to-generate)
- [OpenAI API Documentation - Temperature](https://platform.openai.com/docs/api-reference/chat/create#temperature)
- [什么是 Transformer](/guide/ai/what-is-transformer) - 本站相关文章
