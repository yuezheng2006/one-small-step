import yaml

test_content = """---
title: 什么是 Multi-Head Attention?
description: 什么是 Multi-Head Attention?
author: @karminski-牙医
date: 20250202
plainLanguage: |
  **Multi-Head Attention 说白了就是：** 让 AI "多角度看问题"，而不是只从一个角度看。
  
  说白了，多头注意力就是"三个臭皮匠，顶个诸葛亮"的 AI 版本——虽然每个头单独看可能不够聪明，但合起来就很强大。
---"""

try:
    data = yaml.safe_load(test_content.split('---')[1])
    print("✅ YAML 解析成功")
    print(f"plainLanguage 长度: {len(data['plainLanguage'])}")
except Exception as e:
    print(f"❌ YAML 解析失败: {e}")
