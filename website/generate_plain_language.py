#!/usr/bin/env python3
"""
批量生成所有文档的大白话解释
使用 prompt.md 中的提示词模板
将生成的大白话解释添加到每个文档的 frontmatter 中
"""
import os
import re
from pathlib import Path

def read_prompt_template():
    """读取 prompt.md 中的提示词模板"""
    prompt_file = Path('docs/prompt.md')
    if prompt_file.exists():
        return prompt_file.read_text(encoding='utf-8')
    return """你是一位能让博士论文变成茶余饭后谈资的语言大师。

=== 核心使命 ===
把让人头大的学术词汇，翻译成让人会心一笑的大白话。

=== 价值追求 ===
- 让博导听了想打人，让大爷听了拍大腿
- 宁可粗暴，不可晦涩
- 精髓不丢，装腔全扔
- 最好能让人边笑边懂

=== 世俗化的"味道" ===
好的世俗化应该：
- 像在撸串时跟哥们儿解释，不是在开学术研讨会
- 用菜市场大妈都懂的例子，不是实验室的小白鼠
- 要有"就这？"的恍然大悟感，不是"原来如此"的一本正经

=== 边界 ===
别把"进化论"翻译成"猴子变人"——过度简化就成误导了。

请用上述原则，用大白话解释以下内容：

标题：{title}
内容：{content}"""

def extract_content_text(content):
    """从 markdown 内容中提取纯文本"""
    # 移除图片引用
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    # 移除链接，保留文本
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    # 移除代码块
    content = re.sub(r'```[\s\S]*?```', '', content)
    content = re.sub(r'`[^`]+`', '', content)
    # 移除标题标记
    content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
    # 移除多余的空白
    content = re.sub(r'\n\s*\n+', '\n\n', content)
    # 限制长度
    return content.strip()[:2000]

def parse_frontmatter(content):
    """解析 frontmatter"""
    if not content.startswith('---'):
        return {}, content
    
    parts = content.split('---', 2)
    if len(parts) < 3:
        return {}, content
    
    frontmatter_text = parts[1].strip()
    body = parts[2]
    
    # 简单的 frontmatter 解析
    metadata = {}
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # 处理多行值（以 | 开头）
            if value.startswith('|'):
                # 跳过，暂时不支持多行
                continue
            metadata[key] = value
    
    return metadata, body

def format_frontmatter(metadata):
    """格式化 frontmatter"""
    lines = ['---']
    for key, value in metadata.items():
        if isinstance(value, str) and '\n' in value:
            # 多行值使用 | 格式
            lines.append(f'{key}: |')
            for line in value.split('\n'):
                lines.append(f'  {line}')
        else:
            # 单行值
            value_str = str(value).replace('"', '\\"')
            if ':' in value_str or value_str.startswith(' '):
                lines.append(f'{key}: "{value_str}"')
            else:
                lines.append(f'{key}: {value_str}')
    lines.append('---')
    return '\n'.join(lines)

def generate_plain_language_explanation(title, content, prompt_template):
    """
    生成大白话解释
    这里可以调用 AI API，目前使用占位文本
    """
    # TODO: 调用 AI API 生成真实的大白话解释
    # 示例：
    # import openai
    # prompt = prompt_template.format(title=title, content=content)
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{
    #         "role": "system",
    #         "content": prompt
    #     }]
    # )
    # return response.choices[0].message.content
    
    # 临时占位文本（基于 prompt.md 的原则）
    return f"""**{title}** 说白了就是...

用最简单的话来说，这个概念的核心就是让复杂的东西变得简单。就像你在跟朋友解释一个技术概念时，不会用那些拗口的专业术语，而是用日常生活中的例子来类比。

**举个例子：**
如果这个概念是一个工具，那它就像是你在日常生活中会用到的某个东西。它的作用就是帮你解决某个问题，让你不用那么费劲就能理解或使用。

**记住：**
- 精髓不丢，装腔全扔
- 我们要的是理解，不是背诵
- 宁可粗暴，不可晦涩

*提示：这是占位文本。需要调用 AI API 生成真实的大白话解释。*"""

def process_all_docs():
    """处理所有文档，生成大白话解释"""
    docs_dir = Path('docs/guide')
    prompt_template = read_prompt_template()
    processed = 0
    skipped = 0
    errors = 0
    
    for md_file in sorted(docs_dir.rglob('*.md')):
        try:
            # 读取文件
            content = md_file.read_text(encoding='utf-8')
            metadata, body = parse_frontmatter(content)
            
            # 如果已经有 plainLanguage 字段，跳过
            if 'plainLanguage' in metadata:
                skipped += 1
                print(f"⏭  跳过（已有解释）: {md_file.name}")
                continue
            
            # 提取内容
            title = metadata.get('title', md_file.stem)
            text_content = extract_content_text(body)
            
            # 生成大白话解释
            print(f"🔄 正在处理: {title}...")
            plain_language = generate_plain_language_explanation(title, text_content, prompt_template)
            
            # 添加到 frontmatter
            metadata['plainLanguage'] = plain_language
            
            # 重新组合文件内容
            frontmatter_text = format_frontmatter(metadata)
            new_content = f"{frontmatter_text}\n\n{body}"
            
            # 保存文件
            md_file.write_text(new_content, encoding='utf-8')
            
            processed += 1
            print(f"✓ 完成: {title}")
            
        except Exception as e:
            errors += 1
            print(f"✗ 错误处理 {md_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"完成！处理了 {processed} 个文档")
    print(f"跳过了 {skipped} 个已有解释的文档")
    if errors > 0:
        print(f"错误: {errors} 个文档")
    print(f"{'='*50}")

if __name__ == '__main__':
    process_all_docs()
