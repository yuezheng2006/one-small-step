#!/usr/bin/env python3
"""清理 markdown frontmatter 中的特殊字符和格式问题"""

import os
import re
from pathlib import Path

def clean_frontmatter(file_path):
    """清理文件的 frontmatter"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 frontmatter
        match = re.match(r'---\n(.*?)\n---\n(.*)', content, re.DOTALL)
        if not match:
            print(f"  ⚠️  没有找到 frontmatter: {file_path.name}")
            return False
        
        fm_content = match.group(1)
        body = match.group(2)
        
        # 清理特殊字符
        original_fm = fm_content
        
        # 1. 替换特殊空格字符为普通空格
        fm_content = fm_content.replace('\u00a0', ' ')  # 不间断空格
        fm_content = fm_content.replace('\u200b', '')   # 零宽空格
        fm_content = fm_content.replace('\u3000', ' ')  # 全角空格
        
        # 2. 清理双句号
        fm_content = re.sub(r'。。+', '。', fm_content)
        
        # 3. 统一行尾
        fm_content = fm_content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 4. 确保 plainLanguage 字段正确缩进
        fm_content = re.sub(r'plainLanguage: \|(\n(?:  .+\n)*)', lambda m: 'plainLanguage: |' + m.group(1), fm_content)
        
        if fm_content != original_fm:
            # 写回文件
            new_content = f"---\n{fm_content}\n---\n{body}"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ 处理失败 {file_path.name}: {e}")
        return False

def main():
    docs_dir = Path('/Users/vincentyang/Documents/Github/one-small-step/website/docs/guide')
    
    print("🔍 开始清理 frontmatter...")
    print(f"📁 目录: {docs_dir}\n")
    
    fixed_count = 0
    total_count = 0
    
    for md_file in docs_dir.rglob('*.md'):
        if md_file.name == 'prompt.md':
            continue
            
        total_count += 1
        print(f"检查: {md_file.relative_to(docs_dir)}")
        
        if clean_frontmatter(md_file):
            fixed_count += 1
            print(f"  ✅ 已清理")
        else:
            print(f"  ⏭️  跳过（无需修改）")
    
    print(f"\n{'='*50}")
    print(f"✨ 完成！共处理 {total_count} 个文件，修复了 {fixed_count} 个文件")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()

