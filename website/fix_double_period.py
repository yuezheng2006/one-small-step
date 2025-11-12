#!/usr/bin/env python3
"""移除 markdown frontmatter 中的双句号"""

import os
import re
from pathlib import Path

def fix_double_period(file_path):
    """修复文件中的双句号问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 在 plainLanguage 字段中，将 。。 替换为 。
        content = re.sub(
            r'(plainLanguage:.*?---)',
            lambda m: m.group(0).replace('。。', '。'),
            content,
            flags=re.DOTALL
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ 处理失败 {file_path.name}: {e}")
        return False

def main():
    docs_dir = Path('/Users/vincentyang/Documents/Github/one-small-step/website/docs/guide')
    
    print("🔍 开始移除双句号...")
    print(f"📁 目录: {docs_dir}\n")
    
    fixed_count = 0
    total_count = 0
    
    for md_file in docs_dir.rglob('*.md'):
        if md_file.name == 'prompt.md':
            continue
            
        total_count += 1
        
        if fix_double_period(md_file):
            fixed_count += 1
            print(f"✅ {md_file.relative_to(docs_dir)}")
    
    print(f"\n{'='*50}")
    print(f"✨ 完成！共处理 {total_count} 个文件，修复了 {fixed_count} 个文件")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()

