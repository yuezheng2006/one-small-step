#!/usr/bin/env python3
"""确保 plainLanguage 字段格式正确"""

import os
import re
from pathlib import Path

def fix_plainlanguage_format(file_path):
    """修复 plainLanguage 字段的格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # 确保 plainLanguage 字段的每一行都以两个空格开头
        # 并且末尾没有多余的空格
        lines = content.split('\n')
        new_lines = []
        in_plainlanguage = False
        
        for i, line in enumerate(lines):
            if line.strip() == 'plainLanguage: |':
                in_plainlanguage = True
                new_lines.append(line)
            elif in_plainlanguage:
                if line.strip() == '---':
                    in_plainlanguage = False
                    new_lines.append(line)
                elif line.startswith('  ') or line.strip() == '':
                    # 移除行尾空格
                    new_lines.append(line.rstrip())
                else:
                    # 不是plainLanguage的内容了
                    in_plainlanguage = False
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ 处理失败 {file_path.name}: {e}")
        return False

def main():
    docs_dir = Path('/Users/vincentyang/Documents/Github/one-small-step/website/docs/guide')
    
    print("🔍 开始修复 plainLanguage 格式...")
    print(f"📁 目录: {docs_dir}\n")
    
    fixed_count = 0
    total_count = 0
    
    for md_file in docs_dir.rglob('*.md'):
        if md_file.name == 'prompt.md':
            continue
            
        total_count += 1
        
        if fix_plainlanguage_format(md_file):
            fixed_count += 1
            print(f"✅ {md_file.relative_to(docs_dir)}")
    
    print(f"\n{'='*50}")
    print(f"✨ 完成！共处理 {total_count} 个文件，修复了 {fixed_count} 个文件")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()

