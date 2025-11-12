import { useState } from 'react';
import { usePageData } from 'rspress/runtime';
import './PlainLanguageExplanation.css';

export function PlainLanguageExplanation() {
  const [isExpanded, setIsExpanded] = useState(false);
  const pageData = usePageData();

  // 调试：打印 pageData
  console.log('PlainLanguageExplanation - pageData:', {
    pagePath: pageData.page?.pagePath,
    hasFrontmatter: !!(pageData.page as any)?.frontmatter,
    hasPlainLanguage: !!(pageData.page as any)?.frontmatter?.plainLanguage,
    plainLanguageLength: (pageData.page as any)?.frontmatter?.plainLanguage?.length,
  });

  // 检查是否是首页，首页不显示
  if (pageData.page?.pagePath === '/' || pageData.page?.pagePath === '/index') {
    return null;
  }

  // 从 frontmatter 中获取预生成的大白话解释
  const plainLanguage = (pageData.page as any)?.frontmatter?.plainLanguage;

  // 如果没有大白话解释，不显示模块
  if (!plainLanguage) {
    console.log('PlainLanguageExplanation - No plainLanguage found');
    return null;
  }

  console.log('PlainLanguageExplanation - Rendering with plainLanguage:', plainLanguage.substring(0, 100));

  const handleToggle = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="plain-language-explanation">
      <button
        className="explanation-toggle"
        onClick={handleToggle}
        aria-expanded={isExpanded}
      >
        <span className="explanation-icon">💡</span>
        <span className="explanation-title">大白话解释</span>
        <span className="explanation-subtitle">用最简单的话说清楚</span>
        <span className="explanation-arrow">{isExpanded ? '▼' : '▶'}</span>
      </button>
      {isExpanded && (
        <div className="explanation-content">
          <div 
            className="explanation-text"
            dangerouslySetInnerHTML={{ __html: formatExplanation(plainLanguage) }}
          />
        </div>
      )}
    </div>
  );
}

// 格式化解释文本（支持 Markdown）
function formatExplanation(text: string): string {
  // 简单的 Markdown 转 HTML
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^(.+)$/, '<p>$1</p>');
}
