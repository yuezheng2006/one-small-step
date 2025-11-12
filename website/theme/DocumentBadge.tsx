import { usePageData } from 'rspress/runtime';
import './DocumentBadge.css';

export function DocumentBadge() {
  const pageData = usePageData();

  // 检查是否是首页，首页不显示
  if (pageData.page?.pagePath === '/' || pageData.page?.pagePath === '/index') {
    return null;
  }

  // 从 frontmatter 中获取作者信息
  const author = (pageData.page as any)?.frontmatter?.author;

  // 如果作者是 "AI收集"，显示新增内容标识
  if (author === 'AI收集') {
    return (
      <div className="document-badge">
        <span className="badge-icon">✨</span>
        <span className="badge-text">新增内容</span>
        <span className="badge-description">AI 收集整理</span>
      </div>
    );
  }

  return null;
}
