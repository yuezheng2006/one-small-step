# One Small Step - RSPress 文档站

## 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

## Vercel 部署

项目已配置好 Vercel 部署：

1. 将代码推送到 GitHub
2. 在 Vercel 中导入项目
3. Vercel 会自动检测配置并部署

### 部署配置

- **构建命令**: `npm run build`
- **输出目录**: `dist`
- **Node 版本**: 18.x 或更高

## 项目结构

```
one-small-step/
├── docs/                    # 文档目录
│   ├── index.md            # 首页
│   ├── guide/              # 文章目录
│   │   ├── ai/            # AI 相关文章
│   │   ├── math/          # 数学相关文章
│   │   ├── system/        # 系统相关文章
│   │   └── hardware/      # 硬件相关文章
│   └── assets/            # 图片资源
├── rspress.config.ts       # RSPress 配置
├── package.json           # 项目依赖
└── vercel.json            # Vercel 部署配置
```

## 添加新文章

1. 在对应的分类目录下创建 markdown 文件
2. 添加 frontmatter：
```markdown
---
title: 文章标题
description: 文章描述
author: '@karminski-牙医'
date: YYYYMMDD
---
```
3. 在 `rspress.config.ts` 中添加导航链接


