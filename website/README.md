# One Small Step - 网站文档

这是 One Small Step 项目的 RSPress 文档网站部分。

## 🚀 快速开始

### 安装依赖
```bash
npm install
```

### 本地开发
```bash
npm run dev
# 或指定端口
npm run dev -- --port 4173
```

### 构建生产版本
```bash
npm run build
```

### 预览生产版本
```bash
npm run preview
```

## 📁 目录结构

```
website/
├── docs/                     # 文档内容
│   ├── index.md             # 首页
│   ├── guide/               # 文章目录
│   │   ├── ai/             # 人工智能相关
│   │   ├── hardware/       # 硬件相关
│   │   ├── math/           # 数学相关
│   │   └── system/         # 系统相关
│   └── public/             # 静态资源
│       └── assets/         # 图片等资源
├── theme/                   # 自定义主题
│   ├── index.tsx           # 主题入口
│   ├── index.css           # 主题样式
│   ├── PlainLanguageExplanation.tsx  # 大白话解释组件
│   └── PlainLanguageExplanation.css  # 组件样式
├── rspress.config.ts        # RSPress 配置
├── package.json             # 项目依赖
├── tsconfig.json            # TypeScript 配置
├── vercel.json              # Vercel 部署配置
└── generate_plain_language.py  # 大白话生成脚本

```

## 🌐 部署

### Vercel 部署

**⚠️ 重要：必须设置 Root Directory 为 `website`**

#### 方法一：通过 Vercel Dashboard（推荐）

1. 将项目推送到 GitHub
2. 在 [Vercel](https://vercel.com) 中导入项目
3. **⭐ 关键步骤**：在项目配置中，设置 **Root Directory** 为 `website`
4. 其他配置会自动从 `vercel.json` 读取
5. 点击 Deploy

#### 方法二：通过 Vercel CLI

```bash
cd website
npm install -g vercel
vercel login
vercel
```

#### 为什么要设置 Root Directory？

项目结构已重组，原始文章在根目录，RSPress 网站在 `website/` 目录：

```
one-small-step/
├── 20250104-*/       # 原始文章（Git 内容）
└── website/          # RSPress 网站（部署此目录）
    ├── package.json  # npm 依赖
    ├── docs/         # 文档内容
    └── theme/        # 自定义主题
```

Vercel 需要从 `website/` 目录读取 `package.json` 和其他配置文件。

详见：[../VERCEL_DEPLOY.md](../VERCEL_DEPLOY.md)

## 📝 添加新文章

1. 在 `docs/guide/` 对应分类目录下创建新的 markdown 文件
2. 添加 frontmatter：
```markdown
---
title: 文章标题
description: 文章描述
author: @作者名
date: 20250101
plainLanguage: |
  大白话解释内容...
---
```
3. 更新 `rspress.config.ts` 中的侧边栏配置

## 🎨 主题定制

主题文件位于 `theme/` 目录，可以自定义：
- 导航栏
- 侧边栏
- 页面布局
- 样式和动画效果

详见 [RSPress 主题定制文档](https://rspress.rs/zh/guide/advanced/custom-theme)

## 💡 大白话解释功能

每篇文章底部会显示"大白话解释"模块，内容来自文章 frontmatter 的 `plainLanguage` 字段。

可以使用 `generate_plain_language.py` 批量生成（需要配置 AI API）。

## 📄 License

MIT License - 详见根目录 LICENSE 文件

