# Vercel 部署快速指南

## 🚀 一键部署

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/karminski/one-small-step&root-directory=website)

## ⚙️ 手动部署步骤

### 1. 导入项目到 Vercel

访问 [Vercel Dashboard](https://vercel.com/dashboard) → New Project → 导入 GitHub 仓库

### 2. ⭐ 重要配置

**Root Directory**: 必须设置为 `website`

![Vercel Root Directory 设置](https://vercel.com/_next/image?url=%2Fdocs-proxy%2Fstatic%2Fdocs%2Fconcepts%2Fprojects%2Froot-directory.png&w=3840&q=75)

其他配置会自动从 `website/vercel.json` 读取：
- Build Command: `npm run build`
- Output Directory: `dist`
- Install Command: `npm install`

### 3. 点击 Deploy

等待几分钟，你的文档站就上线了！🎉

## 📋 为什么要设置 Root Directory？

```
one-small-step/
├── 原始文章/          # Git 原有内容
└── website/          # ⭐ RSPress 网站（从这里开始构建）
    ├── package.json
    ├── docs/
    └── theme/
```

## 🔍 验证部署

部署成功后访问你的网站：
- ✅ 首页显示正常
- ✅ 文章列表可访问
- ✅ 搜索功能正常
- ✅ 图片正常加载
- ✅ "大白话解释" 模块显示

## 🆘 遇到问题？

### 问题：构建失败 "Cannot find module 'rspress'"

**原因**：Root Directory 未设置或设置错误

**解决**：进入项目设置 → Root Directory → 设置为 `website` → Redeploy

### 问题：页面空白或 404

**原因**：Output Directory 路径错误

**解决**：确认 Root Directory 为 `website`，Output Directory 为 `dist`（相对于 website 目录）

## 📚 更多帮助

- [完整部署指南](./DEPLOYMENT.md)
- [Vercel 官方文档](https://vercel.com/docs)
- [RSPress 文档](https://rspress.dev)

