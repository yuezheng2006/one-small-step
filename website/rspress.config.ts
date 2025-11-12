import { defineConfig } from 'rspress/config';
import { pluginLastUpdated } from '@rspress/plugin-last-updated';
import path from 'path';

export default defineConfig({
  root: 'docs',
  title: 'One Small Step',
  description: '技术科普教程项目，主要聚焦于解释一些有趣的、前沿的技术概念和原理',
  lang: 'zh',
  themeDir: path.join(__dirname, 'theme'),
  themeConfig: {
    nav: [
      {
        text: '首页',
        link: '/',
      },
      {
        text: '指南',
        link: '/guide/ai/what-is-gguf',
      },
    ],
    footer: {
      message: 'MIT Licensed | Copyright © 2025',
    },
    sidebar: {
      '/guide/': [
        {
          text: '人工智能相关',
          items: [
            {
              text: '什么是 GGUF',
              link: '/guide/ai/what-is-gguf',
            },
            {
              text: '什么是推测性解码',
              link: '/guide/ai/what-is-speculative-decoding',
            },
            {
              text: '什么是 Pythonic 函数调用',
              link: '/guide/ai/what-is-pythonic-function-call',
            },
            {
              text: '如何本地运行 GGUF 格式的 LLM 模型',
              link: '/guide/ai/how-to-run-gguf-LLM-model',
            },
            {
              text: '什么是 LLM 蒸馏技术',
              link: '/guide/ai/what-is-LLM-distill',
            },
            {
              text: '什么是 Transformer',
              link: '/guide/ai/what-is-transformer',
            },
            {
              text: '如何优化 Transformer',
              link: '/guide/ai/how-to-optimize-transformer',
            },
            {
              text: '什么是大语言模型量化',
              link: '/guide/ai/what-is-quantization-in-LLM',
            },
            {
              text: '什么是 Flash Attention',
              link: '/guide/ai/what-is-flash-attention',
            },
            {
              text: '什么是 Multi-Head Attention',
              link: '/guide/ai/what-is-multi-head-attention',
            },
            {
              text: '什么是 Multi-Query Attention',
              link: '/guide/ai/what-is-multi-query-attention',
            },
            {
              text: '什么是 Grouped Query Attention',
              link: '/guide/ai/what-is-gropued-query-attention',
            },
            {
              text: '什么是 LLM 微调技术',
              link: '/guide/ai/what-is-LLM-fine-tuning',
            },
            {
              text: '什么是 RAG 技术',
              link: '/guide/ai/what-is-RAG',
            },
            {
              text: '什么是 Safetensors',
              link: '/guide/ai/what-is-safetensors',
            },
            {
              text: '什么是 ONNX',
              link: '/guide/ai/what-is-onnx',
            },
            {
              text: '大模型微调最佳实践指南',
              link: '/guide/ai/LLM-fine-tuning-summary',
            },
            {
              text: '什么是 MoE 模型',
              link: '/guide/ai/what-is-MoE',
            },
            {
              text: 'LLM 中的 Token 是如何计算的',
              link: '/guide/ai/how-are-tokens-calculated-in-LLMs',
            },
            {
              text: '什么是 AI Agent',
              link: '/guide/ai/what-is-AI-Agent',
            },
            {
              text: '什么是 LoRA',
              link: '/guide/ai/what-is-LoRA',
            },
            {
              text: '什么是向量嵌入',
              link: '/guide/ai/what-is-vector-embedding',
            },
            {
              text: '什么是向量数据库',
              link: '/guide/ai/what-is-vector-database',
            },
            {
              text: '什么是 AI 幻觉',
              link: '/guide/ai/what-is-AI-Hallucination',
            },
            {
              text: '什么是模态编码',
              link: '/guide/ai/what-is-modal-encoding',
            },
            {
              text: '什么是表示空间',
              link: '/guide/ai/what-is-representation-space',
            },
            {
              text: '什么是多模态模型',
              link: '/guide/ai/what-is-multi-model-llm',
            },
            {
              text: '什么是 LLM 的困惑度',
              link: '/guide/ai/what-is-llm-perplexity',
            },
            {
              text: '如何避免 KVCache 失效',
              link: '/guide/ai/How-to-avoid-KVCache-invalidation',
            },
            {
              text: '什么是 Sliding Window Attention',
              link: '/guide/ai/what-is-sliding-window-attention',
            },
            {
              text: '什么时候应该微调, 什么时候不应该微调?',
              link: '/guide/ai/When-to-Use-Fine-Tuning-and-When-Not-To',
            },
            {
              text: '什么是vibe coding?',
              link: '/guide/ai/what-is-vibe-coding',
            },
            {
              text: 'Qwen3 扩展到 1M 上下文是如何做到的?',
              link: '/guide/ai/What-is-Dual-Chunk-Attention',
            },
            {
              text: '什么是召回',
              link: '/guide/ai/What-is-Recall',
            },
            {
              text: '大模型精度格式一览',
              link: '/guide/ai/Parameter-Precision-Formats-for-LLMs',
            },
          ],
        },
        {
          text: '数学相关',
          items: [
            {
              text: '什么是矩阵的秩？什么是低秩矩阵？',
              link: '/guide/math/what-is-rank-in-matrix',
            },
            {
              text: '什么是拟合与过拟合',
              link: '/guide/math/what-is-fitting-and-overfitting',
            },
          ],
        },
        {
          text: '系统相关',
          items: [
            {
              text: 'Windows 任务管理器内存标签说明',
              link: '/guide/system/windows-task-manager-memory-tab-description',
            },
            {
              text: 'RAMMap 使用解析',
              link: '/guide/system/rammap-description',
            },
          ],
        },
        {
          text: '硬件相关',
          items: [
            {
              text: '什么是 PCIe Retimer',
              link: '/guide/hardware/what-is-pcie-retimer',
            },
            {
              text: '为什么有的 NVMe SSD 有 DRAM, 有的没有?',
              link: '/guide/hardware/why-some-NVMe-SSD-have-DRAM-and-some-are-not',
            },
            {
              text: 'CLX 会是大语言模型的内存解决方案吗?',
              link: '/guide/hardware/does-CXL-will-be-LLM-memory-solution',
            },
            {
              text: '什么是 1DPC',
              link: '/guide/hardware/what-is-1DPC',
            },
            {
              text: '什么是 L1 缓存',
              link: '/guide/hardware/what-is-L1-cache',
            },
          ],
        },
      ],
    },
    search: true,
    outlineTitle: '目录',
    prevPageText: '上一页',
    nextPageText: '下一页',
  },
  plugins: [pluginLastUpdated()],
  markdown: {
    checkDeadLinks: false,
  },
});

