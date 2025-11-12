import sharp from 'sharp';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const imagesDir = path.join(__dirname, 'docs/public/assets/images');

// SVG 模板
const svgs = {
  'position-encoding': `
    <svg width="1200" height="600" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#00d9ff;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#7c3aed;stop-opacity:1" />
        </linearGradient>
      </defs>

      <!-- 背景 -->
      <rect width="1200" height="600" fill="#f8fafc"/>

      <!-- 标题 -->
      <text x="600" y="50" font-size="32" font-weight="bold" text-anchor="middle" fill="#1e293b">
        Position Encoding（位置编码）
      </text>

      <!-- 示意图：输入序列 -->
      <g transform="translate(100, 120)">
        <text x="0" y="0" font-size="20" fill="#475569">输入序列</text>
        <rect x="0" y="20" width="100" height="60" fill="#e0f2fe" stroke="#00d9ff" stroke-width="2" rx="5"/>
        <text x="50" y="55" font-size="16" text-anchor="middle" fill="#1e293b">我</text>

        <rect x="120" y="20" width="100" height="60" fill="#e0f2fe" stroke="#00d9ff" stroke-width="2" rx="5"/>
        <text x="170" y="55" font-size="16" text-anchor="middle" fill="#1e293b">爱</text>

        <rect x="240" y="20" width="100" height="60" fill="#e0f2fe" stroke="#00d9ff" stroke-width="2" rx="5"/>
        <text x="290" y="55" font-size="16" text-anchor="middle" fill="#1e293b">你</text>
      </g>

      <!-- 箭头 -->
      <path d="M 600 220 L 600 270" stroke="#475569" stroke-width="3" marker-end="url(#arrowhead)"/>
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="5" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#475569" />
        </marker>
      </defs>

      <!-- 位置编码 -->
      <g transform="translate(100, 300)">
        <text x="0" y="0" font-size="20" fill="#475569">+ 位置编码</text>
        <rect x="0" y="20" width="100" height="60" fill="#ede9fe" stroke="#7c3aed" stroke-width="2" rx="5"/>
        <text x="50" y="45" font-size="14" text-anchor="middle" fill="#1e293b">Pos 0</text>
        <text x="50" y="65" font-size="12" text-anchor="middle" fill="#64748b">[0.0, 1.0...]</text>

        <rect x="120" y="20" width="100" height="60" fill="#ede9fe" stroke="#7c3aed" stroke-width="2" rx="5"/>
        <text x="170" y="45" font-size="14" text-anchor="middle" fill="#1e293b">Pos 1</text>
        <text x="170" y="65" font-size="12" text-anchor="middle" fill="#64748b">[0.84, 0.54...]</text>

        <rect x="240" y="20" width="100" height="60" fill="#ede9fe" stroke="#7c3aed" stroke-width="2" rx="5"/>
        <text x="290" y="45" font-size="14" text-anchor="middle" fill="#1e293b">Pos 2</text>
        <text x="290" y="65" font-size="12" text-anchor="middle" fill="#64748b">[0.91, -0.42...]</text>
      </g>

      <!-- 结果 -->
      <g transform="translate(100, 450)">
        <text x="0" y="0" font-size="20" fill="#475569">编码后的输入</text>
        <rect x="0" y="20" width="100" height="60" fill="#dcfce7" stroke="#10b981" stroke-width="2" rx="5"/>
        <text x="50" y="45" font-size="14" text-anchor="middle" fill="#1e293b">我+位置</text>

        <rect x="120" y="20" width="100" height="60" fill="#dcfce7" stroke="#10b981" stroke-width="2" rx="5"/>
        <text x="170" y="45" font-size="14" text-anchor="middle" fill="#1e293b">爱+位置</text>

        <rect x="240" y="20" width="100" height="60" fill="#dcfce7" stroke="#10b981" stroke-width="2" rx="5"/>
        <text x="290" y="45" font-size="14" text-anchor="middle" fill="#1e293b">你+位置</text>
      </g>

      <!-- 说明文字 -->
      <g transform="translate(500, 350)">
        <text x="0" y="0" font-size="18" font-weight="bold" fill="#1e293b">位置编码方法：</text>
        <text x="0" y="35" font-size="16" fill="#475569">• 绝对位置：直接编号 0, 1, 2...</text>
        <text x="0" y="65" font-size="16" fill="#475569">• 正弦余弦：sin/cos 函数编码</text>
        <text x="0" y="95" font-size="16" fill="#475569">• RoPE：旋转位置编码</text>
      </g>
    </svg>
  `,

  'llm-training-pipeline': `
    <svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="800" fill="#f8fafc"/>

      <!-- 标题 -->
      <text x="600" y="50" font-size="32" font-weight="bold" text-anchor="middle" fill="#1e293b">
        LLM 训练流程：预训练 → SFT → RLHF
      </text>

      <!-- 阶段1：预训练 -->
      <g transform="translate(100, 120)">
        <rect width="300" height="180" fill="#dbeafe" stroke="#3b82f6" stroke-width="3" rx="10"/>
        <text x="150" y="30" font-size="22" font-weight="bold" text-anchor="middle" fill="#1e3a8a">
          阶段1：预训练
        </text>
        <text x="150" y="60" font-size="16" text-anchor="middle" fill="#475569">
          数据：数万亿 token
        </text>
        <text x="150" y="85" font-size="16" text-anchor="middle" fill="#475569">
          成本：数百万美元
        </text>
        <text x="150" y="110" font-size="16" text-anchor="middle" fill="#475569">
          时长：数月
        </text>
        <text x="150" y="140" font-size="16" text-anchor="middle" fill="#1e3a8a">
          目标：学习语言和知识
        </text>
      </g>

      <!-- 箭头1 -->
      <path d="M 400 210 L 480 210" stroke="#475569" stroke-width="3" marker-end="url(#arrow)"/>
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#475569" />
        </marker>
      </defs>

      <!-- 阶段2：监督微调 -->
      <g transform="translate(480, 120)">
        <rect width="300" height="180" fill="#fae8ff" stroke="#a855f7" stroke-width="3" rx="10"/>
        <text x="150" y="30" font-size="22" font-weight="bold" text-anchor="middle" fill="#6b21a8">
          阶段2：监督微调(SFT)
        </text>
        <text x="150" y="60" font-size="16" text-anchor="middle" fill="#475569">
          数据：数万条问答对
        </text>
        <text x="150" y="85" font-size="16" text-anchor="middle" fill="#475569">
          成本：数十万美元
        </text>
        <text x="150" y="110" font-size="16" text-anchor="middle" fill="#475569">
          时长：数周
        </text>
        <text x="150" y="140" font-size="16" text-anchor="middle" fill="#6b21a8">
          目标：学会遵循指令
        </text>
      </g>

      <!-- 箭头2 -->
      <path d="M 780 210 L 860 210" stroke="#475569" stroke-width="3" marker-end="url(#arrow)"/>

      <!-- 阶段3：RLHF -->
      <g transform="translate(860, 120)">
        <rect width="300" height="180" fill="#dcfce7" stroke="#10b981" stroke-width="3" rx="10"/>
        <text x="150" y="30" font-size="22" font-weight="bold" text-anchor="middle" fill="#065f46">
          阶段3：RLHF
        </text>
        <text x="150" y="60" font-size="16" text-anchor="middle" fill="#475569">
          数据：数千条排序数据
        </text>
        <text x="150" y="85" font-size="16" text-anchor="middle" fill="#475569">
          成本：数十万美元
        </text>
        <text x="150" y="110" font-size="16" text-anchor="middle" fill="#475569">
          时长：数周
        </text>
        <text x="150" y="140" font-size="16" text-anchor="middle" fill="#065f46">
          目标：符合人类偏好
        </text>
      </g>

      <!-- 能力对比 -->
      <g transform="translate(100, 380)">
        <text x="0" y="0" font-size="24" font-weight="bold" fill="#1e293b">能力演进：</text>

        <g transform="translate(0, 40)">
          <circle cx="10" cy="10" r="8" fill="#3b82f6"/>
          <text x="30" y="15" font-size="18" fill="#475569">预训练后：知识丰富但不会对话</text>
          <text x="30" y="40" font-size="15" fill="#64748b">
            示例："今天天气" → "很好明天后天大后天..." (续写，不是回答)
          </text>
        </g>

        <g transform="translate(0, 100)">
          <circle cx="10" cy="10" r="8" fill="#a855f7"/>
          <text x="30" y="15" font-size="18" fill="#475569">SFT后：会遵循指令</text>
          <text x="30" y="40" font-size="15" fill="#64748b">
            示例："今天天气怎么样？" → "今天天气晴朗，温度20度。"
          </text>
        </g>

        <g transform="translate(0, 160)">
          <circle cx="10" cy="10" r="8" fill="#10b981"/>
          <text x="30" y="15" font-size="18" fill="#475569">RLHF后：更符合人类偏好</text>
          <text x="30" y="40" font-size="15" fill="#64748b">
            示例：回答更友好、更安全、更有条理
          </text>
        </g>
      </g>

      <!-- 底部注释 -->
      <g transform="translate(100, 700)">
        <text x="0" y="0" font-size="16" fill="#64748b">
          代表模型：ChatGPT = GPT-3.5(预训练) + SFT + RLHF
        </text>
      </g>
    </svg>
  `,
};

async function generateDiagrams() {
  console.log('🎨 Generating diagram images...\n');

  for (const [name, svgContent] of Object.entries(svgs)) {
    const outputPath = path.join(imagesDir, `${name}.png`);
    const svgBuffer = Buffer.from(svgContent);

    await sharp(svgBuffer)
      .png()
      .toFile(outputPath);

    console.log(`✅ Generated: ${name}.png`);
  }

  console.log('\n🎉 Diagram images generated successfully!');
}

generateDiagrams().catch(console.error);
