import sharp from 'sharp';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const imagesDir = path.join(__dirname, 'docs/public/assets/images');

// 简洁风格的 SVG 图表
const diagrams = {
  'position-encoding': `
    <svg width="1200" height="400" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="400" fill="#ffffff"/>

      <!-- 输入词 -->
      <text x="150" y="80" font-size="16" fill="#666" text-anchor="middle">Token 0</text>
      <rect x="100" y="100" width="100" height="60" fill="#e3f2fd" stroke="#2196f3" stroke-width="2" rx="4"/>
      <text x="150" y="135" font-size="18" text-anchor="middle" fill="#333">我</text>

      <text x="350" y="80" font-size="16" fill="#666" text-anchor="middle">Token 1</text>
      <rect x="300" y="100" width="100" height="60" fill="#e3f2fd" stroke="#2196f3" stroke-width="2" rx="4"/>
      <text x="350" y="135" font-size="18" text-anchor="middle" fill="#333">爱</text>

      <text x="550" y="80" font-size="16" fill="#666" text-anchor="middle">Token 2</text>
      <rect x="500" y="100" width="100" height="60" fill="#e3f2fd" stroke="#2196f3" stroke-width="2" rx="4"/>
      <text x="550" y="135" font-size="18" text-anchor="middle" fill="#333">你</text>

      <!-- 加号 -->
      <text x="150" y="210" font-size="24" text-anchor="middle" fill="#666">+</text>
      <text x="350" y="210" font-size="24" text-anchor="middle" fill="#666">+</text>
      <text x="550" y="210" font-size="24" text-anchor="middle" fill="#666">+</text>

      <!-- 位置编码 -->
      <text x="150" y="260" font-size="16" fill="#666" text-anchor="middle">Pos 0</text>
      <rect x="100" y="280" width="100" height="60" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2" rx="4"/>
      <text x="150" y="315" font-size="14" text-anchor="middle" fill="#333">sin/cos(0)</text>

      <text x="350" y="260" font-size="16" fill="#666" text-anchor="middle">Pos 1</text>
      <rect x="300" y="280" width="100" height="60" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2" rx="4"/>
      <text x="350" y="315" font-size="14" text-anchor="middle" fill="#333">sin/cos(1)</text>

      <text x="550" y="260" font-size="16" fill="#666" text-anchor="middle">Pos 2</text>
      <rect x="500" y="280" width="100" height="60" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2" rx="4"/>
      <text x="550" y="315" font-size="14" text-anchor="middle" fill="#333">sin/cos(2)</text>

      <!-- 说明 -->
      <g transform="translate(750, 150)">
        <text x="0" y="0" font-size="20" font-weight="bold" fill="#333">位置编码方法</text>
        <text x="0" y="40" font-size="16" fill="#666">• 正弦余弦编码</text>
        <text x="0" y="70" font-size="16" fill="#666">• 可学习编码</text>
        <text x="0" y="100" font-size="16" fill="#666">• RoPE 旋转编码</text>
      </g>
    </svg>
  `,

  'llm-training-pipeline': `
    <svg width="1200" height="400" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="400" fill="#ffffff"/>

      <!-- 阶段1 -->
      <rect x="50" y="100" width="300" height="200" fill="#e3f2fd" stroke="#2196f3" stroke-width="2" rx="8"/>
      <text x="200" y="140" font-size="20" font-weight="bold" text-anchor="middle" fill="#1565c0">预训练</text>
      <text x="200" y="170" font-size="14" text-anchor="middle" fill="#666">数据: 数万亿token</text>
      <text x="200" y="195" font-size="14" text-anchor="middle" fill="#666">目标: 学习语言</text>
      <text x="200" y="270" font-size="16" font-weight="bold" text-anchor="middle" fill="#1565c0">GPT-3</text>

      <!-- 箭头1 -->
      <path d="M 360 200 L 440 200" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow1)"/>
      <defs>
        <marker id="arrow1" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#666"/>
        </marker>
      </defs>

      <!-- 阶段2 -->
      <rect x="450" y="100" width="300" height="200" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2" rx="8"/>
      <text x="600" y="140" font-size="20" font-weight="bold" text-anchor="middle" fill="#6a1b9a">监督微调(SFT)</text>
      <text x="600" y="170" font-size="14" text-anchor="middle" fill="#666">数据: 数万条问答</text>
      <text x="600" y="195" font-size="14" text-anchor="middle" fill="#666">目标: 遵循指令</text>
      <text x="600" y="270" font-size="16" font-weight="bold" text-anchor="middle" fill="#6a1b9a">InstructGPT</text>

      <!-- 箭头2 -->
      <path d="M 760 200 L 840 200" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrow2)"/>
      <defs>
        <marker id="arrow2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
          <polygon points="0 0, 10 3, 0 6" fill="#666"/>
        </marker>
      </defs>

      <!-- 阶段3 -->
      <rect x="850" y="100" width="300" height="200" fill="#e8f5e9" stroke="#4caf50" stroke-width="2" rx="8"/>
      <text x="1000" y="140" font-size="20" font-weight="bold" text-anchor="middle" fill="#2e7d32">RLHF</text>
      <text x="1000" y="170" font-size="14" text-anchor="middle" fill="#666">数据: 人类反馈</text>
      <text x="1000" y="195" font-size="14" text-anchor="middle" fill="#666">目标: 符合偏好</text>
      <text x="1000" y="270" font-size="16" font-weight="bold" text-anchor="middle" fill="#2e7d32">ChatGPT</text>

      <!-- 时间轴 -->
      <text x="200" y="50" font-size="14" fill="#999">数月</text>
      <text x="600" y="50" font-size="14" fill="#999">数周</text>
      <text x="1000" y="50" font-size="14" fill="#999">数周</text>
    </svg>
  `,

  'context-window-comparison': `
    <svg width="1200" height="400" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="400" fill="#ffffff"/>

      <text x="600" y="40" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">上下文窗口对比</text>

      <!-- GPT-3.5 -->
      <rect x="100" y="100" width="150" height="200" fill="#ffebee" stroke="#f44336" stroke-width="2" rx="4"/>
      <text x="175" y="130" font-size="16" font-weight="bold" text-anchor="middle" fill="#c62828">GPT-3.5</text>
      <text x="175" y="160" font-size="18" text-anchor="middle" fill="#333">4K</text>
      <text x="175" y="280" font-size="14" text-anchor="middle" fill="#666">≈3000字</text>

      <!-- GPT-4 -->
      <rect x="300" y="100" width="200" height="200" fill="#fff3e0" stroke="#ff9800" stroke-width="2" rx="4"/>
      <text x="400" y="130" font-size="16" font-weight="bold" text-anchor="middle" fill="#e65100">GPT-4</text>
      <text x="400" y="160" font-size="18" text-anchor="middle" fill="#333">32K</text>
      <text x="400" y="280" font-size="14" text-anchor="middle" fill="#666">≈2.4万字</text>

      <!-- Claude 3 -->
      <rect x="550" y="70" width="250" height="230" fill="#e8eaf6" stroke="#3f51b5" stroke-width="2" rx="4"/>
      <text x="675" y="100" font-size="16" font-weight="bold" text-anchor="middle" fill="#1a237e">Claude 3</text>
      <text x="675" y="130" font-size="18" text-anchor="middle" fill="#333">200K</text>
      <text x="675" y="280" font-size="14" text-anchor="middle" fill="#666">≈15万字</text>

      <!-- Gemini 1.5 -->
      <rect x="850" y="40" width="300" height="260" fill="#e0f2f1" stroke="#009688" stroke-width="2" rx="4"/>
      <text x="1000" y="70" font-size="16" font-weight="bold" text-anchor="middle" fill="#004d40">Gemini 1.5</text>
      <text x="1000" y="100" font-size="18" text-anchor="middle" fill="#333">1M</text>
      <text x="1000" y="280" font-size="14" text-anchor="middle" fill="#666">≈75万字</text>

      <!-- 说明 -->
      <text x="600" y="360" font-size="14" text-anchor="middle" fill="#999">
        上下文窗口越大，能处理的文本越长
      </text>
    </svg>
  `,

  'scaling-law': `
    <svg width="1200" height="400" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="400" fill="#ffffff"/>

      <text x="600" y="40" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">Scaling Law: 性能 vs 模型规模</text>

      <!-- 坐标轴 -->
      <line x1="100" y1="350" x2="1100" y2="350" stroke="#ddd" stroke-width="2"/>
      <line x1="100" y1="350" x2="100" y2="80" stroke="#ddd" stroke-width="2"/>

      <!-- X轴标签 -->
      <text x="600" y="380" font-size="14" text-anchor="middle" fill="#666">模型参数量</text>
      <text x="200" y="370" font-size="12" fill="#999">1B</text>
      <text x="400" y="370" font-size="12" fill="#999">10B</text>
      <text x="600" y="370" font-size="12" fill="#999">100B</text>
      <text x="800" y="370" font-size="12" fill="#999">1T</text>

      <!-- Y轴标签 -->
      <text x="40" y="220" font-size="14" text-anchor="middle" fill="#666" transform="rotate(-90, 40, 220)">性能</text>
      <text x="85" y="350" font-size="12" fill="#999">低</text>
      <text x="85" y="100" font-size="12" fill="#999">高</text>

      <!-- 曲线 -->
      <path d="M 200 330 Q 400 250, 600 180 T 1000 100" stroke="#2196f3" stroke-width="4" fill="none"/>

      <!-- 数据点 -->
      <circle cx="200" cy="330" r="6" fill="#2196f3"/>
      <text x="200" y="315" font-size="12" text-anchor="middle" fill="#666">GPT-2</text>

      <circle cx="400" cy="250" r="6" fill="#2196f3"/>
      <text x="400" y="235" font-size="12" text-anchor="middle" fill="#666">GPT-3</text>

      <circle cx="600" cy="180" r="6" fill="#2196f3"/>
      <text x="600" y="165" font-size="12" text-anchor="middle" fill="#666">LLaMA</text>

      <circle cx="800" cy="130" r="6" fill="#2196f3"/>
      <text x="800" y="115" font-size="12" text-anchor="middle" fill="#666">GPT-4</text>

      <!-- 说明 -->
      <rect x="900" y="200" width="250" height="100" fill="#f5f5f5" stroke="#ddd" stroke-width="1" rx="4"/>
      <text x="1025" y="230" font-size="14" font-weight="bold" text-anchor="middle" fill="#333">Scaling Law</text>
      <text x="1025" y="255" font-size="12" text-anchor="middle" fill="#666">参数量 ↑ → 性能 ↑</text>
      <text x="1025" y="280" font-size="12" text-anchor="middle" fill="#666">但收益递减</text>
    </svg>
  `,

  'temperature-sampling': `
    <svg width="1200" height="400" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="400" fill="#ffffff"/>

      <text x="600" y="40" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">Temperature 采样对比</text>

      <!-- Temperature = 0 -->
      <g transform="translate(100, 100)">
        <text x="100" y="0" font-size="16" font-weight="bold" text-anchor="middle" fill="#1565c0">Temperature = 0</text>
        <rect x="0" y="20" width="200" height="40" fill="#1976d2" rx="4"/>
        <text x="100" y="45" font-size="14" text-anchor="middle" fill="white">确定性 (100%)</text>
        <text x="100" y="90" font-size="12" text-anchor="middle" fill="#666">每次输出相同</text>
      </g>

      <!-- Temperature = 0.7 -->
      <g transform="translate(400, 100)">
        <text x="100" y="0" font-size="16" font-weight="bold" text-anchor="middle" fill="#7b1fa2">Temperature = 0.7</text>
        <rect x="0" y="20" width="150" height="40" fill="#9c27b0" rx="4"/>
        <rect x="155" y="20" width="30" height="40" fill="#ba68c8" rx="4"/>
        <rect x="190" y="20" width="10" height="40" fill="#ce93d8" rx="4"/>
        <text x="100" y="45" font-size="14" text-anchor="middle" fill="white">平衡</text>
        <text x="100" y="90" font-size="12" text-anchor="middle" fill="#666">推荐用于对话</text>
      </g>

      <!-- Temperature = 1.5 -->
      <g transform="translate(800, 100)">
        <text x="100" y="0" font-size="16" font-weight="bold" text-anchor="middle" fill="#c62828">Temperature = 1.5</text>
        <rect x="0" y="20" width="60" height="40" fill="#f44336" rx="4"/>
        <rect x="65" y="20" width="50" height="40" fill="#ef5350" rx="4"/>
        <rect x="120" y="20" width="40" height="40" fill="#e57373" rx="4"/>
        <rect x="165" y="20" width="35" height="40" fill="#ef9a9a" rx="4"/>
        <text x="100" y="45" font-size="14" text-anchor="middle" fill="white">随机</text>
        <text x="100" y="90" font-size="12" text-anchor="middle" fill="#666">创意写作</text>
      </g>

      <!-- 概率分布示意 -->
      <g transform="translate(100, 220)">
        <text x="500" y="0" font-size="18" font-weight="bold" text-anchor="middle" fill="#333">概率分布变化</text>

        <!-- 低温 -->
        <rect x="50" y="30" width="20" height="100" fill="#1976d2"/>
        <rect x="75" y="80" width="20" height="50" fill="#42a5f5"/>
        <rect x="100" y="110" width="20" height="20" fill="#90caf9"/>
        <text x="85" y="150" font-size="12" text-anchor="middle" fill="#666">T=0.3</text>

        <!-- 中温 -->
        <rect x="400" y="50" width="20" height="80" fill="#9c27b0"/>
        <rect x="425" y="70" width="20" height="60" fill="#ab47bc"/>
        <rect x="450" y="90" width="20" height="40" fill="#ba68c8"/>
        <text x="435" y="150" font-size="12" text-anchor="middle" fill="#666">T=0.7</text>

        <!-- 高温 -->
        <rect x="750" y="70" width="20" height="60" fill="#f44336"/>
        <rect x="775" y="75" width="20" height="55" fill="#ef5350"/>
        <rect x="800" y="80" width="20" height="50" fill="#e57373"/>
        <rect x="825" y="90" width="20" height="40" fill="#ef9a9a"/>
        <text x="790" y="150" font-size="12" text-anchor="middle" fill="#666">T=1.5</text>
      </g>
    </svg>
  `,

  'few-shot-zero-shot': `
    <svg width="1200" height="400" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="400" fill="#ffffff"/>

      <text x="600" y="40" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">Few-shot vs Zero-shot</text>

      <!-- Zero-shot -->
      <g transform="translate(100, 100)">
        <rect x="0" y="0" width="450" height="250" fill="#e3f2fd" stroke="#2196f3" stroke-width="2" rx="8"/>
        <text x="225" y="35" font-size="20" font-weight="bold" text-anchor="middle" fill="#1565c0">Zero-shot</text>

        <rect x="20" y="60" width="410" height="60" fill="white" stroke="#90caf9" stroke-width="1" rx="4"/>
        <text x="30" y="85" font-size="14" fill="#666">问题: 翻译成英文：你好</text>
        <text x="30" y="105" font-size="14" fill="#333">AI: Hello</text>

        <text x="225" y="150" font-size="14" text-anchor="middle" fill="#666">✓ 无需示例</text>
        <text x="225" y="175" font-size="14" text-anchor="middle" fill="#666">✓ 速度快</text>
        <text x="225" y="200" font-size="14" text-anchor="middle" fill="#999">✗ 准确性较低</text>
      </g>

      <!-- Few-shot -->
      <g transform="translate(650, 100)">
        <rect x="0" y="0" width="450" height="250" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2" rx="8"/>
        <text x="225" y="35" font-size="20" font-weight="bold" text-anchor="middle" fill="#6a1b9a">Few-shot</text>

        <rect x="20" y="60" width="410" height="120" fill="white" stroke="#ce93d8" stroke-width="1" rx="4"/>
        <text x="30" y="80" font-size="13" fill="#999">示例1: 苹果 → Apple</text>
        <text x="30" y="100" font-size="13" fill="#999">示例2: 香蕉 → Banana</text>
        <text x="30" y="125" font-size="14" fill="#666">问题: 橙子 → ?</text>
        <text x="30" y="150" font-size="14" fill="#333">AI: Orange</text>

        <text x="225" y="205" font-size="14" text-anchor="middle" fill="#666">✓ 准确性高</text>
        <text x="225" y="230" font-size="14" text-anchor="middle" fill="#999">✗ 需要示例</text>
      </g>
    </svg>
  `,

  'function-calling': `
    <svg width="1200" height="500" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="500" fill="#ffffff"/>

      <text x="600" y="40" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">Function Calling 流程</text>

      <!-- 用户 -->
      <rect x="100" y="100" width="200" height="80" fill="#e3f2fd" stroke="#2196f3" stroke-width="2" rx="8"/>
      <text x="200" y="130" font-size="16" font-weight="bold" text-anchor="middle" fill="#1565c0">用户</text>
      <text x="200" y="155" font-size="14" text-anchor="middle" fill="#666">北京今天</text>
      <text x="200" y="175" font-size="14" text-anchor="middle" fill="#666">天气怎么样？</text>

      <!-- 箭头1 -->
      <path d="M 310 140 L 390 140" stroke="#666" stroke-width="2" marker-end="url(#a1)"/>
      <defs><marker id="a1" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><polygon points="0 0, 10 3, 0 6" fill="#666"/></marker></defs>

      <!-- AI分析 -->
      <rect x="400" y="100" width="200" height="80" fill="#fff3e0" stroke="#ff9800" stroke-width="2" rx="8"/>
      <text x="500" y="130" font-size="16" font-weight="bold" text-anchor="middle" fill="#e65100">AI分析</text>
      <text x="500" y="155" font-size="13" text-anchor="middle" fill="#666">需要调用</text>
      <text x="500" y="175" font-size="13" text-anchor="middle" fill="#666">get_weather()</text>

      <!-- 箭头2 -->
      <path d="M 500 180 L 500 250" stroke="#666" stroke-width="2" marker-end="url(#a2)"/>
      <defs><marker id="a2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><polygon points="0 0, 10 3, 0 6" fill="#666"/></marker></defs>

      <!-- 执行函数 -->
      <rect x="400" y="260" width="200" height="80" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2" rx="8"/>
      <text x="500" y="290" font-size="16" font-weight="bold" text-anchor="middle" fill="#6a1b9a">执行函数</text>
      <text x="500" y="315" font-size="13" text-anchor="middle" fill="#666">调用天气API</text>
      <text x="500" y="335" font-size="13" text-anchor="middle" fill="#333">返回: 晴,15°C</text>

      <!-- 箭头3 -->
      <path d="M 610 300 L 690 300" stroke="#666" stroke-width="2" marker-end="url(#a3)"/>
      <defs><marker id="a3" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><polygon points="0 0, 10 3, 0 6" fill="#666"/></marker></defs>

      <!-- 生成回复 -->
      <rect x="700" y="260" width="200" height="80" fill="#e8f5e9" stroke="#4caf50" stroke-width="2" rx="8"/>
      <text x="800" y="290" font-size="16" font-weight="bold" text-anchor="middle" fill="#2e7d32">生成回复</text>
      <text x="800" y="315" font-size="13" text-anchor="middle" fill="#666">北京今天晴</text>
      <text x="800" y="335" font-size="13" text-anchor="middle" fill="#666">温度15度</text>

      <!-- 箭头4 回到用户 -->
      <path d="M 800 260 L 800 220 L 200 220 L 200 180" stroke="#4caf50" stroke-width="2" stroke-dasharray="5,5" marker-end="url(#a4)"/>
      <defs><marker id="a4" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><polygon points="0 0, 10 3, 0 6" fill="#4caf50"/></marker></defs>

      <!-- 说明 -->
      <g transform="translate(100, 400)">
        <text x="0" y="0" font-size="14" fill="#666">核心优势: AI可以调用真实API获取数据,而非编造答案</text>
      </g>
    </svg>
  `,

  'chain-of-thought': `
    <svg width="1200" height="400" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="400" fill="#ffffff"/>

      <text x="600" y="40" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">Chain of Thought (思维链)</text>

      <!-- 普通回答 -->
      <g transform="translate(100, 100)">
        <rect x="0" y="0" width="450" height="120" fill="#ffebee" stroke="#f44336" stroke-width="2" rx="8"/>
        <text x="225" y="30" font-size="18" font-weight="bold" text-anchor="middle" fill="#c62828">普通回答</text>
        <text x="20" y="60" font-size="14" fill="#666">问: 5个苹果15元,8个多少钱?</text>
        <text x="20" y="85" font-size="14" fill="#333">答: 24元</text>
        <text x="225" y="110" font-size="12" text-anchor="middle" fill="#999">✗ 可能出错</text>
      </g>

      <!-- CoT回答 -->
      <g transform="translate(650, 100)">
        <rect x="0" y="0" width="450" height="250" fill="#e8f5e9" stroke="#4caf50" stroke-width="2" rx="8"/>
        <text x="225" y="30" font-size="18" font-weight="bold" text-anchor="middle" fill="#2e7d32">CoT 分步推理</text>
        <text x="20" y="60" font-size="14" fill="#666">问: 5个苹果15元,8个多少钱?</text>
        <text x="20" y="90" font-size="13" fill="#666">步骤1: 1个苹果 = 15÷5 = 3元</text>
        <text x="20" y="115" font-size="13" fill="#666">步骤2: 8个苹果 = 8×3 = 24元</text>
        <text x="20" y="145" font-size="14" font-weight="bold" fill="#333">答: 24元</text>
        <text x="225" y="190" font-size="12" text-anchor="middle" fill="#2e7d32">✓ 准确率提升 3倍+</text>
        <text x="225" y="215" font-size="12" text-anchor="middle" fill="#2e7d32">✓ 可追溯推理过程</text>
        <text x="225" y="240" font-size="12" text-anchor="middle" fill="#2e7d32">✓ 易发现错误</text>
      </g>

      <!-- 提示 -->
      <text x="600" y="380" font-size="14" text-anchor="middle" fill="#666">
        魔法提示词: "Let's think step by step."
      </text>
    </svg>
  `,

  'encoder-decoder': `
    <svg width="1200" height="400" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="400" fill="#ffffff"/>

      <text x="600" y="40" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">Encoder-Decoder 架构</text>

      <!-- Encoder -->
      <g transform="translate(100, 100)">
        <rect x="0" y="0" width="350" height="250" fill="#e3f2fd" stroke="#2196f3" stroke-width="3" rx="8"/>
        <text x="175" y="35" font-size="20" font-weight="bold" text-anchor="middle" fill="#1565c0">Encoder (编码器)</text>

        <text x="30" y="70" font-size="14" fill="#666">输入: "我爱你"</text>
        <rect x="30" y="85" width="80" height="35" fill="white" stroke="#90caf9" stroke-width="1" rx="4"/>
        <text x="70" y="107" font-size="13" text-anchor="middle" fill="#333">我</text>
        <rect x="120" y="85" width="80" height="35" fill="white" stroke="#90caf9" stroke-width="1" rx="4"/>
        <text x="160" y="107" font-size="13" text-anchor="middle" fill="#333">爱</text>
        <rect x="210" y="85" width="80" height="35" fill="white" stroke="#90caf9" stroke-width="1" rx="4"/>
        <text x="250" y="107" font-size="13" text-anchor="middle" fill="#333">你</text>

        <text x="175" y="150" font-size="14" text-anchor="middle" fill="#1565c0">↓ Self-Attention</text>

        <rect x="60" y="170" width="230" height="40" fill="#90caf9" stroke="#1565c0" stroke-width="2" rx="4"/>
        <text x="175" y="195" font-size="13" text-anchor="middle" fill="white">上下文向量 [0.2, -0.5, 0.8...]</text>

        <text x="175" y="235" font-size="12" text-anchor="middle" fill="#666">双向理解,压缩语义</text>
      </g>

      <!-- 箭头 -->
      <path d="M 460 225 L 640 225" stroke="#666" stroke-width="3" marker-end="url(#arrow)"/>
      <defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="3" orient="auto"><polygon points="0 0, 12 3, 0 6" fill="#666"/></marker></defs>
      <text x="550" y="215" font-size="14" text-anchor="middle" fill="#666">传递</text>

      <!-- Decoder -->
      <g transform="translate(650, 100)">
        <rect x="0" y="0" width="450" height="250" fill="#f3e5f5" stroke="#9c27b0" stroke-width="3" rx="8"/>
        <text x="225" y="35" font-size="20" font-weight="bold" text-anchor="middle" fill="#6a1b9a">Decoder (解码器)</text>

        <text x="30" y="70" font-size="14" fill="#666">生成输出 (逐词):</text>

        <rect x="30" y="85" width="80" height="35" fill="white" stroke="#ce93d8" stroke-width="1" rx="4"/>
        <text x="70" y="107" font-size="13" text-anchor="middle" fill="#333">I</text>
        <text x="70" y="130" font-size="11" text-anchor="middle" fill="#999">步骤1</text>

        <rect x="120" y="85" width="80" height="35" fill="white" stroke="#ce93d8" stroke-width="1" rx="4"/>
        <text x="160" y="107" font-size="13" text-anchor="middle" fill="#333">love</text>
        <text x="160" y="130" font-size="11" text-anchor="middle" fill="#999">步骤2</text>

        <rect x="210" y="85" width="80" height="35" fill="white" stroke="#ce93d8" stroke-width="1" rx="4"/>
        <text x="250" y="107" font-size="13" text-anchor="middle" fill="#333">you</text>
        <text x="250" y="130" font-size="11" text-anchor="middle" fill="#999">步骤3</text>

        <text x="225" y="165" font-size="14" text-anchor="middle" fill="#6a1b9a">↑ Cross-Attention</text>
        <text x="225" y="185" font-size="12" text-anchor="middle" fill="#666">(看Encoder的理解)</text>

        <text x="225" y="215" font-size="12" text-anchor="middle" fill="#666">单向生成,逐词输出</text>
        <text x="225" y="235" font-size="12" text-anchor="middle" fill="#666">适用: 翻译,摘要,问答</text>
      </g>
    </svg>
  `,
};

async function generateDiagrams() {
  console.log('🎨 Generating simple diagram images...\n');

  for (const [name, svgContent] of Object.entries(diagrams)) {
    const outputPath = path.join(imagesDir, `${name}.png`);
    const svgBuffer = Buffer.from(svgContent);

    await sharp(svgBuffer)
      .png()
      .toFile(outputPath);

    console.log(`✅ Generated: ${name}`);
  }

  console.log('\n🎉 All diagram images generated successfully!');
  console.log('\n📊 Generated diagrams:');
  console.log('  - Position Encoding');
  console.log('  - LLM Training Pipeline');
  console.log('  - Context Window Comparison');
  console.log('  - Scaling Law');
  console.log('  - Temperature Sampling');
  console.log('  - Few-shot vs Zero-shot');
  console.log('  - Function Calling');
  console.log('  - Chain of Thought');
  console.log('  - Encoder-Decoder');
}

generateDiagrams().catch(console.error);
