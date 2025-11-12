import sharp from 'sharp';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const sizes = [16, 32, 48, 64, 128, 256];
const publicDir = path.join(__dirname, 'docs/public');
const svgPath = path.join(publicDir, 'favicon.svg');

async function generateFavicons() {
  console.log('🎨 Generating favicons...\n');

  // 读取 SVG
  const svgBuffer = fs.readFileSync(svgPath);

  // 生成 PNG 文件（各种尺寸）
  for (const size of sizes) {
    const outputPath = path.join(publicDir, `favicon-${size}x${size}.png`);
    await sharp(svgBuffer)
      .resize(size, size)
      .png()
      .toFile(outputPath);
    console.log(`✅ Generated: favicon-${size}x${size}.png`);
  }

  // 生成标准 favicon.png (32x32)
  await sharp(svgBuffer)
    .resize(32, 32)
    .png()
    .toFile(path.join(publicDir, 'favicon.png'));
  console.log('✅ Generated: favicon.png (32x32)');

  // 生成 Apple Touch Icon (180x180)
  await sharp(svgBuffer)
    .resize(180, 180)
    .png()
    .toFile(path.join(publicDir, 'apple-touch-icon.png'));
  console.log('✅ Generated: apple-touch-icon.png (180x180)');

  // 生成 Android Chrome icons
  await sharp(svgBuffer)
    .resize(192, 192)
    .png()
    .toFile(path.join(publicDir, 'android-chrome-192x192.png'));
  console.log('✅ Generated: android-chrome-192x192.png');

  await sharp(svgBuffer)
    .resize(512, 512)
    .png()
    .toFile(path.join(publicDir, 'android-chrome-512x512.png'));
  console.log('✅ Generated: android-chrome-512x512.png');

  console.log('\n🎉 All favicons generated successfully!');
  console.log('\n📝 Next: Update rspress.config.ts to use the favicon');
}

generateFavicons().catch(console.error);
