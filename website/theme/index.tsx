import Theme from 'rspress/theme';
import { PlainLanguageExplanation } from './PlainLanguageExplanation';
import { DocumentBadge } from './DocumentBadge';
import { PodcastPlayer } from './PodcastPlayer';
import './index.css';

// 全局音频管理
let globalAudioElements: HTMLAudioElement[] = [];

// 监听路由变化，停止所有音频
const setupAudioCleanup = () => {
  // 停止所有音频的函数
  const stopAllAudio = () => {
    globalAudioElements.forEach(audio => {
      if (!audio.paused) {
        audio.pause();
        audio.currentTime = 0;
      }
    });
    globalAudioElements = [];
  };

  // 监听路由变化
  if (typeof window !== 'undefined') {
    // 使用 popstate 监听浏览器的前进/后退
    window.addEventListener('popstate', stopAllAudio);

    // 监听 pushState 和 replaceState
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;

    history.pushState = function(...args) {
      stopAllAudio();
      return originalPushState.apply(this, args);
    };

    history.replaceState = function(...args) {
      stopAllAudio();
      return originalReplaceState.apply(this, args);
    };
  }
};

const Layout = () => {
  // 只在客户端执行一次
  if (typeof window !== 'undefined' && !window.__audioCleanupSetup) {
    setupAudioCleanup();
    window.__audioCleanupSetup = true;
  }

  return (
    <Theme.Layout
      beforeNavTitle={
        <span style={{ fontSize: '1.5rem', marginRight: '0.5rem' }}>🚀</span>
      }
      beforeDocContent={
        <>
          <DocumentBadge />
          <PodcastPlayer />
          <PlainLanguageExplanation />
        </>
      }
    />
  );
};

// 扩展 Window 接口
declare global {
  interface Window {
    __audioCleanupSetup?: boolean;
  }
}

export default {
  ...Theme,
  Layout,
};

export * from 'rspress/theme';

