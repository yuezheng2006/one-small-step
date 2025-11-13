import { usePageData } from 'rspress/runtime';
import { useEffect, useRef } from 'react';
import './PodcastPlayer.css';

// 全局管理当前播放的音频
let currentPlayingAudio: HTMLAudioElement | null = null;

export function PodcastPlayer() {
  const { page } = usePageData();
  const frontmatter = (page as any)?.frontmatter || {};
  const podcastUrl = frontmatter.podcastUrl;
  const podcastTitle = frontmatter.podcastTitle || '音频讲解';
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    // 如果有其他音频正在播放，停止它
    if (currentPlayingAudio && currentPlayingAudio !== audio) {
      currentPlayingAudio.pause();
      currentPlayingAudio.currentTime = 0;
    }

    // 设置当前播放的音频
    currentPlayingAudio = audio;

    // 添加事件监听
    const handlePlay = () => {
      // 确保只有当前音频在播放
      document.querySelectorAll('.podcast-audio').forEach(el => {
        const audioEl = el as HTMLAudioElement;
        if (audioEl !== audio && !audioEl.paused) {
          audioEl.pause();
          audioEl.currentTime = 0;
        }
      });
    };

    audio.addEventListener('play', handlePlay);

    // 页面卸载时清理
    return () => {
      audio.removeEventListener('play', handlePlay);
      if (currentPlayingAudio === audio) {
        currentPlayingAudio = null;
      }
      // 不强制暂停，让用户继续收听直到主动切换页面
    };
  }, [podcastUrl]);

  if (!podcastUrl) {
    return null;
  }

  return (
    <div className="podcast-player">
      <audio
        ref={audioRef}
        controls
        className="podcast-audio"
        preload="metadata"
        key={podcastUrl}
      >
        <source src={podcastUrl} type="audio/mpeg" />
        您的浏览器不支持音频播放。
      </audio>
      <div className="podcast-title">AI播客：{podcastTitle}</div>
    </div>
  );
}

