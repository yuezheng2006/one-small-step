import { usePageData } from 'rspress/runtime';
import './PodcastPlayer.css';

export function PodcastPlayer() {
  const { page } = usePageData();
  const frontmatter = (page as any)?.frontmatter || {};
  const podcastUrl = frontmatter.podcastUrl;
  const podcastTitle = frontmatter.podcastTitle || '音频讲解';

  if (!podcastUrl) {
    return null;
  }

  return (
    <div className="podcast-player">
      <audio 
        controls 
        className="podcast-audio"
        preload="metadata"
      >
        <source src={podcastUrl} type="audio/mpeg" />
        您的浏览器不支持音频播放。
      </audio>
      <div className="podcast-title">AI播客：{podcastTitle}</div>
    </div>
  );
}

