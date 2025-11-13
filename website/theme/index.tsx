import Theme from 'rspress/theme';
import { PlainLanguageExplanation } from './PlainLanguageExplanation';
import { DocumentBadge } from './DocumentBadge';
import { PodcastPlayer } from './PodcastPlayer';
import './index.css';

const Layout = () => (
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

export default {
  ...Theme,
  Layout,
};

export * from 'rspress/theme';

