import Theme from 'rspress/theme';
import { PlainLanguageExplanation } from './PlainLanguageExplanation';
import './index.css';

const Layout = () => (
  <Theme.Layout
    beforeNavTitle={
      <div className="nav-logo-wrapper">
        <span className="nav-logo-icon">🚀</span>
      </div>
    }
    afterNavMenu={
      <div className="nav-tech-indicator">
        <div className="tech-dot"></div>
        <span className="tech-text">Tech</span>
      </div>
    }
    beforeDocContent={
      <>
        <PlainLanguageExplanation />
        <div className="doc-header-gradient"></div>
      </>
    }
    afterDocContent={
      <div className="doc-footer-gradient"></div>
    }
    top={
      <div className="top-tech-border"></div>
    }
    bottom={
      <div className="bottom-tech-border"></div>
    }
  />
);

export default {
  ...Theme,
  Layout,
};

export * from 'rspress/theme';

