// frontend/src/LiveFeed.jsx
import React, { useState, useEffect } from 'react';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

function LiveFeed({ videoStreamKey }) {
  const [feedStatus, setFeedStatus] = useState('loading'); // 'loading', 'playing', 'stopped'
  
  // Construct the stream URL. It will be re-evaluated whenever the key changes.
  const streamUrl = `${API_BASE_URL}/video_feed?key=${videoStreamKey}`;

  useEffect(() => {
    // Whenever the component is told to render a new stream, reset its status.
    setFeedStatus('loading');
  }, [videoStreamKey]);

  const renderFeedContent = () => {
    switch (feedStatus) {
      case 'loading':
        return (
          <>
            <p>Connecting to video stream...</p>
            {/* Hidden image tag to attempt the connection */}
            <img
              src={streamUrl}
              onLoad={() => setFeedStatus('playing')}
              onError={() => setFeedStatus('stopped')}
              style={{ display: 'none' }}
              alt="Connecting..."
            />
          </>
        );
      case 'playing':
        return (
          <img
            src={streamUrl}
            onError={() => setFeedStatus('stopped')}
            alt="Live video feed from the parking gate"
            width="100%"
          />
        );
      case 'stopped':
        return <p>Feed stopped or video has ended.</p>;
      default:
        return <p>An unknown error occurred with the feed.</p>;
    }
  };

  return (
    <div className="card live-feed-card">
      <h2>Live Gate Camera</h2>
      <div className="video-container">
        {renderFeedContent()}
      </div>
    </div>
  );
}

export default LiveFeed;