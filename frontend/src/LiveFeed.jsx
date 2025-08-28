// frontend/src/LiveFeed.jsx
import React from 'react';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

// Accept the videoStreamKey as a prop
function LiveFeed({ videoStreamKey }) {
  // Use the key to create a unique URL that bypasses the browser cache
  const streamUrl = `${API_BASE_URL}/video_feed?key=${videoStreamKey || Date.now()}`;

  return (
    <div className="card live-feed-card">
      <h2>Live Gate Camera</h2>
      <div className="video-container">
        <img
          src={streamUrl}
          alt="Live video feed from the parking gate"
          width="100%"
        />
      </div>
    </div>
  );
}

export default LiveFeed;