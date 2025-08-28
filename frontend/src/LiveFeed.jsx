// frontend/src/LiveFeed.jsx
import React from 'react';

const API_BASE_URL = 'http://127.0.0.1:5000/api';

function LiveFeed() {
  return (
    <div className="card live-feed-card">
      <h2>Live Gate Camera</h2>
      <div className="video-container">
        <img
          src={`${API_BASE_URL}/video_feed`}
          alt="Live video feed from the parking gate"
          width="100%"
        />
      </div>
    </div>
  );
}

export default LiveFeed;