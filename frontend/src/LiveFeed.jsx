// frontend/src/LiveFeed.jsx
import React, { useState, useEffect } from 'react';
import { uploadVideoFile } from './api';

function LiveFeed({ socket }) {
  const [frameData, setFrameData] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  // --- The state variable is defined here ---
  const [source, setSource] = useState(null);
  const [message, setMessage] = useState('Select a source to begin.');

  useEffect(() => {
    socket.on('video_frame', (data) => {
      setFrameData(`data:image/jpeg;base64,${data.frame}`);
      if (!isStreaming) setIsStreaming(true);
    });

    socket.on('stream_stopped', () => {
      setIsStreaming(false);
      setFrameData(null);
      setMessage('Stream stopped. Select a source to begin.');
    });

    return () => {
      socket.off('video_frame');
      socket.off('stream_stopped');
    };
  }, [socket, isStreaming]);

  const startWebcam = () => {
    setMessage('Requesting webcam stream...');
    // --- FIX #1: Make sure to set the source state ---
    setSource('webcam');
    socket.emit('start_processing', { source: 'webcam' });
  };
  
  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setMessage('Uploading video...');
      const result = await uploadVideoFile(file);
      // --- FIX #2: Make sure to set the source state ---
      setSource('uploaded video');
      socket.emit('start_processing', { source: 'video', filename: result.filename });
      setMessage('Requesting video stream...');
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    }
  };

  const stopStream = () => {
    socket.emit('stop_processing');
  };

  return (
    <div className="card live-feed-card">
      <h2>Live Gate Camera</h2>
      <div className="video-container">
        {isStreaming ? (
          <img src={frameData} alt="Live video feed" />
        ) : (
          <div className="placeholder">{message}</div>
        )}
      </div>
      <div className="controls">
        {!isStreaming ? (
          <>
            <button onClick={startWebcam}>Start Webcam</button>
            <label className="button-style-label">
              Upload Video
              <input type="file" accept="video/*" onChange={handleVideoUpload} />
            </label>
          </>
        ) : (
          <button onClick={stopStream} className="stop-button">Stop Feed</button>
        )}
      </div>
    </div>
  );
}

export default LiveFeed;