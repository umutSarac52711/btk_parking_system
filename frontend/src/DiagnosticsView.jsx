// frontend/src/DiagnosticsView.jsx
import React, { useState } from 'react';
import { setDiagnosticVideoSource, startWebcamFeed, stopVideoFeed } from './api';
import LiveFeed from './LiveFeed';
import './DiagnosticsView.css'; // We'll add a little CSS for the buttons

function DiagnosticsView() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [videoStreamKey, setVideoStreamKey] = useState(null); // Start with no key

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setMessage('');
  };

  const handleUploadSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setMessage('Please select a video file.');
      return;
    }
    setIsLoading(true);
    setMessage('Uploading and initializing video...');
    try {
      await setDiagnosticVideoSource(selectedFile);
      setMessage('Processing started. Connecting to feed...');
      setVideoStreamKey(Date.now()); // Update key to force LiveFeed to refresh
    } catch (error) {
      setMessage(`Error: ${error.message}`);
      setVideoStreamKey(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleStartWebcam = async () => {
    setIsLoading(true);
    setMessage('Starting webcam. Connecting to feed...');
    try {
      await startWebcamFeed();
      setMessage('Webcam active. Connecting to feed...');
      setVideoStreamKey(Date.now());
    } catch (error) {
      setMessage(`Error: ${error.message}`);
      setVideoStreamKey(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleStopFeed = async () => {
    setIsLoading(true);
    setMessage('Stopping feed...');
    try {
      await stopVideoFeed();
      setMessage('Feed stopped successfully.');
      setVideoStreamKey(null); // Set key to null to signal stop
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Diagnostics & Testing</h2>
      <div className="diagnostics-controls">
        <form onSubmit={handleUploadSubmit}>
          <input type="file" accept="video/*" onChange={handleFileChange} disabled={isLoading} />
          <button type="submit" disabled={isLoading || !selectedFile}>
            Upload & Process
          </button>
        </form>
        <div className="control-divider">OR</div>
        <div className="button-group">
          <button onClick={handleStartWebcam} disabled={isLoading}>Start Webcam</button>
          <button onClick={handleStopFeed} disabled={isLoading || !videoStreamKey} className="stop-button">Stop Feed</button>
        </div>
      </div>
      {message && <p className="message">{message}</p>}
      
      {/* The LiveFeed component will now only be rendered if a stream is active */}
      {videoStreamKey && <LiveFeed key={videoStreamKey} videoStreamKey={videoStreamKey} />}
    </div>
  );
}

export default DiagnosticsView;