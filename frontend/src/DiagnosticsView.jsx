// frontend/src/DiagnosticsView.jsx
import React, { useState } from 'react';
import { setDiagnosticVideoSource } from './api';
import LiveFeed from './LiveFeed';

function DiagnosticsView() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [videoStreamKey, setVideoStreamKey] = useState(Date.now()); // For cache-busting

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setMessage('');
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setMessage('Please select a video file.');
      return;
    }

    setIsLoading(true);
    setMessage('Uploading and processing video...');

    try {
      await setDiagnosticVideoSource(selectedFile);
      setMessage('Video uploaded and processing started.');
      // Update the key to force LiveFeed to refresh
      setVideoStreamKey(Date.now());
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Diagnostics & Testing</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="video/*" onChange={handleFileChange} disabled={isLoading} />
        <button type="submit" disabled={isLoading || !selectedFile}>
          {isLoading ? 'Upload & Process...' : 'Upload & Process'}
        </button>
      </form>
      {message && <p className="message">{message}</p>}
      <LiveFeed key={videoStreamKey} videoStreamKey={videoStreamKey} />
    </div>
  );
}

export default DiagnosticsView;