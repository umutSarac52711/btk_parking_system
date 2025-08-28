import React, { useState } from 'react';
import { checkInVehicle } from './api';

function CheckInForm({ onCheckInSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setMessage('');
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setMessage('Please select an image file.');
      return;
    }

    setIsLoading(true);
    setMessage('Processing image...');

    try {
      const result = await checkInVehicle(selectedFile);
      setMessage(`Success! Plate: ${result.plate_number} checked in.`);
      onCheckInSuccess(); // Tell the parent component to refresh
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
      setSelectedFile(null);
      event.target.reset(); // Clear the file input
    }
  };

  return (
    <div className="card">
      <h2>Check-In Vehicle</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} disabled={isLoading} />
        <button type="submit" disabled={isLoading || !selectedFile}>
          {isLoading ? 'Checking In...' : 'Check-In'}
        </button>
      </form>
      {message && <p className="message">{message}</p>}
    </div>
  );
}

export default CheckInForm;