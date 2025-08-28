// frontend/src/CheckInForm.jsx
import React, { useState } from 'react';
import { checkInVehicleByImage } from './api';
import ManualCheckInForm from './ManualCheckInForm'; // Import the new component

function CheckInForm({ onCheckInSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState(null); // State for the preview image

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setMessage('');
    setAnnotatedImageUrl(null); // Clear previous preview
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setMessage('Please select an image file.');
      return;
    }
    setIsLoading(true);
    setMessage('Processing image...');
    setAnnotatedImageUrl(null);

    try {
      const result = await checkInVehicleByImage(selectedFile);
      setMessage(`Success! Plate: ${result.plate_number} checked in.`);
      setAnnotatedImageUrl(result.annotated_image_url); // Set the URL for the preview
      onCheckInSuccess();
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
      setSelectedFile(null);
      event.target.reset();
    }
  };

  return (
    <div className="card">
      <h2>Check-In Vehicle</h2>
      <div className="check-in-container">
        <div className="check-in-section">
          <h3>By Image Upload</h3>
          <form onSubmit={handleSubmit}>
            <input type="file" accept="image/*" onChange={handleFileChange} disabled={isLoading} />
            <button type="submit" disabled={isLoading || !selectedFile}>
              {isLoading ? 'Processing...' : 'Check-In by Image'}
            </button>
          </form>
          {message && <p className="message">{message}</p>}
        </div>
        <div className="check-in-section">
            <ManualCheckInForm onCheckInSuccess={onCheckInSuccess} />
        </div>
      </div>
      {annotatedImageUrl && (
        <div className="annotated-image-container">
          <h4>Recognition Result:</h4>
          <img src={annotatedImageUrl} alt="Annotated license plate" />
        </div>
      )}
    </div>
  );
}

export default CheckInForm;