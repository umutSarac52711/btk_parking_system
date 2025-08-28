// frontend/src/ManualCheckInForm.jsx
import React, { useState } from 'react';
import { checkInVehicleManually } from './api';

function ManualCheckInForm({ onCheckInSuccess }) {
  const [plateNumber, setPlateNumber] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!plateNumber.trim()) {
      setMessage('Please enter a license plate number.');
      return;
    }
    
    try {
      await checkInVehicleManually(plateNumber);
      setMessage(`Success! ${plateNumber} checked in.`);
      setPlateNumber(''); // Clear input on success
      onCheckInSuccess();
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    }
  };

  return (
    <div className="card">
      <h3>Manual Check-In</h3>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={plateNumber}
          onChange={(e) => setPlateNumber(e.target.value.toUpperCase())}
          placeholder="Enter Plate Number"
        />
        <button type="submit">Check-In Manually</button>
      </form>
      {message && <p className="message">{message}</p>}
    </div>
  );
}

export default ManualCheckInForm;