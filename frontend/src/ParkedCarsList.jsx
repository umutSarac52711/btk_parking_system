import React from 'react';
import { checkOutVehicle } from './api';

function ParkedCarsList({ cars, onCheckOutSuccess }) {
  const handleCheckOut = async (plateNumber) => {
    if (window.confirm(`Are you sure you want to check out ${plateNumber}?`)) {
      try {
        await checkOutVehicle(plateNumber);
        alert(`${plateNumber} checked out successfully.`);
        onCheckOutSuccess(); // Tell the parent component to refresh
      } catch (error) {
        alert(`Error checking out: ${error.message}`);
      }
    }
  };

  return (
    <div className="card">
      <h2>Currently Parked Vehicles ({cars.length})</h2>
      {cars.length === 0 ? (
        <p>The parking lot is empty.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>License Plate</th>
              <th>Entry Time</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {cars.map((car) => (
              <tr key={car.plate_number}>
                <td>{car.plate_number}</td>
                <td>{new Date(car.entry_time).toLocaleString()}</td>
                <td>
                  <button onClick={() => handleCheckOut(car.plate_number)}>
                    Check-Out
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default ParkedCarsList;