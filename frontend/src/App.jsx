import React, { useState, useEffect, useCallback } from 'react';
import { getParkedCars } from './api';
import ParkedCarsList from './ParkedCarsList';
import CheckInForm from './CheckInForm';
import './App.css'; // We will create this file for styling

function App() {
  const [parkedCars, setParkedCars] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAndSetCars = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      const cars = await getParkedCars();
      setParkedCars(cars);
    } catch (err) {
      setError('Failed to load data from the server.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAndSetCars();
  }, [fetchAndSetCars]);

  return (
    <div className="App">
      <header>
        <h1>Smart Parking Lot Dashboard</h1>
      </header>
      <main>
        <CheckInForm onCheckInSuccess={fetchAndSetCars} />
        {error && <p className="error-message">{error}</p>}
        {isLoading ? (
          <p>Loading parked cars...</p>
        ) : (
          <ParkedCarsList cars={parkedCars} onCheckOutSuccess={fetchAndSetCars} />
        )}
      </main>
    </div>
  );
}

export default App;