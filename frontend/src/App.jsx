// frontend/src/App.jsx
import React, { useState, useEffect, useCallback } from 'react';
import io from 'socket.io-client';
import { getParkedCars } from './api';
import ParkedCarsList from './ParkedCarsList';
import CheckInForm from './CheckInForm';
import LiveFeed from './LiveFeed';
import './App.css';

// Initialize the socket connection
const socket = io('http://127.0.0.1:5000');

function App() {
  const [parkedCars, setParkedCars] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastEvent, setLastEvent] = useState(null);

  const [sid, setSid] = useState(null); // Store this client's unique session ID


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
    // Initial fetch
    fetchAndSetCars();

    socket.on('connect', () => {
      console.log('Connected to WebSocket with SID:', socket.id);
      setSid(socket.id); // Save our unique ID from the server
    });

    // Set up WebSocket event listener
    socket.on('plate_confirmed', (data) => {
      console.log('Plate confirmed event received:', data);
      setLastEvent(`Live event: ${data.plate_number} checked in!`);
      // When a plate is confirmed by the live feed, we just refresh the list.
      fetchAndSetCars();
    });

    // Clean up the listener when the component unmounts
    return () => {
      socket.off('plate_confirmed');
    };
  }, [fetchAndSetCars]);

  return (
    <div className="App">
      <header>
        <h1>Smart Parking Lot Dashboard</h1>
        {lastEvent && <p className="event-notification">{lastEvent}</p>}
      </header>
      <main>
        <div className="dashboard-layout">
          <div className="left-column">
            <CheckInForm onCheckInSuccess={fetchAndSetCars} />
            {error && <p className="error-message">{error}</p>}
            {isLoading ? (
              <p>Loading parked cars...</p>
            ) : (
              <ParkedCarsList cars={parkedCars} onCheckOutSuccess={fetchAndSetCars} />
            )}
          </div>
          <div className="right-column">
            {sid && <LiveFeed socket={socket} sid={sid} />}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;