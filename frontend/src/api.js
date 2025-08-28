// frontend/src/api.js
const API_BASE_URL = 'http://127.0.0.1:5000/api';

/**
 * Fetches the list of currently parked cars.
 */
export const getParkedCars = async () => {
  const response = await fetch(`${API_BASE_URL}/parked_cars`);
  if (!response.ok) {
    throw new Error('Failed to fetch parked cars.');
  }
  return response.json();
};

/**
 * Checks in a vehicle by uploading an image file.
 */
export const checkInVehicleByImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch(`${API_BASE_URL}/checkin/image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Image check-in failed.');
  }
  return response.json();
};

/**
 * Checks out a vehicle by its plate number.
 */
export const checkOutVehicle = async (plateNumber) => {
  const response = await fetch(`${API_BASE_URL}/checkout`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ plate_number: plateNumber }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Check-out failed.');
  }
  return response.json();
};

/**
 * Checks in a vehicle by manually providing the plate number.
 */
export const checkInVehicleManually = async (plateNumber) => {
  const response = await fetch(`${API_BASE_URL}/checkin/manual`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plate_number: plateNumber }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Manual check-in failed.');
  }
  return response.json();
};

/**
 * Uploads a video file to set as the new diagnostic video source.
 */
export const setDiagnosticVideoSource = async (videoFile) => {
  const formData = new FormData();
  formData.append('video', videoFile);

  const response = await fetch(`${API_BASE_URL}/diagnostics/set_video_source`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to set diagnostic video source.');
  }
  return response.json();
};

// --- NEW API FUNCTIONS FOR FEED CONTROL ---

/**
 * Sends a request to start the live feed using the default webcam.
 */
export const startWebcamFeed = async () => {
  const response = await fetch(`${API_BASE_URL}/diagnostics/start_webcam`, {
    method: 'POST',
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to start webcam feed.');
  }
  return response.json();
};

/**
 * Sends a request to stop the currently running video feed processor.
 */
export const stopVideoFeed = async () => {
  const response = await fetch(`${API_BASE_URL}/diagnostics/stop_processor`, {
    method: 'POST',
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to stop video feed.');
  }
  return response.json();
};