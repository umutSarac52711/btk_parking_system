// frontend/src/api.js
const API_BASE_URL = 'http://127.0.0.1:5000/api';

/**
 * Fetches the list of currently parked cars.
 */
// --- FIX: Add the 'export' keyword back ---
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
// --- FIX: Add the 'export' keyword back ---
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

export const uploadVideoFile = async (videoFile) => {
  const formData = new FormData();
  formData.append('video', videoFile);
  
  const response = await fetch(`${API_BASE_URL}/upload_video`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Video upload failed.');
  }
  return response.json();
};