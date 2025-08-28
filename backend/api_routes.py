# backend/api_routes.py
from flask import Blueprint, jsonify, request
import os
import cv2
from werkzeug.utils import secure_filename
from . import database
# Import the newly refactored service
from .services import recognition_service

api = Blueprint('api', __name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@api.route('/api/checkin', methods=['POST'])
def checkin():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Read the image with OpenCV for our service
    image_bgr = cv2.imread(filepath)
    
    # Call the recognition service
    detections = recognition_service.recognize_plate_from_image(image_bgr)
    
    os.remove(filepath)
    
    if not detections:
        return jsonify({"error": "Could not recognize a license plate"}), 400

    # For a static image check-in, we take the most confident detection
    best_detection = max(detections, key=lambda d: d['confidence'])
    plate_number = best_detection['text']
    
    result = database.check_in_vehicle(plate_number)
    
    if result:
        result['entry_time'] = result['entry_time'].isoformat()
        return jsonify(result), 201
    else:
        return jsonify({"error": "Database operation failed"}), 500

# ... (The other routes for parked_cars, history, and checkout are the same) ...
@api.route('/api/parked_cars', methods=['GET'])
def get_parked_cars():
    cars = database.get_currently_parked_vehicles()
    for car in cars: car['entry_time'] = car['entry_time'].isoformat()
    return jsonify(cars)

@api.route('/api/history', methods=['GET'])
def get_history():
    history = database.get_parking_history()
    for item in history:
        if item['entry_time']: item['entry_time'] = item['entry_time'].isoformat()
        if item['exit_time']: item['exit_time'] = item['exit_time'].isoformat()
    return jsonify(history)

@api.route('/api/checkout', methods=['POST'])
def checkout():
    data = request.get_json()
    plate_number = data.get('plate_number')
    if not plate_number: return jsonify({"error": "Plate number is required"}), 400
    result = database.check_out_vehicle(plate_number)
    if result:
        if result.get('details'):
             result['details']['entry_time'] = result['details']['entry_time'].isoformat()
             result['details']['exit_time'] = result['details']['exit_time'].isoformat()
        return jsonify(result), 200
    else: return jsonify({"error": "Database operation failed"}), 500