# backend/routes.py
from flask import Blueprint, jsonify, request
import os
from werkzeug.utils import secure_filename
from . import database
from .yolo_approach import recognize_plate

api = Blueprint('api', __name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@api.route('/api/parked_cars', methods=['GET'])
def get_parked_cars():
    """
    Get a list of all currently parked vehicles.
    ---
    responses:
      200:
        description: A list of parked cars.
        schema:
          type: array
          items:
            type: object
            properties:
              plate_number:
                type: string
                example: "2A52718"
              entry_time:
                type: string
                format: date-time
                example: "2025-08-28T10:30:00+00:00"
    """
    cars = database.get_currently_parked_vehicles()
    for car in cars:
        car['entry_time'] = car['entry_time'].isoformat()
    return jsonify(cars)

@api.route('/api/checkin', methods=['POST'])
def checkin():
    """
    Check-in a vehicle by uploading an image of its license plate.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: image
        type: file
        required: true
        description: The image file of the car's license plate.
    responses:
      201:
        description: Check-in was successful.
      400:
        description: Bad request (e.g., no image, plate not recognized).
      500:
        description: Internal server error.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    _, recognized_info = recognize_plate(filepath, debug=False, display_windows=False)
    os.remove(filepath)
    
    if not recognized_info or not recognized_info['text']:
        return jsonify({"error": "Could not recognize a license plate"}), 400

    plate_number = recognized_info['text']
    result = database.check_in_vehicle(plate_number)
    
    if result:
        result['entry_time'] = result['entry_time'].isoformat()
        return jsonify(result), 201
    else:
        return jsonify({"error": "Database operation failed"}), 500

@api.route('/api/checkout', methods=['POST'])
def checkout():
    """
    Check-out a vehicle using its license plate number.
    ---
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            plate_number:
              type: string
              example: "2A52718"
    responses:
      200:
        description: Check-out was successful.
      400:
        description: Bad request (e.g., missing plate number).
      500:
        description: Internal server error.
    """
    data = request.get_json()
    plate_number = data.get('plate_number')
    if not plate_number:
        return jsonify({"error": "Plate number is required"}), 400
        
    result = database.check_out_vehicle(plate_number)
    
    if result:
        if result.get('details'):
             result['details']['entry_time'] = result['details']['entry_time'].isoformat()
             result['details']['exit_time'] = result['details']['exit_time'].isoformat()
        return jsonify(result), 200
    else:
        return jsonify({"error": "Database operation failed"}), 500

# You can document the history endpoint in the same way!
@api.route('/api/history', methods=['GET'])
def get_history():
    """
    Get the complete parking history of all vehicles.
    ---
    responses:
      200:
        description: The complete parking log.
    """
    history = database.get_parking_history()
    for item in history:
        if item['entry_time']: item['entry_time'] = item['entry_time'].isoformat()
        if item['exit_time']: item['exit_time'] = item['exit_time'].isoformat()
    return jsonify(history)