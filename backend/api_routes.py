# backend/api_routes.py
from flask import Blueprint, jsonify, request, Response, current_app
import os
import cv2
from werkzeug.utils import secure_filename
from . import database
from .services import recognition_service

api = Blueprint('api', __name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ANNOTATED_FOLDER = os.path.join(UPLOAD_FOLDER, 'annotated')
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)


@api.route('/api/checkin/image', methods=['POST'])
def checkin_image():
    if 'image' not in request.files: return jsonify({"error": "No image file"}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # In the provided code, this function returns two values. We'll adjust to that.
    detections = recognition_service.recognize_plate_from_image(filepath)
    
    # We need to draw the results on the image to save the annotated version.
    # This logic was slightly different in the original file, we'll adapt.
    image_bgr = cv2.imread(filepath)
    best_detection = None
    if detections:
        best_detection = max(detections, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = [int(c) for c in best_detection['bbox']]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, best_detection['text'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if not best_detection:
        os.remove(filepath)
        return jsonify({"error": "Could not recognize a license plate"}), 400

    # Save the annotated image and create a URL for it
    annotated_filename = f"annotated_{filename}"
    annotated_filepath = os.path.join(ANNOTATED_FOLDER, annotated_filename)
    cv2.imwrite(annotated_filepath, image_bgr)
    annotated_url = request.host_url + f'api/uploads/annotated/{annotated_filename}'

    os.remove(filepath)
    
    plate_number = best_detection['text']
    result = database.check_in_vehicle(plate_number)
    
    if result:
        result['entry_time'] = result['entry_time'].isoformat()
        result['annotated_image_url'] = annotated_url
        result['plate_number'] = plate_number # Add plate number to response
        return jsonify(result), 201
    else:
        return jsonify({"error": "Database operation failed"}), 500

# --- NEW DIAGNOSTICS ENDPOINT ---
@api.route('/api/diagnostics/set_video_source', methods=['POST'])
def set_video_source():
    print("[API_ROUTE] Received request for /api/diagnostics/set_video_source")
    if 'video' not in request.files:
        print("[API_ROUTE] Error: 'video' field not in request.files")
        return jsonify({"error": "No video file provided in the 'video' field"}), 400
    file = request.files['video']
    if file.filename == '':
        print("[API_ROUTE] Error: No file selected.")
        return jsonify({"error": "No selected file"}), 400
    
    filename = "diagnostic_video.mp4"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    print(f"[API_ROUTE] Saving uploaded file to: {filepath}")
    file.save(filepath)
    print(f"[API_ROUTE] File saved. Calling restart_video_processor...")
    
    current_app.restart_video_processor(filepath)
    
    return jsonify({"message": f"Successfully set video source to {file.filename}"}), 200


@api.route('/api/checkin/manual', methods=['POST'])
def checkin_manual():
    data = request.get_json()
    plate_number = data.get('plate_number')
    if not plate_number: return jsonify({"error": "Plate number is required"}), 400
    
    cleaned_plate = "".join(filter(str.isalnum, plate_number)).upper()
    
    result = database.check_in_vehicle(cleaned_plate)
    if result:
        result['entry_time'] = result['entry_time'].isoformat()
        return jsonify(result), 201
    else:
        return jsonify({"error": "Database operation failed"}), 500

from flask import send_from_directory
@api.route('/api/uploads/annotated/<filename>')
def uploaded_file(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

video_processor = None

@api.route('/api/video_feed')
def video_feed():
    # --- THE FIX IS HERE ---
    # Instead of checking the module-level 'video_processor',
    # we check the 'video_processor' attribute on the 'api' Blueprint itself.
    if api.video_processor:
        return Response(api.video_processor.generate_annotated_feed(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # This is the source of the 503 error
        print("[API_ROUTE] WARNING: /api/video_feed requested, but api.video_processor is None.")
        return "Video processor is not running or is currently restarting.", 503

@api.route('/api/parked_cars', methods=['GET'])
def get_parked_cars():
    cars = database.get_currently_parked_vehicles(); [c.update({'entry_time': c['entry_time'].isoformat()}) for c in cars]; return jsonify(cars)

@api.route('/api/history', methods=['GET'])
def get_history():
    history = database.get_parking_history()
    for item in history:
        if item['entry_time']: item['entry_time'] = item['entry_time'].isoformat()
        if item['exit_time']: item['exit_time'] = item['exit_time'].isoformat()
    return jsonify(history)

@api.route('/api/checkout', methods=['POST'])
def checkout():
    data = request.get_json(); plate_number = data.get('plate_number')
    if not plate_number: return jsonify({"error": "Plate number is required"}), 400
    result = database.check_out_vehicle(plate_number)
    if result:
        if result.get('details') and result['details'].get('entry_time'):
             result['details']['entry_time'] = result['details']['entry_time'].isoformat()
        if result.get('details') and result['details'].get('exit_time'):
             result['details']['exit_time'] = result['details']['exit_time'].isoformat()
        return jsonify(result), 200
    else: return jsonify({"error": "Database operation failed"}), 500