from flask import Blueprint, jsonify, request, Response, send_from_directory
import os, cv2
from werkzeug.utils import secure_filename
from . import database
from .services import recognition_service

api = Blueprint('api', __name__)
UPLOAD_FOLDER = 'uploads'
VIDEO_FOLDER = os.path.join(UPLOAD_FOLDER, 'videos')
ANNOTATED_FOLDER = os.path.join(UPLOAD_FOLDER, 'annotated')
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

@api.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify({"error": "No video file"}), 400
    file = request.files['video']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(VIDEO_FOLDER, filename)
    file.save(filepath)
    return jsonify({"message": "Upload successful", "filename": filename})


@api.route('/api/checkin/image', methods=['POST'])
def checkin_image():
    if 'image' not in request.files: return jsonify({"error": "No image file"}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    image_bgr = cv2.imread(filepath)
    annotated_image, best_detection = recognition_service.recognize_plate_from_image(image_bgr)
    
    if not best_detection:
        os.remove(filepath)
        return jsonify({"error": "Could not recognize a license plate"}), 400

    # Save the annotated image and create a URL for it
    annotated_filename = f"annotated_{filename}"
    annotated_filepath = os.path.join(ANNOTATED_FOLDER, annotated_filename)
    cv2.imwrite(annotated_filepath, annotated_image)
    annotated_url = request.host_url + f'api/uploads/annotated/{annotated_filename}'

    os.remove(filepath)
    
    plate_number = best_detection['text']
    result = database.check_in_vehicle(plate_number)
    
    if result:
        result['entry_time'] = result['entry_time'].isoformat()
        result['annotated_image_url'] = annotated_url
        return jsonify(result), 201
    else:
        return jsonify({"error": "Database operation failed"}), 500

@api.route('/api/checkin/manual', methods=['POST'])
def checkin_manual():
    data = request.get_json()
    plate_number = data.get('plate_number')
    if not plate_number: return jsonify({"error": "Plate number is required"}), 400
    
    # Clean and validate the plate number string here if needed
    cleaned_plate = "".join(filter(str.isalnum, plate_number)).upper()
    
    result = database.check_in_vehicle(cleaned_plate)
    if result:
        result['entry_time'] = result['entry_time'].isoformat()
        return jsonify(result), 201
    else:
        return jsonify({"error": "Database operation failed"}), 500

# Route to serve the static annotated images
from flask import send_from_directory
@api.route('/api/uploads/annotated/<filename>')
def uploaded_file(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

# The global instance of our processor will be managed in app.py
video_processor = None

@api.route('/api/video_feed')
def video_feed():
    if video_processor:
        return Response(video_processor.generate_annotated_feed_bytes(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Video processor is not running.", 503


# ... (The other routes: parked_cars, history, checkout are the same) ...
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
        if result.get('details'):
             result['details']['entry_time'] = result['details']['entry_time'].isoformat()
             result['details']['exit_time'] = result['details']['exit_time'].isoformat()
        return jsonify(result), 200
    else: return jsonify({"error": "Database operation failed"}), 500