# backend/app.py
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO
from .api_routes import api, VIDEO_FOLDER
from . import database
from .live_feed_processor import LiveFeedProcessor
import os

# ... (App Initialization is the same) ...
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key'
CORS(app, resources={r"/*": {}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
app.register_blueprint(api)
active_processors = {}

@socketio.on('connect')
def handle_connect(): print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    if request.sid in active_processors:
        active_processors[request.sid].stop()
        del active_processors[request.sid]
        print(f"Processor for {request.sid} stopped and cleaned up.")

@socketio.on('start_processing')
def handle_start_processing(data):
    sid = request.sid
    if sid in active_processors: active_processors[sid].stop()

    source_type = data.get('source')
    video_source = 0 if source_type == 'webcam' else os.path.join(VIDEO_FOLDER, data.get('filename', ''))

    if video_source is not None:
        processor = LiveFeedProcessor(video_source, socketio, sid)
        active_processors[sid] = processor
        
        # --- THE FIX: START THE UNIFIED RUN METHOD ---
        processor.run() # This method now manages both internal threads
        
        print(f"Processor started for {sid}")
    else:
        socketio.emit('error', {'message': 'Invalid source'}, to=sid)

@socketio.on('stop_processing')
def handle_stop_processing():
    sid = request.sid
    if sid in active_processors:
        active_processors[sid].stop()
        del active_processors[sid]
        print(f"Processor for {sid} stopped by client request.")
        socketio.emit('stream_stopped', to=sid)

# The /api/video_feed route is GONE and NOT needed.

if __name__ == '__main__':
    database.create_tables()
    socketio.run(app, debug=True, port=5000, use_reloader=False)