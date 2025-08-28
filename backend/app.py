# backend/app.py
from flask import Flask
from flask_cors import CORS
from flasgger import Swagger
from flask_socketio import SocketIO
from .api_routes import api
from . import database
from .live_feed_processor import LiveFeedProcessor
import os

# --- Configuration ---
# Define video source here. Use 0 for webcam or a path to a video file.
VIDEO_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_video.mp4')

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key' # Change this
CORS(app, resources={r"/api/*": {}, r"/socket.io/*": {}})
Swagger(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Register Blueprints ---
app.register_blueprint(api)

# --- Background Task Management ---
# Create a global instance of our processor
# This is a bit of a shortcut for a demo; in a production app, you'd use a more robust task queue.
api.video_processor = LiveFeedProcessor(VIDEO_SOURCE, socketio, app.app_context())

@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# --- Main Entry Point ---
if __name__ == '__main__':
    database.create_tables()
    # Start the live feed processor in a background thread
    socketio.start_background_task(target=api.video_processor.run_processor_in_background)
    # Run the Flask-SocketIO server
    socketio.run(app, debug=True, port=5000, use_reloader=False)