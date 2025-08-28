# backend/app.py
from flask import Flask
from flask_cors import CORS
from flasgger import Swagger
from flask_socketio import SocketIO
from .api_routes import api
from . import database
from .live_feed_processor import LiveFeedProcessor
import os

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-very-secret-key' # Change this
CORS(app, resources={r"/api/*": {}, r"/socket.io/*": {}})
Swagger(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Register Blueprints ---
app.register_blueprint(api)

# --- Background Task Management ---
# --- MODIFIED --- Start with no processor running.
api.video_processor = None
print("[APP_INIT] Application starting with no video processor active.")


# --- DYNAMIC PROCESSOR RESTART LOGIC ---
def restart_video_processor(new_source_path):
    print(f"--> [HOT-SWAP] Received call to restart processor with source: {new_source_path}")
    print("--> [HOT-SWAP] Step 1: Creating new LiveFeedProcessor instance.")
    new_processor = LiveFeedProcessor(new_source_path, socketio, app.app_context())
    print("--> [HOT-SWAP] Step 2: Overwriting global 'api.video_processor' instance.")
    api.video_processor = new_processor
    print("--> [HOT-SWAP] Step 3: Starting new processor's background task.")
    socketio.start_background_task(target=api.video_processor.run_processor_in_background)
    print("--> [HOT-SWAP] Hot-swap complete. New processor should be running.")

app.restart_video_processor = restart_video_processor


@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# --- Main Entry Point ---
if __name__ == '__main__':
    database.create_tables()
    # --- MODIFIED --- Do NOT start a background task on application startup.
    # It will now be started on-demand via an API call.
    print("[APP_START] Starting Flask-SocketIO server.")
    socketio.run(app, debug=True, port=5000, use_reloader=False)