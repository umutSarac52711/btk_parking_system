# backend/live_feed_processor.py
import cv2
import math
import time
import base64
import requests
from threading import Event, Lock
from collections import Counter
from .services import recognition_service

# PlateTracker class remains the same as the previous version.
class PlateTracker:
    # ... (code is identical to the previous 'live_feed_processor.py') ...
    def __init__(self, tracker_id, initial_bbox, initial_ocr):
        self.id = tracker_id; self.bbox = initial_bbox; self.center_point = self._get_center(initial_bbox)
        self.observations = [initial_ocr] if initial_ocr else []; self.frames_since_seen = 0
        self.consensus_text = None; self.has_fired_event = False
    def _get_center(self, bbox): x1, y1, x2, y2 = bbox; return int((x1 + x2) / 2), int((y1 + y2) / 2)
    def update(self, bbox, ocr_text):
        self.bbox = bbox; self.center_point = self._get_center(bbox); self.frames_since_seen = 0
        if ocr_text: self.observations.append(ocr_text); self._update_consensus()
    def _update_consensus(self):
        if len(self.observations) < 5: return
        try:
            common_length = Counter(len(p) for p in self.observations).most_common(1)[0][0]
            valid_obs = [p for p in self.observations if len(p) == common_length]
            if len(valid_obs) < 3: return
            consensus = "".join([Counter(c).most_common(1)[0][0] for c in zip(*valid_obs)])
            self.consensus_text = consensus
            if self.observations.count(self.consensus_text) >= 3 and not self.has_fired_event:
                self.fire_checkin_event()
        except IndexError: pass
    def fire_checkin_event(self):
        self.has_fired_event = True
        plate = self.consensus_text
        print(f"[EVENT] Firing check-in for confirmed plate: {plate}")
        try:
            requests.post('http://127.0.0.1:5000/api/checkin/manual', json={'plate_number': plate})
        except Exception as e:
            print(f"API call exception for {plate}: {e}")

class LiveFeedProcessor:
    """Manages an on-demand video processing session using a multi-threaded producer-consumer model."""
    def __init__(self, video_source, socketio_instance, sid):
        self.video_source = video_source
        self.socketio = socketio_instance
        self.sid = sid
        
        # --- Shared State ---
        self.trackers = {}
        # --- THE FIX: Initialize the tracker ID counter ---
        self.next_tracker_id = 0
        self.last_processed_frame_num = -1
        
        # --- Threading Control ---
        self.lock = Lock()
        self._stop_event = Event()
        self.recognition_thread = None
        self.streaming_thread = None

    def stop(self):
        print(f"[CONTROL] Stop signal received for session: {self.sid}")
        self._stop_event.set()

    def _manage_trackers(self, detections):
        # ... (This method is identical to the previous version) ...
        matched_ids = set()
        for det in detections:
            det_center = (int((det['bbox'][0] + det['bbox'][2]) / 2), int((det['bbox'][1] + det['bbox'][3]) / 2))
            best_match_id, min_dist = None, 150
            for tracker_id, tracker in self.trackers.items():
                if tracker.id in matched_ids: continue
                dist = math.dist(tracker.center_point, det_center)
                if dist < min_dist: min_dist, best_match_id = dist, tracker_id
            if best_match_id is not None:
                self.trackers[best_match_id].update(det['bbox'], det['text']); matched_ids.add(best_match_id)
            else:
                new_tracker = PlateTracker(self.next_tracker_id, det['bbox'], det['text'])
                self.trackers[self.next_tracker_id] = new_tracker; self.next_tracker_id += 1
        
        lost_ids = []
        for tracker_id, tracker in self.trackers.items():
            if tracker_id not in matched_ids:
                tracker.frames_since_seen += 1 # Increment if not seen
            if tracker.frames_since_seen > 20: # Check the new value
                lost_ids.append(tracker_id)
        for tid in lost_ids: del self.trackers[tid]

    def _recognition_loop(self):
        """PRODUCER THREAD: Runs heavy recognition at a slower pace."""
        print(f"[{self.sid}] Recognition thread started.")
        cap = cv2.VideoCapture(self.video_source)
        frame_num = 0
        while not self._stop_event.is_set():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # --- DEBUG LOG ---
            print(f"[{self.sid}] Recognition Loop: Processing frame #{frame_num}")
            
            detections = recognition_service.recognize_plate_from_image(frame)
            
            # Use the lock to safely update the shared trackers dictionary
            with self.lock:
                self._manage_trackers(detections)
                self.last_processed_frame_num = frame_num
                
            frame_num += 1
            self.socketio.sleep(0.1) # Aim for ~10 FPS recognition
        
        cap.release()
        print(f"[{self.sid}] Recognition thread stopped.")

    def _streaming_loop(self):
        """CONSUMER THREAD: Streams annotated frames at a high, consistent FPS."""
        print(f"[{self.sid}] Streaming thread started.")
        cap = cv2.VideoCapture(self.video_source)
        frame_num = 0
        while not self._stop_event.is_set():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # --- DEBUG LOG ---
            # Check if the recognition is keeping up with the stream
            if frame_num % 30 == 0: # Print every second
                with self.lock:
                    rec_frame = self.last_processed_frame_num
                print(f"[{self.sid}] Streaming Loop: On frame #{frame_num}, last processed frame was #{rec_frame}")
            
            # Use the lock to safely read the shared trackers dictionary
            with self.lock:
                # Draw annotations based on the latest data from the other thread
                for tracker in self.trackers.values():
                    color = (0, 255, 0) if tracker.has_fired_event else (0, 255, 255)
                    x1, y1, x2, y2 = tracker.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"ID:{tracker.id}"
                    if tracker.consensus_text: text = f"{tracker.consensus_text} (ID:{tracker.id})"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            self.socketio.emit('video_frame', {'frame': frame_b64}, to=self.sid)
            
            frame_num += 1
            self.socketio.sleep(0.033) # Aim for ~30 FPS streaming
        
        cap.release()
        print(f"[{self.sid}] Streaming thread stopped.")

    def run(self):
        """Starts both the recognition and streaming threads."""
        self.recognition_thread = self.socketio.start_background_task(target=self._recognition_loop)
        self.streaming_thread = self.socketio.start_background_task(target=self._streaming_loop)