# backend/live_feed_processor.py
import cv2
import math
import time
from collections import Counter
from .services import recognition_service

class PlateTracker:
    # ... (This class is exactly the same as the previous version) ...
    def __init__(self, tracker_id, initial_bbox):
        self.id = tracker_id; self.bbox = initial_bbox
        self.center_point = self._get_center(initial_bbox)
        self.observations = []; self.frames_since_seen = 0
        self.consensus_text = None; self.confirmed = False
    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    def update(self, bbox, ocr_text):
        self.bbox = bbox; self.center_point = self._get_center(bbox)
        self.frames_since_seen = 0
        if ocr_text: self.observations.append(ocr_text); self._update_consensus()
    def _update_consensus(self):
        if len(self.observations) < 5: return
        try:
            common_length = Counter(len(plate) for plate in self.observations).most_common(1)[0][0]
            valid_obs = [p for p in self.observations if len(p) == common_length]
            if not valid_obs: return
            consensus = "".join([Counter(chars).most_common(1)[0][0] for chars in zip(*valid_obs)])
            self.consensus_text = consensus
            if self.observations[-3:].count(self.consensus_text) >= 2 and not self.confirmed:
                self.confirmed = True
        except IndexError: pass # Handles cases with no valid observations

class LiveFeedProcessor:
    def __init__(self, video_source, socketio_instance, app_context):
        self.video_source = video_source
        self.socketio = socketio_instance
        self.app_context = app_context
        self.trackers = {}
        self.next_tracker_id = 0

    def _process_frame(self, frame):
        # --- Detection, Tracking, and Cleanup Logic (from previous main loop) ---
        detections = recognition_service.recognize_plate_from_image(frame)
        matched_tracker_ids = set()
        for det in detections:
            det_center = (int((det['bbox'][0] + det['bbox'][2]) / 2), int((det['bbox'][1] + det['bbox'][3]) / 2))
            best_match_id, min_distance = None, 150
            for tracker_id, tracker in self.trackers.items():
                distance = math.dist(tracker.center_point, det_center)
                if distance < min_distance: min_distance, best_match_id = distance, tracker_id
            
            if best_match_id is not None:
                self.trackers[best_match_id].update(det['bbox'], det['text'])
                matched_tracker_ids.add(best_match_id)
            else:
                new_tracker = PlateTracker(self.next_tracker_id, det['bbox'])
                new_tracker.update(det['bbox'], det['text'])
                self.trackers[self.next_tracker_id] = new_tracker
                matched_tracker_ids.add(self.next_tracker_id)
                self.next_tracker_id += 1
        
        lost_trackers = []
        for tracker_id, tracker in self.trackers.items():
            if tracker_id not in matched_tracker_ids: tracker.frames_since_seen += 1
            if tracker.frames_since_seen > 15: lost_trackers.append(tracker_id)
            
            # --- EMIT EVENT ON CONFIRMATION ---
            if tracker.confirmed and tracker.consensus_text:
                print(f"[EVENT] Emitting plate_confirmed: {tracker.consensus_text}")
                with self.app_context: # Use app context for database operations
                    from . import database
                    database.check_in_vehicle(tracker.consensus_text)
                    self.socketio.emit('plate_confirmed', {'plate_number': tracker.consensus_text})
                # Remove tracker after confirmation to prevent re-triggering
                lost_trackers.append(tracker_id)

        for tracker_id in lost_trackers: del self.trackers[tracker_id]

    def generate_annotated_feed(self):
        """Generator for the MJPEG video stream."""
        cap = cv2.VideoCapture(self.video_source)
        while True:
            success, frame = cap.read()
            if not success: time.sleep(0.1); continue
            
            # Draw trackers on the frame for visualization
            for tracker in self.trackers.values():
                color = (0, 255, 0) if tracker.confirmed else (0, 255, 255)
                x1, y1, x2, y2 = tracker.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"ID:{tracker.id}"
                if tracker.consensus_text: text = f"{tracker.consensus_text} (ID:{tracker.id})"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            self.socketio.sleep(0.05) # Yield control to other tasks

    def run_processor_in_background(self):
        """The main loop for processing frames without visualization."""
        cap = cv2.VideoCapture(self.video_source)
        while True:
            success, frame = cap.read()
            if not success:
                # If video ends, reset to the beginning to loop it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self._process_frame(frame)
            self.socketio.sleep(0.1) # Process roughly 10 FPS