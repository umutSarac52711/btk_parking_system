# backend/live_feed_processor.py
import cv2
import math
import time
import base64
import requests
from threading import Event
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
    """Manages an on-demand video processing session for a single client."""
    def __init__(self, video_source, socketio_instance, sid):
        self.video_source = video_source
        self.socketio = socketio_instance
        self.sid = sid # Now the processor knows which client it belongs to
        self.trackers = {}
        self.next_tracker_id = 0
        self._stop_event = Event()

    def stop(self):
        print(f"Stopping processor for source: {self.video_source}")
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

    def run_recognition_loop(self):
        """The main processing loop that runs in a background thread until stopped."""
        cap = cv2.VideoCapture(self.video_source)
        while not self._stop_event.is_set():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            
            detections = recognition_service.recognize_plate_from_image(frame)
            self._manage_trackers(detections)
            self.socketio.sleep(0.1)
        
        cap.release()
        print(f"Recognition loop stopped for {self.video_source}")

    def stream_annotated_frames(self):
        """Generates annotated frames and pushes them to the client via WebSocket."""
        cap = cv2.VideoCapture(self.video_source)
        while not self._stop_event.is_set():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

            for tracker in self.trackers.values():
                # ... (drawing logic is the same) ...
                color = (0, 255, 0) if tracker.has_fired_event else (0, 255, 255)
                x1, y1, x2, y2 = tracker.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"ID:{tracker.id}"
                if tracker.consensus_text: text = f"{tracker.consensus_text} (ID:{tracker.id})"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            # Encode the frame as a Base64 string
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit the frame directly to the specific client
            self.socketio.emit('video_frame', {'frame': frame_b64}, to=self.sid)
            self.socketio.sleep(0.03) # ~30 FPS
        
        cap.release()
        print(f"Annotated stream stopped for {self.sid}")

    #def generate_annotated_feed_bytes(self):
    #    """Generator for the MJPEG video stream."""
    #    cap = cv2.VideoCapture(self.video_source)
    #    while not self._stop_event.is_set():
    #        success, frame = cap.read()
    #        if not success:
    #            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
    #
    #        for tracker in self.trackers.values():
    #            color = (0, 255, 0) if tracker.has_fired_event else (0, 255, 255)
    #            x1, y1, x2, y2 = tracker.bbox
    #            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #            text = f"ID:{tracker.id}"
    #            if tracker.consensus_text: text = f"{tracker.consensus_text} (ID:{tracker.id})"
    #            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    #
    #        _, buffer = cv2.imencode('.jpg', frame)
    #        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    #        self.socketio.sleep(0.03)
    #    
    #    cap.release()
    #    print(f"Annotated feed stopped for {self.video_source}")