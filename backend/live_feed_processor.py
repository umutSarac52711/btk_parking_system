# backend/live_feed_processor.py
import cv2
import math
import time
from collections import Counter
from .services import recognition_service

# --- Helper Function for Robust Matching ---
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --- The new, intelligent PlateTracker class ---
class PlateTracker:
    def __init__(self, tracker_id, initial_bbox):
        self.id = tracker_id; self.bbox = initial_bbox
        self.observations = []; self.consensus_text = None
        self.frames_since_seen = 0; self.confidence_score = 0.0
        self.action_taken = False
    def _calculate_consensus(self):
        if len(self.observations) < 3: return
        try:
            common_length = Counter(len(p) for p in self.observations).most_common(1)[0][0]
            valid_obs = [p for p in self.observations if len(p) == common_length]
            if not valid_obs: return
            consensus = "".join([Counter(c).most_common(1)[0][0] for c in zip(*valid_obs)])
            self.consensus_text = consensus
            last_n = self.observations[-5:]
            self.confidence_score = last_n.count(self.consensus_text) / len(last_n)
        except IndexError:
            self.consensus_text = None; self.confidence_score = 0.0
    def update(self, new_bbox, ocr_reading):
        self.bbox = new_bbox; self.frames_since_seen = 0
        if ocr_reading and len(ocr_reading) > 4: self.observations.append(ocr_reading)
        self._calculate_consensus()

class LiveFeedProcessor:
    def __init__(self, video_source, socketio_instance, app_context):
        self.video_source = video_source
        self.socketio = socketio_instance
        self.app_context = app_context
        self.active_trackers = {}
        self.next_tracker_id = 0
        self.cap = None # Will be initialized in the background task
        print(f"[PROCESSOR_INIT] LiveFeedProcessor created for source: '{self.video_source}'")

    def _initialize_capture(self):
        """Initializes the video capture object."""
        try:
            print(f"[PROCESSOR_INIT] Attempting to open video source: {self.video_source}")
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                print(f"[ERROR] CRITICAL: Failed to open video source: {self.video_source}")
                self.cap = None
            else:
                print(f"[SUCCESS] Video source opened successfully.")
        except Exception as e:
            print(f"[ERROR] CRITICAL: An exception occurred while opening video source: {e}")
            self.cap = None

    def _process_frame(self, frame):
        detections = recognition_service.recognize_plate_from_image(frame)
        matched_tracker_ids = set()
        if not self.active_trackers:
            for det in detections:
                new_tracker = PlateTracker(self.next_tracker_id, det['bbox'])
                new_tracker.update(det['bbox'], det['text'])
                self.active_trackers[self.next_tracker_id] = new_tracker
                self.next_tracker_id += 1
        else:
            unmatched_detections = list(range(len(detections)))
            tracker_ids = list(self.active_trackers.keys())
            for tracker_id in tracker_ids:
                tracker = self.active_trackers[tracker_id]
                best_match_iou, best_match_idx = 0, -1
                for i, det_idx in enumerate(unmatched_detections):
                    iou = calculate_iou(tracker.bbox, detections[det_idx]['bbox'])
                    if iou > best_match_iou:
                        best_match_iou, best_match_idx = iou, i
                if best_match_iou > 0.5:
                    det_idx = unmatched_detections.pop(best_match_idx)
                    det = detections[det_idx]
                    tracker.update(det['bbox'], det['text'])
                    matched_tracker_ids.add(tracker_id)
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                new_tracker = PlateTracker(self.next_tracker_id, det['bbox'])
                new_tracker.update(det['bbox'], det['text'])
                self.active_trackers[self.next_tracker_id] = new_tracker
                self.next_tracker_id += 1
        for tracker_id, tracker in self.active_trackers.items():
            if tracker_id not in matched_tracker_ids: tracker.frames_since_seen += 1
        for tracker_id, tracker in self.active_trackers.items():
            if tracker.confidence_score > 0.8 and not tracker.action_taken:
                print(f"[ACTION] Plate Confirmed: {tracker.consensus_text} (ID: {tracker_id})")
                with self.app_context:
                    from . import database
                    database.check_in_vehicle(tracker.consensus_text)
                    self.socketio.emit('plate_confirmed', {'plate_number': tracker.consensus_text})
                tracker.action_taken = True
        stale_trackers = [tid for tid, t in self.active_trackers.items() if t.frames_since_seen > 15]
        for tracker_id in stale_trackers:
            # print(f"[CLEANUP] Removing stale tracker ID: {tracker_id}")
            del self.active_trackers[tracker_id]

    def generate_annotated_feed(self):
        if not self.cap or not self.cap.isOpened():
            print("[FEED_GEN] Cannot generate feed, video capture is not available.")
            return
        while True:
            success, frame = self.cap.read()
            if not success: time.sleep(0.1); continue
            for tracker in self.active_trackers.values():
                x1, y1, x2, y2 = [int(c) for c in tracker.bbox]
                color = (0, 255, 0) if tracker.action_taken else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"ID: {tracker.id}"
                if tracker.consensus_text:
                    conf_percent = int(tracker.confidence_score * 100)
                    text = f"{tracker.consensus_text} ({conf_percent}%)"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            self.socketio.sleep(0.05)

    def run_processor_in_background(self):
        print("[BACKGROUND_TASK] Starting background processor loop...")
        self._initialize_capture()
        if not self.cap:
            print("[BACKGROUND_TASK] Halting due to failed video capture initialization.")
            return
        
        while True:
            try:
                success, frame = self.cap.read()
                if not success:
                    print("[PROCESSOR_LOOP] Video ended. Rewinding.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._process_frame(frame)
                self.socketio.sleep(0.1)
            except Exception as e:
                print(f"[ERROR] An exception occurred in the main processing loop: {e}")
                time.sleep(1) # Prevent rapid-fire error loops