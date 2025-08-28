import os
import cv2
import numpy as np
from collections import Counter
import math

# Import our powerful, refactored recognition engine
from backend.services import recognition_service

class PlateTracker:
    """
    A class to track a single license plate over multiple frames.
    This implements the "temporal consistency" logic.
    """
    def __init__(self, tracker_id, initial_bbox):
        self.id = tracker_id
        self.bbox = initial_bbox
        self.center_point = self._get_center(initial_bbox)
        
        self.observations = []
        self.frames_since_seen = 0
        self.consensus_text = None
        self.confirmed = False

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bbox, ocr_text):
        """Update the tracker with a new detection."""
        self.bbox = bbox
        self.center_point = self._get_center(bbox)
        self.frames_since_seen = 0
        
        # Only add valid, non-empty OCR readings
        if ocr_text:
            self.observations.append(ocr_text)
            self._update_consensus()

    def _update_consensus(self):
        """Perform a majority vote to get the most likely plate number."""
        # We need a minimum number of observations to be confident
        if len(self.observations) < 5:
            return

        # Get the most common length of observed plates
        common_length = Counter(len(plate) for plate in self.observations).most_common(1)[0][0]
        
        # Filter for observations of that common length
        valid_observations = [plate for plate in self.observations if len(plate) == common_length]
        
        if not valid_observations:
            return

        # Perform character-by-character majority vote
        consensus_chars = []
        for i in range(common_length):
            char_votes = Counter(plate[i] for plate in valid_observations)
            best_char = char_votes.most_common(1)[0][0]
            consensus_chars.append(best_char)
            
        self.consensus_text = "".join(consensus_chars)
        
        # If our consensus has been stable for a few observations, confirm it
        if self.observations[-3:].count(self.consensus_text) >= 2:
            if not self.confirmed:
                print(f"[CONFIRMED] Plate Tracker #{self.id}: {self.consensus_text}")
                # This is where you would call the API to check-in the vehicle
                # For example: requests.post('http://127.0.0.1:5000/api/checkin', json={'plate_number': self.consensus_text})
                self.confirmed = True


def main():
    # --- Configuration ---
    # Change this to a video file path or a camera index (e.g., 0 for webcam)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_source = os.path.join(script_dir, 'test_video.mp4')

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return

    trackers = {}
    next_tracker_id = 0
    
    # --- Main Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # --- Detection Phase ---
        # We run our powerful recognition service on the current frame
        detections = recognition_service.recognize_plate_from_image(frame)
        
        # --- Tracking Phase ---
        matched_tracker_ids = set()

        # Match new detections to existing trackers
        for det in detections:
            det_center = (int((det['bbox'][0] + det['bbox'][2]) / 2), int((det['bbox'][1] + det['bbox'][3]) / 2))
            
            best_match_id = None
            min_distance = 100 # Max distance to be considered a match

            for tracker_id, tracker in trackers.items():
                distance = math.sqrt((tracker.center_point[0] - det_center[0])**2 + (tracker.center_point[1] - det_center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = tracker_id
            
            if best_match_id is not None:
                trackers[best_match_id].update(det['bbox'], det['text'])
                matched_tracker_ids.add(best_match_id)
            else:
                # This is a new, untracked plate
                trackers[next_tracker_id] = PlateTracker(next_tracker_id, det['bbox'])
                trackers[next_tracker_id].update(det['bbox'], det['text'])
                matched_tracker_ids.add(next_tracker_id)
                next_tracker_id += 1
        
        # --- Cleanup Phase ---
        lost_trackers = []
        for tracker_id, tracker in trackers.items():
            if tracker_id not in matched_tracker_ids:
                tracker.frames_since_seen += 1
            
            # If a tracker is lost for too long, remove it
            if tracker.frames_since_seen > 10:
                lost_trackers.append(tracker_id)
        
        for tracker_id in lost_trackers:
            print(f"[LOST] Plate Tracker #{trackers[tracker_id].id} (Last seen text: {trackers[tracker_id].consensus_text})")
            del trackers[tracker_id]

        # --- Visualization ---
        display_frame = frame.copy()
        for tracker_id, tracker in trackers.items():
            color = (0, 255, 255) # Yellow for "tracking"
            if tracker.confirmed:
                color = (0, 255, 0) # Green for "confirmed"
            
            x1, y1, x2, y2 = tracker.bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            
            display_text = f"ID: {tracker.id}"
            if tracker.consensus_text:
                display_text = f"{tracker.consensus_text} (ID: {tracker.id})"
                
            cv2.putText(display_frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        cv2.imshow('Live Feed Processor', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # You will need a test video file in your backend folder named 'test_video.mp4'
    # Or change the video_source to 0 to use your webcam.
    main()