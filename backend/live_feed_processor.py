# backend/live_feed_processor.py
import cv2
import math
import time
import logging
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager
from .services import recognition_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessorConfig:
    """Configuration class for LiveFeedProcessor parameters"""
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.8
    min_observations: int = 3
    max_observations: int = 10
    stale_tracker_threshold: int = 15
    very_old_threshold: int = 30
    min_plate_length: int = 4
    max_trackers: int = 50
    cleanup_interval: int = 100
    processing_sleep: float = 0.1
    streaming_sleep: float = 0.05

# --- Optimized Helper Functions ---
def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union for two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_iou_batch(boxes_a: List[List[float]], boxes_b: List[List[float]]) -> np.ndarray:
    """Vectorized IoU calculation for multiple boxes (performance optimization)"""
    if not boxes_a or not boxes_b:
        return np.array([])
        
    boxes_a = np.array(boxes_a)
    boxes_b = np.array(boxes_b)
    
    # Calculate intersection
    x1 = np.maximum(boxes_a[:, 0][:, None], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1][:, None], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2][:, None], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3][:, None], boxes_b[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    
    union = area_a[:, None] + area_b - intersection
    
    return intersection / union

# --- Optimized PlateTracker Class ---
class PlateTracker:
    """
    Intelligent license plate tracker that builds consensus over multiple observations.
    
    Features:
    - Consensus building for improved OCR accuracy
    - Configurable observation limits for memory management
    - Confidence scoring based on recent observations
    - Action tracking to prevent duplicate processing
    """
    
    def __init__(self, tracker_id: int, initial_bbox: List[float], config: ProcessorConfig):
        self.id = tracker_id
        self.bbox = initial_bbox
        self.observations: List[str] = []
        self.consensus_text: Optional[str] = None
        self.frames_since_seen = 0
        self.confidence_score = 0.0
        self.action_taken = False
        self.config = config
        
        logger.debug(f"Created tracker {tracker_id} with bbox {initial_bbox}")
    
    def _calculate_consensus(self):
        """
        Calculate consensus text from multiple OCR observations.
        
        Algorithm:
        1. Require minimum number of observations
        2. Find most common text length
        3. Filter observations by common length
        4. Build character-wise consensus
        5. Calculate confidence based on recent matches
        """
        if len(self.observations) < self.config.min_observations:
            return
            
        try:
            # Keep only recent observations for better performance and accuracy
            recent_obs = self.observations[-self.config.max_observations:]
            
            if not recent_obs:
                return
                
            # Find most common length among recent observations
            lengths = [len(obs) for obs in recent_obs]
            if not lengths:
                return
                
            # Use Counter for more robust most common length detection
            length_counter = Counter(lengths)
            common_length = length_counter.most_common(1)[0][0]
            
            # Filter observations by common length
            valid_obs = [obs for obs in recent_obs if len(obs) == common_length]
            
            if len(valid_obs) < 2:
                self.consensus_text = None
                self.confidence_score = 0.0
                return
            
            # Build character-wise consensus
            consensus_chars = []
            for pos in range(common_length):
                chars_at_pos = [obs[pos] for obs in valid_obs if pos < len(obs)]
                if chars_at_pos:
                    # Find most common character at this position
                    char_counter = Counter(chars_at_pos)
                    most_common_char = char_counter.most_common(1)[0][0]
                    consensus_chars.append(most_common_char)
            
            self.consensus_text = ''.join(consensus_chars)
            
            # Calculate confidence based on recent exact matches
            recent_matches = sum(1 for obs in recent_obs[-5:] if obs == self.consensus_text)
            self.confidence_score = recent_matches / min(5, len(recent_obs))
            
            logger.debug(f"Tracker {self.id}: consensus='{self.consensus_text}', confidence={self.confidence_score:.2f}")
            
        except (IndexError, ValueError) as e:
            logger.warning(f"Error calculating consensus for tracker {self.id}: {e}")
            self.consensus_text = None
            self.confidence_score = 0.0
    
    def update(self, new_bbox: List[float], ocr_reading: Optional[str]):
        """
        Update tracker with new detection information.
        
        Args:
            new_bbox: New bounding box coordinates [x1, y1, x2, y2]
            ocr_reading: OCR text result from current frame
        """
        self.bbox = new_bbox
        self.frames_since_seen = 0
        
        # Only add valid OCR readings
        if (ocr_reading and 
            len(ocr_reading) >= self.config.min_plate_length and
            ocr_reading.strip()):
            
            self.observations.append(ocr_reading.strip().upper())
            
            # Limit observation history for memory management
            if len(self.observations) > self.config.max_observations:
                self.observations = self.observations[-self.config.max_observations:]
        
        self._calculate_consensus()
    
    def is_high_confidence(self) -> bool:
        """Check if tracker has high confidence consensus"""
        return (self.confidence_score >= self.config.confidence_threshold and 
                self.consensus_text is not None and 
                len(self.consensus_text) >= self.config.min_plate_length)
    
    def should_flush(self) -> bool:
        """Check if tracker should be flushed (submitted) during cleanup"""
        return (not self.action_taken and 
                self.consensus_text is not None and 
                len(self.observations) >= self.config.min_observations and
                len(self.consensus_text) >= self.config.min_plate_length)

class LiveFeedProcessor:
    """
    Real-time license plate detection and tracking processor.
    
    This class handles:
    - Live video feed processing
    - Multi-object tracking of license plates
    - Real-time annotated video streaming
    - Database integration for parking management
    - WebSocket communication for live updates
    
    Features:
    - Configurable processing parameters
    - Robust error handling and recovery
    - Memory management and cleanup
    - Performance optimization with vectorized operations
    """
    
    def __init__(self, video_source, socketio_instance, app_context, config: Optional[ProcessorConfig] = None):
        self.video_source = video_source
        self.socketio = socketio_instance
        self.app_context = app_context
        self.config = config or ProcessorConfig()
        
        # Tracking state
        self.active_trackers: Dict[int, PlateTracker] = {}
        self.next_tracker_id = 0
        
        # Video capture
        self.cap = None
        
        # Performance and error tracking
        self.frame_count = 0
        self.error_count = 0
        self.max_errors = 10
        
        logger.info(f"LiveFeedProcessor created for source: '{self.video_source}'")
        logger.info(f"Configuration: {self.config}")

    @contextmanager
    def error_handler(self, operation_name: str):
        """Context manager for robust error handling"""
        try:
            yield
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in {operation_name}: {e}")
            
            if self.error_count > self.max_errors:
                logger.critical("Too many errors, stopping processor")
                raise
                
            time.sleep(0.1)  # Brief pause on error

    def _initialize_capture(self):
        """Initialize video capture with robust error handling"""
        try:
            logger.info(f"Attempting to open video source: {self.video_source}")
            self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                logger.error(f"CRITICAL: Failed to open video source: {self.video_source}")
                self.cap = None
                return False
            
            # Set buffer size to reduce latency for live feeds
            if isinstance(self.video_source, int):  # Camera index
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            logger.info("Video source opened successfully")
            return True
            
        except Exception as e:
            logger.error(f"CRITICAL: Exception while opening video source: {e}")
            self.cap = None
            return False

    def _cleanup_old_trackers(self):
        """Optimized cleanup of old and stale trackers"""
        if self.frame_count % self.config.cleanup_interval != 0:
            return
            
        initial_count = len(self.active_trackers)
        to_remove = []
        
        for tracker_id, tracker in self.active_trackers.items():
            # Remove very old trackers or stale trackers without consensus
            if (tracker.frames_since_seen > self.config.very_old_threshold or 
                (tracker.frames_since_seen > self.config.stale_tracker_threshold and 
                 not tracker.consensus_text)):
                to_remove.append(tracker_id)
        
        # Remove identified trackers
        for tracker_id in to_remove:
            del self.active_trackers[tracker_id]
        
        # Enforce maximum tracker limit
        if len(self.active_trackers) > self.config.max_trackers:
            # Sort by frames_since_seen (descending) and remove oldest
            sorted_trackers = sorted(
                self.active_trackers.items(), 
                key=lambda x: x[1].frames_since_seen, 
                reverse=True
            )
            
            excess_count = len(self.active_trackers) - self.config.max_trackers
            for tracker_id, _ in sorted_trackers[:excess_count]:
                del self.active_trackers[tracker_id]
        
        removed_count = initial_count - len(self.active_trackers)
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} old trackers. Active: {len(self.active_trackers)}")

    def _match_detections_to_trackers(self, detections: List[Dict]) -> Tuple[Dict[int, int], List[int]]:
        """
        Optimized detection-to-tracker matching using IoU.
        
        Returns:
            Tuple of (tracker_to_detection_map, unmatched_detection_indices)
        """
        if not self.active_trackers or not detections:
            return {}, list(range(len(detections)))
        
        # Use vectorized IoU calculation for better performance
        tracker_boxes = [tracker.bbox for tracker in self.active_trackers.values()]
        detection_boxes = [det['bbox'] for det in detections]
        
        try:
            iou_matrix = calculate_iou_batch(tracker_boxes, detection_boxes)
        except Exception as e:
            logger.warning(f"Batch IoU calculation failed, falling back to individual: {e}")
            # Fallback to individual calculations
            iou_matrix = np.zeros((len(tracker_boxes), len(detection_boxes)))
            for i, tracker_box in enumerate(tracker_boxes):
                for j, det_box in enumerate(detection_boxes):
                    iou_matrix[i, j] = calculate_iou(tracker_box, det_box)
        
        # Greedy matching based on IoU scores
        matches = {}
        unmatched_detections = list(range(len(detections)))
        tracker_ids = list(self.active_trackers.keys())
        
        for i, tracker_id in enumerate(tracker_ids):
            if i >= len(iou_matrix) or len(unmatched_detections) == 0:
                break
                
            # Find best match among unmatched detections
            best_iou = 0
            best_det_idx = -1
            
            for j in unmatched_detections:
                if j < iou_matrix.shape[1] and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_det_idx = j
            
            # Accept match if above threshold
            if best_iou > self.config.iou_threshold:
                matches[tracker_id] = best_det_idx
                unmatched_detections.remove(best_det_idx)
        
        return matches, unmatched_detections

    def _process_frame(self, frame):
        """
        Process a single frame for license plate detection and tracking.
        
        Algorithm:
        1. Detect license plates in current frame
        2. Match detections to existing trackers
        3. Update matched trackers
        4. Create new trackers for unmatched detections
        5. Update tracker states and handle confirmations
        6. Clean up old trackers
        """
        self.frame_count += 1
        
        with self.error_handler("plate detection"):
            detections = recognition_service.recognize_plate_from_image(frame)
        
        if not detections:
            # No detections, just increment frames_since_seen for all trackers
            for tracker in self.active_trackers.values():
                tracker.frames_since_seen += 1
            self._cleanup_old_trackers()
            return
        
        # Match detections to existing trackers
        matches, unmatched_detections = self._match_detections_to_trackers(detections)
        
        # Update matched trackers
        matched_tracker_ids = set()
        for tracker_id, det_idx in matches.items():
            if tracker_id in self.active_trackers and det_idx < len(detections):
                detection = detections[det_idx]
                self.active_trackers[tracker_id].update(detection['bbox'], detection['text'])
                matched_tracker_ids.add(tracker_id)
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_detections:
            if det_idx < len(detections):
                detection = detections[det_idx]
                new_tracker = PlateTracker(self.next_tracker_id, detection['bbox'], self.config)
                new_tracker.update(detection['bbox'], detection['text'])
                self.active_trackers[self.next_tracker_id] = new_tracker
                self.next_tracker_id += 1
        
        # Update frames_since_seen for unmatched trackers
        for tracker_id, tracker in self.active_trackers.items():
            if tracker_id not in matched_tracker_ids:
                tracker.frames_since_seen += 1
        
        # Process high-confidence trackers
        self._process_confirmed_plates()
        
        # Regular cleanup
        self._cleanup_old_trackers()

    def _process_confirmed_plates(self):
        """Process trackers that have reached high confidence threshold"""
        for tracker_id, tracker in self.active_trackers.items():
            if tracker.is_high_confidence() and not tracker.action_taken:
                logger.info(f"Plate Confirmed: {tracker.consensus_text} (ID: {tracker_id})")
                
                with self.error_handler("database operation"):
                    with self.app_context:
                        from . import database
                        database.check_in_vehicle(tracker.consensus_text)
                        self.socketio.emit('plate_confirmed', {
                            'plate_number': tracker.consensus_text,
                            'confidence': tracker.confidence_score,
                            'tracker_id': tracker_id
                        })
                
                tracker.action_taken = True

    def _flush_remaining_trackers(self):
        """
        Submit any high-quality, unconfirmed plates after the video ends.
        
        This method is called when the video stream ends to ensure that
        plates that were detected but didn't reach the full confidence
        threshold are still processed.
        """
        logger.info("Video stream ended. Flushing remaining high-quality trackers...")
        flushed_count = 0
        
        for tracker_id, tracker in self.active_trackers.items():
            if tracker.should_flush():
                logger.info(f"Submitting pending plate: {tracker.consensus_text} (ID: {tracker_id})")
                
                with self.error_handler("flush database operation"):
                    with self.app_context:
                        from . import database
                        database.check_in_vehicle(tracker.consensus_text)
                        self.socketio.emit('plate_confirmed', {
                            'plate_number': tracker.consensus_text,
                            'confidence': tracker.confidence_score,
                            'tracker_id': tracker_id,
                            'flushed': True
                        })
                
                tracker.action_taken = True
                flushed_count += 1
        
        logger.info(f"Flush complete. Submitted {flushed_count} pending plates.")

    def generate_annotated_feed(self):
        """
        Generate real-time annotated video feed with bounding boxes and plate text.
        
        Yields MJPEG frames for HTTP streaming.
        
        Color coding:
        - Green: Confirmed plates (action taken)
        - Yellow: Tracking in progress
        """
        if not self.cap or not self.cap.isOpened():
            logger.error("Cannot generate feed, video capture is not available.")
            return
        
        logger.info("Starting annotated feed generation")
        
        while True:
            with self.error_handler("frame capture"):
                success, frame = self.cap.read()
                
            if not success:
                logger.info("Video source ended or became unavailable.")
                break
            
            # Draw annotations for all active trackers
            for tracker in self.active_trackers.values():
                x1, y1, x2, y2 = [int(c) for c in tracker.bbox]
                
                # Color coding: Green for confirmed, Yellow for tracking
                color = (0, 255, 0) if tracker.action_taken else (0, 255, 255)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare text label
                if tracker.consensus_text:
                    conf_percent = int(tracker.confidence_score * 100)
                    text = f"{tracker.consensus_text} ({conf_percent}%)"
                else:
                    text = f"ID: {tracker.id}"
                
                # Draw text with background for better visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw background rectangle
                cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), color, -1)
                
                # Draw text
                cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
            
            # Encode frame to JPEG
            with self.error_handler("frame encoding"):
                ret, buffer = cv2.imencode('.jpg', frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n' 
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          frame_bytes + b'\r\n')
            
            # Control streaming frame rate
            self.socketio.sleep(self.config.streaming_sleep)

    def run_processor_in_background(self):
        """
        Main background processing loop for license plate detection and tracking.
        
        This method:
        1. Initializes video capture
        2. Processes frames continuously
        3. Handles errors gracefully
        4. Flushes remaining trackers on completion
        """
        logger.info("Starting background processor loop...")
        
        if not self._initialize_capture():
            logger.error("Halting due to failed video capture initialization.")
            return
        
        try:
            while True:
                with self.error_handler("main processing loop"):
                    success, frame = self.cap.read()
                    
                    if not success:
                        logger.info("Video source has ended.")
                        break
                    
                    self._process_frame(frame)
                
                # Control processing frame rate
                self.socketio.sleep(self.config.processing_sleep)
                
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in processing loop: {e}")
        finally:
            # Always flush remaining trackers and cleanup
            self._flush_remaining_trackers()
            
            if self.cap:
                self.cap.release()
                
            logger.info("Background processor loop has finished.")
    
    def get_statistics(self) -> Dict:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary containing processing metrics
        """
        return {
            'active_trackers': len(self.active_trackers),
            'next_tracker_id': self.next_tracker_id,
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'confirmed_plates': sum(1 for t in self.active_trackers.values() if t.action_taken),
            'high_confidence_plates': sum(1 for t in self.active_trackers.values() if t.is_high_confidence())
        }