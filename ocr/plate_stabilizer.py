"""
Temporal Plate Stabilizer and Multi-Frame Voting for Real-Time Video/Camera Streams.
Associates plate detections between consecutive frames using bounding-box IoU,
maintains a sliding window of raw normalized OCR predictions, and performs majority voting
to eliminate single-frame OCR jitter and suppress duplicate terminal logging.
"""

from collections import Counter, deque
import time
from typing import List, Dict, Any, Tuple, Optional


def calculate_box_iou(box1: List[int], box2: List[int]) -> float:
    """Calculates Intersection-over-Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / float(union) if union > 0 else 0.0


class PlateTrack:
    """Represents an active plate track with sliding window OCR history."""

    def __init__(self, track_id: int, bbox: List[int], initial_text: str, confidence: float, timestamp: float, window_size: int = 5):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen_time = timestamp
        self.history: deque = deque(maxlen=window_size)
        if initial_text:
            self.history.append(initial_text)
        self.last_reported_text: Optional[str] = None
        self.last_reported_time: float = 0.0
        self.consecutive_misses: int = 0

    def update(self, bbox: List[int], text: str, confidence: float, timestamp: float):
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen_time = timestamp
        self.consecutive_misses = 0
        if text:
            self.history.append(text)

    def get_voted_result(self, min_votes: int = 3) -> Tuple[str, bool, int, int]:
        """
        Calculates majority vote over the sliding window history.
        Returns:
            voted_text: string with highest frequency (or latest if tied)
            is_stable: True if highest frequency >= min_votes
            votes: count of highest frequency
            total_samples: total valid predictions in history window
        """
        if not self.history:
            return "", False, 0, 0

        counts = Counter(self.history)
        top_text, top_count = counts.most_common(1)[0]
        total_samples = len(self.history)
        is_stable = (top_count >= min_votes)
        return top_text, is_stable, top_count, total_samples


class PlateStabilizer:
    """
    Lightweight Temporal Association and Voting Engine.
    Uses pure BBox IoU and Counter-based majority voting.
    """

    def __init__(
        self,
        vote_window: int = 5,
        min_votes: int = 3,
        iou_threshold: float = 0.40,
        plate_cooldown: float = 3.0,
        max_missed_frames: int = 10,
    ):
        self.vote_window = vote_window
        self.min_votes = min_votes
        self.iou_threshold = iou_threshold
        self.plate_cooldown = plate_cooldown
        self.max_missed_frames = max_missed_frames

        self._next_track_id: int = 1
        self.active_tracks: Dict[int, PlateTrack] = {}

        # Global statistics
        self.total_raw_predictions: int = 0
        self.total_stable_events: int = 0
        self.unstable_transitions: int = 0

    def update(
        self,
        detections: List[Dict[str, Any]],
        current_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Updates active tracks with current detections and returns stabilized detections.
        Each returned dict contains:
            bbox, confidence, layout, raw_text, plate_text, is_stable, votes, window_size,
            track_id, is_new_stable_event
        """
        if current_time is None:
            current_time = time.perf_counter()

        matched_tracks = set()
        matched_det_indices = set()
        stabilized_results = []

        # 1. Match current detections with active tracks using greedy IoU
        if self.active_tracks and detections:
            cost_matrix: List[Tuple[float, int, int]] = []
            for det_idx, det in enumerate(detections):
                for track_id, track in self.active_tracks.items():
                    iou = calculate_box_iou(det["bbox"], track.bbox)
                    if iou >= self.iou_threshold:
                        cost_matrix.append((iou, det_idx, track_id))

            # Sort descending by IoU
            cost_matrix.sort(key=lambda x: x[0], reverse=True)

            for iou, det_idx, track_id in cost_matrix:
                if det_idx in matched_det_indices or track_id in matched_tracks:
                    continue
                matched_det_indices.add(det_idx)
                matched_tracks.add(track_id)

                det = detections[det_idx]
                raw_text = det.get("plate_text", "")
                if raw_text:
                    self.total_raw_predictions += 1

                track = self.active_tracks[track_id]
                prev_text, prev_stable, _, _ = track.get_voted_result(self.min_votes)

                track.update(det["bbox"], raw_text, det["confidence"], current_time)
                voted_text, is_stable, votes, total_samples = track.get_voted_result(self.min_votes)

                # Check transition stability
                if prev_stable and not is_stable:
                    self.unstable_transitions += 1

                # Cooldown check for terminal notification
                is_new_stable_event = False
                if is_stable and voted_text:
                    time_since_report = current_time - track.last_reported_time
                    if track.last_reported_text != voted_text or time_since_report >= self.plate_cooldown:
                        is_new_stable_event = True
                        track.last_reported_text = voted_text
                        track.last_reported_time = current_time
                        self.total_stable_events += 1

                stabilized_results.append({
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "layout": det.get("layout", "unknown"),
                    "raw_text": raw_text,
                    "plate_text": voted_text if is_stable else (raw_text or voted_text),
                    "is_stable": is_stable,
                    "votes": votes,
                    "total_samples": total_samples,
                    "track_id": track_id,
                    "is_new_stable_event": is_new_stable_event,
                })

        # 2. Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                raw_text = det.get("plate_text", "")
                if raw_text:
                    self.total_raw_predictions += 1

                track_id = self._next_track_id
                self._next_track_id += 1

                new_track = PlateTrack(
                    track_id=track_id,
                    bbox=det["bbox"],
                    initial_text=raw_text,
                    confidence=det["confidence"],
                    timestamp=current_time,
                    window_size=self.vote_window,
                )
                self.active_tracks[track_id] = new_track

                voted_text, is_stable, votes, total_samples = new_track.get_voted_result(self.min_votes)

                is_new_stable_event = False
                if is_stable and voted_text:
                    is_new_stable_event = True
                    new_track.last_reported_text = voted_text
                    new_track.last_reported_time = current_time
                    self.total_stable_events += 1

                stabilized_results.append({
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "layout": det.get("layout", "unknown"),
                    "raw_text": raw_text,
                    "plate_text": voted_text if is_stable else raw_text,
                    "is_stable": is_stable,
                    "votes": votes,
                    "total_samples": total_samples,
                    "track_id": track_id,
                    "is_new_stable_event": is_new_stable_event,
                })

        # 3. Clean up inactive tracks that missed too many frames
        dead_tracks = []
        for track_id, track in self.active_tracks.items():
            if track_id not in matched_tracks:
                track.consecutive_misses += 1
                if track.consecutive_misses > self.max_missed_frames:
                    dead_tracks.append(track_id)

        for track_id in dead_tracks:
            del self.active_tracks[track_id]

        return stabilized_results
