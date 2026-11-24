"""Video-local helmet IDs from one-to-one motion and overlap matching."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    frame: int
    box: np.ndarray
    velocity: np.ndarray

    def predict(self, frame):
        box = self.box.copy()
        box[:2] += self.velocity * (frame - self.frame)
        return box


class HelmetTracker:
    def __init__(self, min_iou=0.1, max_missing=4):
        if not 0 < min_iou <= 1 or max_missing < 0:
            raise ValueError("Tracking requires 0 < min_iou <= 1 and max_missing >= 0")
        self.min_iou = min_iou
        self.max_missing = max_missing
        self.active = {}
        self.next_id = 1

    def update(self, frame, boxes):
        """Return one track ID per detection, in detector order."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.active = {key: track for key, track in self.active.items()
                       if frame - track.frame <= self.max_missing + 1}
        track_ids = list(self.active)
        assigned = {}
        if track_ids and len(boxes):
            predicted = np.stack([track.predict(frame) for track in self.active.values()])
            overlap = self.overlaps(predicted, boxes)
            valid = overlap >= self.min_iou
            # Prefer the maximum number of valid matches, then the highest total IoU.
            benefit = np.where(valid, 1 + overlap / (1 + min(overlap.shape)), 0)
            track_rows, detections = linear_sum_assignment(benefit, maximize=True)
            assigned = {detection: track_ids[row] for row, detection in zip(track_rows, detections)
                        if valid[row, detection]}

        for index, box in enumerate(boxes):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                assigned[index] = track_id
                velocity = np.zeros(2)
            else:
                track = self.active[track_id]
                displacement = (box[:2] - track.box[:2]) + (box[2:] - track.box[2:]) / 2
                velocity = displacement / (frame - track.frame)
            self.active[track_id] = Track(frame, box, velocity)
        return [assigned[index] for index in range(len(boxes))]

    @staticmethod
    def overlaps(predicted, boxes):
        predicted, boxes = predicted[:, None, :], boxes[None, :, :]
        left_top = np.maximum(predicted[..., :2], boxes[..., :2])
        right_bottom = np.minimum(predicted[..., :2] + predicted[..., 2:],
                                  boxes[..., :2] + boxes[..., 2:])
        intersection = np.maximum(0, right_bottom - left_top).prod(axis=-1)
        union = predicted[..., 2:].prod(axis=-1) + boxes[..., 2:].prod(axis=-1) - intersection
        return intersection / np.maximum(union, 1e-12)
