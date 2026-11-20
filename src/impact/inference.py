"""Streaming YOLO candidates -> nine-frame classifier -> temporal NMS.

Only local, existing checkpoints are accepted. This module never trains.
"""
import argparse
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.detection.common import checkpoint_path, configure_runtime, require_helmet_checkpoint
from src.impact.common import load_config, project_path, device_for
from src.impact.dataset import BOX, clip_tensor, aligned_boxes
from src.impact.evaluate import group_predictions, iou
from src.impact.model import load_model


def video_windows(video, detector, confidence, device):
    """Yield center index and nine (RGB, boxes, actual frame) records, O(9) frames."""
    cap = cv2.VideoCapture(str(video))
    window = deque(maxlen=9)
    number = 0
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open {video}")
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, bgr = cap.read()
            if not ok:
                if number < expected:
                    raise OSError(f"Decode failed at frame {number+1} of {expected}")
                break
            number += 1
            result = detector.predict(bgr, conf=confidence, device=str(device), verbose=False)[0]
            xyxy = result.boxes.xyxy.cpu().numpy()
            boxes = np.column_stack([xyxy[:, :2], xyxy[:, 2:]-xyxy[:, :2]])
            item = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), boxes, number)
            if number == 1:
                window.extend([item]*4)
            window.append(item)
            if len(window) == 9:
                yield number-4, list(window)
        if number == 0:
            raise OSError("Empty video")
        for offset in range(1, 5):
            window.append(item)
            center = number + offset - 4
            if len(window) == 9 and center >= 1:
                yield center, list(window)
    finally:
        cap.release()


def render_events(video, events, output):
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Cannot create {output}")
    by_frame = {int(f): rows for f, rows in events.groupby("frame")}
    number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            number += 1
            if number in by_frame:
                for row in by_frame[number].itertuples():
                    x, y, w, h = (int(getattr(row, key)) for key in BOX)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f"impact {row.probability:.2f}", (x, max(15, y-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


@torch.inference_mode()
def infer(config, video, classifier, helmet, output, render=False, threshold=None):
    video, classifier, helmet, output = map(project_path, (video, classifier, helmet, output))
    if not video.is_file():
        raise FileNotFoundError(video)
    checkpoint_path(str(helmet))
    if not classifier.is_file():
        raise FileNotFoundError(classifier)
    if output.exists():
        raise FileExistsError(output)
    device = device_for(config["train"]["device"])
    model, saved = load_model(classifier, device)
    if saved["config"]["data"] != config["data"]:
        raise ValueError("Inference preprocessing differs from training checkpoint")
    if threshold is None:
        threshold = config["inference"].get("threshold")
    threshold = float(saved["threshold"] if threshold is None else threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")
    configure_runtime()
    from ultralytics import YOLO
    detector = YOLO(str(helmet))
    require_helmet_checkpoint(detector)
    if config["data"]["boundary"] != "repeat":
        raise ValueError("Streaming deployment requires boundary=repeat")
    output.mkdir(parents=True)
    records = []
    for center, window in video_windows(video, detector, config["inference"]["detector_confidence"], device):
        frames = [item[0] for item in window]
        numbers = [item[2] for item in window]
        candidates = window[4][1]
        for start in range(0, len(candidates), config["train"]["batch_size"]):
            clips, details = [], []
            for box in candidates[start:start+config["train"]["batch_size"]]:
                positions = {center: box}
                # Local geometric association, not a claimed player tracker.
                # Missing matches use the same explicit fallback as training.
                for _, others, number in window:
                    if number == center or len(others) == 0:
                        continue
                    scores = [iou(box, other) for other in others]
                    best = int(np.argmax(scores))
                    if scores[best] >= .1:
                        positions[number] = others[best]
                boxes, missing = aligned_boxes(positions, center, numbers, config["data"]["missing_position"])
                clips.append(clip_tensor(frames, boxes, config["data"]["size"], config["data"]["context"]))
                details.append((box, sum(missing)))
            scores = model(torch.stack(clips).to(device)).flatten().sigmoid().cpu().tolist()
            for (box, missing), score in zip(details, scores):
                records.append({"video": video.name, "frame": center, "player": "unknown",
                                "view": video.stem.rsplit("_", 1)[-1], **dict(zip(BOX, box.tolist())),
                                "probability": score, "missing_positions": missing})
    predictions = pd.DataFrame(records, columns=["video", "frame", "player", "view", *BOX,
                                                 "probability", "missing_positions"])
    predictions.to_csv(output / "candidates.csv", index=False)
    events = group_predictions(predictions, threshold, config["inference"]["event_gap"], config["inference"]["event_iou"])
    events.to_csv(output / "events.csv", index=False)
    report = {"video": str(video), "classifier": str(classifier), "helmet": str(helmet),
              "threshold": threshold, "candidates": len(predictions), "events": len(events),
              "position_source": "predicted boxes; local IoU association; player identity unknown"}
    (output / "inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if render:
        render_events(video, events, output / "annotated.mp4")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/impact.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--helmet-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, help="Override checkpoint's validation threshold")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(infer(cfg, args.video, args.checkpoint,
                           args.helmet_checkpoint or cfg["paths"]["helmet_checkpoint"],
                           args.output, args.render, args.threshold), indent=2))


if __name__ == "__main__":
    main()
