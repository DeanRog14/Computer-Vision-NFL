"""Run helmet detection, tracking, and nine-frame impact inference on a video."""

import argparse
from collections import defaultdict, deque
from contextlib import closing, contextmanager
import json
import math
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.detection.common import checkpoint_path, configure_runtime, require_helmet_checkpoint
from src.impact.common import device_for, load_config, project_path
from src.impact.dataset import BOX, aligned_boxes, clip_tensor
from src.impact.model import load_model
from src.impact.tracking import HelmetTracker


CANDIDATE_COLUMNS = ["video", "frame", "player", "view", *BOX, "probability",
                     "missing_positions", "track_id", "detector_confidence"]
EVENT_COLUMNS = [*CANDIDATE_COLUMNS, "event_id", "start_frame", "end_frame",
                 "start_seconds", "peak_seconds", "end_seconds", "positive_frames"]


@contextmanager
def open_video(video):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open {video}")
        yield cap
    finally:
        cap.release()


def video_windows(video, detector, confidence, device, *, tracker=None, progress=False):
    """Yield one centered nine-frame RGB window per source frame, with edge repeats."""
    tracker = HelmetTracker() if tracker is None else tracker
    window = deque(maxlen=9)
    number = 0
    with open_video(video) as cap:
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=expected or None, desc="Detect and score", unit="frame",
                  disable=not progress) as bar:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    if number < expected:
                        raise OSError(f"Decode failed at frame {number + 1} of {expected}")
                    break
                number += 1
                result = detector.predict(bgr, conf=confidence, device=str(device), verbose=False)[0]
                xyxy = result.boxes.xyxy.cpu().numpy()
                boxes = np.column_stack((xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]))
                scores = result.boxes.conf.cpu().tolist()
                track_ids = tracker.update(number, boxes)
                detections = dict(zip(track_ids, zip(boxes, scores)))
                item = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), detections, number)
                if number == 1:
                    window.extend([item] * 4)
                window.append(item)
                if len(window) == 9:
                    yield number - 4, list(window)
                    bar.update()

            if number == 0:
                raise OSError(f"Empty video: {video}")
            for offset in range(1, 5):
                window.append(item)
                center = number + offset - 4
                if center >= 1:
                    yield center, list(window)
                    bar.update()


def score_window(model, device, window, settings, batch_size):
    frames, detections, numbers = zip(*window)
    center = numbers[4]
    candidates = list(detections[4].items())
    for start in range(0, len(candidates), batch_size):
        clips, details = [], []
        for track_id, (box, confidence) in candidates[start:start + batch_size]:
            positions = {number: observed[track_id][0]
                         for number, observed in zip(numbers, detections) if track_id in observed}
            aligned, missing = aligned_boxes(positions, center, numbers, settings["missing_position"])
            clips.append(clip_tensor(frames, aligned, settings["size"], settings["context"]))
            details.append({**dict(zip(BOX, box.tolist())), "track_id": track_id,
                            "detector_confidence": confidence, "missing_positions": sum(missing)})
        probabilities = model(torch.stack(clips).to(device)).flatten().sigmoid().cpu().tolist()
        for row, probability in zip(details, probabilities):
            yield {**row, "probability": probability}


def group_events(predictions, threshold, gap, fps):
    """Group above-threshold frames by track and retain each group's peak box."""
    positive = predictions.loc[predictions.probability.ge(threshold)]
    events = []
    for _, track in positive.groupby(["video", "view", "track_id"], sort=False):
        track = track.sort_values("frame")
        segment = track.frame.diff().gt(gap).cumsum()
        for _, group in track.groupby(segment, sort=False):
            peak = group.loc[group.probability.idxmax()].to_dict()
            start, end = int(group.frame.iloc[0]), int(group.frame.iloc[-1])
            peak.update(start_frame=start, end_frame=end, positive_frames=len(group),
                        start_seconds=(start - 1) / fps, peak_seconds=(peak["frame"] - 1) / fps,
                        end_seconds=(end - 1) / fps)
            events.append(peak)
    events.sort(key=lambda row: (row["video"], row["start_frame"], row["track_id"]))
    for event_id, row in enumerate(events, start=1):
        row["event_id"] = event_id
    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def render_events(video, events, output, *, predictions=None, frames=None, progress=False):
    """Draw detected helmets and highlight their active impact intervals."""
    by_frame = defaultdict(list)
    for row in (events if predictions is None else predictions).itertuples(index=False):
        by_frame[row.frame].append(row)

    starts, ends = defaultdict(list), defaultdict(list)
    if predictions is not None:
        for event in events.itertuples(index=False):
            starts[event.start_frame].append(event)
            ends[event.end_frame].append(event.track_id)

    with open_video(video) as cap:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        active = {}
        number = 0
        try:
            if not writer.isOpened():
                raise OSError(f"Cannot create {output}")
            with tqdm(total=frames, desc="Render MP4", unit="frame", disable=not progress) as bar:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    number += 1
                    for event in starts.get(number, ()):
                        active[event.track_id] = event
                    for row in by_frame.get(number, ()):
                        event = row if predictions is None else active.get(row.track_id)
                        draw_helmet(frame, row, event)
                    for track_id in ends.get(number, ()):
                        del active[track_id]
                    writer.write(frame)
                    bar.update()
            if frames is not None and number != frames:
                raise OSError(f"Rendered {number} frames, expected {frames}")
        finally:
            writer.release()


def draw_helmet(frame, row, event):
    x, y, w, h = (round(getattr(row, key)) for key in BOX)
    color = (0, 0, 255) if event is not None else (0, 210, 0)
    label = f"#{row.track_id}" if hasattr(row, "track_id") else ""
    if event is not None:
        event_id = f" {event.event_id}" if hasattr(event, "event_id") else ""
        label += f" impact{event_id} {event.probability:.2f}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label.strip(), (max(0, x), max(12, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


@torch.inference_mode()
def infer(config, video, classifier, helmet, output, render=False, threshold=None, *,
          device=None, batch_size=None, progress=False):
    video, classifier, helmet, output = map(project_path, (video, classifier, helmet, output))
    if not video.is_file():
        raise FileNotFoundError(video)
    checkpoint_path(str(helmet))
    if not classifier.is_file():
        raise FileNotFoundError(classifier)
    if output.exists():
        raise FileExistsError(output)

    settings = config["inference"]
    device = device_for(device or settings.get("device", config["train"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device auto or --device cpu.")
    if batch_size is None:
        batch_size = settings.get("batch_size", config["train"]["batch_size"])
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    gap = settings["event_gap"]
    if gap < 0:
        raise ValueError("Event gap must be nonnegative")
    if config["data"]["boundary"] != "repeat":
        raise ValueError("Streaming deployment requires boundary=repeat")
    tracker = HelmetTracker(settings.get("track_iou", 0.1), settings.get("track_max_missing", 4))
    with open_video(video) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid frame rate for {video}: {fps}")

    model, saved = load_model(classifier, device)
    if saved["config"]["data"] != config["data"]:
        raise ValueError("Inference preprocessing differs from training checkpoint")
    if threshold is None:
        threshold = settings.get("threshold")
    threshold = float(saved["threshold"] if threshold is None else threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")

    configure_runtime()
    from ultralytics import YOLO

    detector = YOLO(str(helmet))
    require_helmet_checkpoint(detector)
    records = []
    view = video.stem.rsplit("_", 1)[-1]
    frames = 0
    with closing(video_windows(video, detector, settings["detector_confidence"], device,
                               tracker=tracker, progress=progress)) as windows:
        for frames, window in windows:
            records.extend({"video": video.name, "frame": frames, "player": "unknown", "view": view,
                            **row} for row in score_window(model, device, window, config["data"], batch_size))

    predictions = pd.DataFrame(records, columns=CANDIDATE_COLUMNS)
    events = group_events(predictions, threshold, gap, fps)
    report = {"video": str(video), "classifier": str(classifier), "helmet": str(helmet),
              "device": str(device), "fps": fps, "frames": frames, "threshold": threshold,
              "batch_size": batch_size, "detector_confidence": settings["detector_confidence"],
              "track_iou": tracker.min_iou, "track_max_missing": tracker.max_missing, "event_gap": gap,
              "tracks": int(predictions.track_id.nunique()), "candidates": len(predictions),
              "events": len(events), "annotated_video": "annotated.mp4" if render else None,
              "position_source": "YOLO detections linked by video-local helmet tracks"}

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".impact-", dir=output.parent) as temporary:
        staging = Path(temporary)
        predictions.to_csv(staging / "candidates.csv", index=False)
        events.to_csv(staging / "impacts.csv", index=False)
        events.to_json(staging / "impacts.json", orient="records", indent=2, double_precision=15)
        if render:
            render_events(video, events, staging / "annotated.mp4", predictions=predictions,
                          frames=frames, progress=progress)
        (staging / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        copyfile(staging / "impacts.csv", staging / "events.csv")
        copyfile(staging / "summary.json", staging / "inference.json")
        output.mkdir()
        for artifact in staging.iterdir():
            artifact.rename(output / artifact.name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/impact.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", help="Override the configured impact classifier")
    parser.add_argument("--helmet-checkpoint", help="Override the configured helmet detector")
    parser.add_argument("--output", required=True, help="New directory for reports and annotated video")
    parser.add_argument("--threshold", type=float, help="Override checkpoint's validation threshold")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Inference device override")
    parser.add_argument("--batch-size", type=int, help="Maximum impact clips per forward pass")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True,
                        help="Write annotated.mp4 (enabled by default)")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                        help="Show inference and rendering progress (enabled by default)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    classifier = args.checkpoint or paths.get("impact_checkpoint", "weights/impact_best.pt")
    helmet = args.helmet_checkpoint or paths["helmet_checkpoint"]
    report = infer(cfg, args.video, classifier, helmet, args.output, args.render, args.threshold,
                   device=args.device, batch_size=args.batch_size, progress=args.progress)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
