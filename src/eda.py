"""Read-only media and annotation helpers used by the exploration notebook."""
from pathlib import Path
import warnings

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image


def helmet_mask(labels):
    """Include all supplied helmet variants, e.g. blurred, partial, sideline."""
    return labels["label"].astype("string").str.startswith("Helmet", na=False)


def find_repo_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "src" / "eda.py").is_file() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Run from the repository root or notebooks/ directory.")


def media_index(data_dir, suffixes, preferred_subdir=None):
    """Resolve basenames recursively; explicitly prefer train over mock test copies."""
    paths = sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in suffixes)
    index, duplicates = {}, []
    for path in paths:
        if path.name in index:
            duplicates.append(path.name)
            if preferred_subdir and preferred_subdir in path.relative_to(data_dir).parts:
                index[path.name] = path
        else:
            index[path.name] = path
    return index, sorted(set(duplicates))


def image_dimensions(names, index):
    """Read headers, not all pixel arrays; missing/unreadable files stay visible."""
    rows = []
    for name in sorted(names):
        path = index.get(name)
        row = {"image": name, "image_width": np.nan, "image_height": np.nan,
               "status": "missing"}
        if path:
            try:
                with Image.open(path) as image:
                    row.update(image_width=image.width, image_height=image.height, status="available")
            except (OSError, ValueError):
                row["status"] = "unreadable"
        rows.append(row)
    return pd.DataFrame(rows)


def video_metadata(names, index):
    rows = []
    for name in sorted(names):
        row = {"video": name, "fps": np.nan, "frame_count": np.nan,
               "video_width": np.nan, "video_height": np.nan, "status": "missing"}
        if name in index:
            cap = cv2.VideoCapture(str(index[name]))
            try:
                if cap.isOpened():
                    row.update(fps=cap.get(cv2.CAP_PROP_FPS),
                               frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT),
                               video_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                               video_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT), status="available")
                    if row["fps"] <= 0 or row["frame_count"] <= 0:
                        row["status"] = "invalid metadata"
                else:
                    row["status"] = "unreadable"
            finally:
                cap.release()
        rows.append(row)
    return pd.DataFrame(rows)


def box_checks(labels, dimensions=None, key="image", width_col="image_width", height_col="image_height"):
    boxes = labels.copy()
    for col in ["left", "top", "width", "height"]:
        boxes[col] = pd.to_numeric(boxes[col], errors="coerce")
    boxes["invalid_box"] = (~np.isfinite(boxes[["left", "top", "width", "height"]]).all(axis=1)
                            | boxes.width.le(0) | boxes.height.le(0)
                            | boxes.left.lt(0) | boxes.top.lt(0))
    boxes["area"] = boxes.width * boxes.height
    boxes["aspect_ratio"] = boxes.width.div(boxes.height.where(boxes.height.gt(0)))
    if dimensions is not None:
        boxes = boxes.merge(dimensions[[key, width_col, height_col]], on=key, how="left", validate="many_to_one")
        boxes["bounds_checked"] = boxes[width_col].gt(0) & boxes[height_col].gt(0)
        boxes["outside_image"] = boxes.bounds_checked & (
            boxes.left.lt(0) | boxes.top.lt(0)
            | (boxes.left + boxes.width).gt(boxes[width_col])
            | (boxes.top + boxes.height).gt(boxes[height_col]))
    return boxes


def draw_boxes(ax, rgb, annotations, title="", show_boxes=True, crop=None):
    ax.imshow(rgb)
    if show_boxes:
        for row in annotations.to_dict("records"):
            coords = [row[k] for k in ("left", "top", "width", "height")]
            if not np.isfinite(coords).all() or row["width"] <= 0 or row["height"] <= 0:
                continue
            positive = pd.notna(row.get("impact")) and row.get("impact") == 1
            color = "#ef4444" if positive else "#22c55e"
            ax.add_patch(Rectangle((row["left"], row["top"]), row["width"], row["height"],
                                   fill=False, edgecolor=color, linewidth=1.5 if positive else 0.8,
                                   zorder=5 if positive else 2))
            if "video" in row:
                ax.text(row["left"], row["top"] - 2, str(row["label"]), fontsize=7,
                        color="white", clip_on=True, zorder=6 if positive else 3,
                        bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"))
    if crop:
        left, top, right, bottom = crop
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def show_image(image_name, labels, index, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7))
    path = index.get(image_name)
    image = cv2.imread(str(path)) if path else None
    if image is None:
        ax.text(0.5, 0.5, f"Missing/unreadable image\n{image_name}", ha="center", va="center")
        ax.axis("off")
        return False
    annotations = labels.loc[labels.image.eq(image_name)]
    draw_boxes(ax, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), annotations,
               f"{image_name}\n{helmet_mask(annotations).sum()} helmets")
    return True


def read_video_frame(video_name, frame_number, index, frame_base=1):
    """CSV frames are one-based; OpenCV positions are zero-based. Never clamp."""
    if pd.isna(frame_number) or not np.isfinite(float(frame_number)) or float(frame_number) != int(frame_number):
        raise ValueError("Frame number must be a finite integer.")
    position = int(frame_number) - frame_base
    if position < 0:
        raise ValueError(f"Frame must be >= {frame_base}.")
    path = index.get(video_name)
    if path is None:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or position >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
            return None
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, position):
            return None
        ok, bgr = cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
    finally:
        cap.release()


def plot_video_frame(video_name, frame_number, labels, index, show_boxes=True, ax=None, crop=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 7))
    rgb = read_video_frame(video_name, frame_number, index)
    if rgb is None:
        ax.text(0.5, 0.5, f"Frame unavailable\n{video_name} / {frame_number}", ha="center", va="center")
        ax.axis("off")
        return False
    rows = labels.loc[labels.video.eq(video_name) & labels.frame.eq(frame_number)]
    draw_boxes(ax, rgb, rows, f"{video_name} / frame {frame_number}\n{len(rows)} helmets; red = impact label", show_boxes, crop)
    return True


def impact_sequence(row, labels, index):
    """Nine exact consecutive frames, fixed context window from the center label."""
    center = int(row.frame)
    rgb = read_video_frame(row.video, center, index)
    if rgb is None:
        warnings.warn("Selected impact frame cannot be decoded.")
        return None
    height, width = rgb.shape[:2]
    cx, cy = row.left + row.width / 2, row.top + row.height / 2
    radius = max(100, row.width * 4, row.height * 4)
    crop = (max(0, cx - radius), max(0, cy - radius), min(width, cx + radius), min(height, cy + radius))
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), layout="constrained")
    for offset, ax in zip(range(-4, 5), axes.flat):
        frame = center + offset
        if frame < 1:
            ax.text(0.5, 0.5, "Outside video", ha="center"); ax.axis("off")
            continue
        plot_video_frame(row.video, frame, labels, index, ax=ax, crop=crop)
        ax.set_title(f"Frame {frame} ({offset:+d})" + ("\nANNOTATED IMPACT" if offset == 0 else ""),
                     color="#dc2626" if offset == 0 else "#334155")
    fig.suptitle(f"Helmet context: player {row.label} | {row.video}", fontsize=15)
    return fig
