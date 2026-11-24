"""Manifest preparation and RGB clips with one-based NFL frame indices.

Raw annotations are retained as a position source, including ambiguous impacts.
Only definitive impacts and explicitly selected background enter classification.
"""
import argparse
import json
from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.prepare_yolo_data import assign_game_splits, sha256
from src.impact.common import load_config, project_path

BOX = ["left", "top", "width", "height"]
MEAN = np.array([.43216, .394666, .37645], dtype=np.float32)
STD = np.array([.22803, .22145, .216989], dtype=np.float32)


def classify_labels(rows, blank_is_negative=False):
    positive = rows.impact.eq(1) & rows.confidence.gt(1) & rows.visibility.gt(0)
    negative = rows.impact.eq(0)
    if blank_is_negative:
        # Only completely unannotated impact fields are background. Partial
        # annotations and low-confidence/occluded impacts remain ambiguous.
        negative |= rows[["impact", "confidence", "visibility"]].isna().all(axis=1)
    return pd.Series(np.where(positive, 1, np.where(negative, 0, -1)), index=rows.index)


def read_labels(path):
    rows = pd.read_csv(path, dtype={"video": str, "label": str, "gameKey": str})
    required = {"video", "frame", "label", "gameKey", "view", "impact", "confidence", "visibility", *BOX}
    if required - set(rows):
        raise ValueError(f"Missing columns: {required - set(rows)}")
    values = rows[["frame", *BOX]].to_numpy(float)
    if (not np.isfinite(values).all()
            or (rows.frame % 1 != 0).any() or (rows[["width", "height"]] <= 0).any().any()
            or rows[["video", "label", "gameKey", "view"]].isna().any().any()):
        raise ValueError("Invalid annotation metadata, frame or bounding box")
    invalid = rows.frame.lt(1)
    if invalid.any():
        if rows.loc[invalid].impact.eq(1).any():
            raise ValueError("Impact annotations have invalid one-based frames; review source")
        warnings.warn(f"Excluding {int(invalid.sum())} background annotation(s) with frame < 1", stacklevel=2)
        rows = rows.loc[~invalid].copy()
    rows.attrs["invalid_frame_rows"] = int(invalid.sum())
    rows.attrs["ambiguous_position_rows"] = int(rows.duplicated(["video", "frame", "label"], keep=False).sum())
    if not rows.view.isin(["Endzone", "Sideline"]).all():
        raise ValueError("Unexpected camera view")
    if any(Path(name).name != name for name in rows.video):
        raise ValueError("video must be a basename")
    return rows


def mark_events(rows, gap=4):
    """Consecutive positive annotations for one player/view share an event ID."""
    rows = rows.sort_values(["video", "label", "frame"]).copy()
    positives = rows.loc[rows.target.eq(1)]
    rows["event_id"] = ""
    number = 0
    for _, group in positives.groupby(["video", "label"], sort=True):
        active = []
        for index, row in group.iterrows():
            box = row[BOX].to_numpy(float)
            center = box[:2] + box[2:]/2
            active = [entry for entry in active if row.frame-entry[0] <= gap]
            matches = [(np.linalg.norm((center-entry[1])/box[2:]), k)
                       for k, entry in enumerate(active) if entry[0] < row.frame]
            distance, k = min(matches, default=(np.inf, -1))
            if distance <= 2:
                event_id = active[k][2]
                active[k] = (row.frame, center, event_id)
            else:
                number += 1
                event_id = f"event_{number}"
                active.append((row.frame, center, event_id))
            rows.loc[index, "event_id"] = event_id
    return rows


def mark_hard_negatives(rows, distance=2.0):
    """Nearby helmet centers, normalized by mean helmet size; no impact inferred."""
    result = pd.Series(False, index=rows.index)
    for _, group in rows.groupby(["video", "frame"], sort=False):
        b = group[BOX].to_numpy(float)
        centers = b[:, :2] + b[:, 2:] / 2
        delta = centers[:, None] - centers[None, :]
        scale = (b[:, None, 2:] + b[None, :, 2:]) / 2
        distances = np.sqrt(((delta / scale) ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        result.loc[group.index] = (distances <= distance).any(axis=1)
    return result


def split_games(rows, existing=None, seed=42):
    mapping = assign_game_splits(rows.gameKey, seed)
    if existing is not None and Path(existing).is_file():
        old = pd.read_csv(existing, dtype={"game": str})
        if old.groupby("game").split.nunique().gt(1).any() or not old.split.isin(["train", "val", "test"]).all():
            raise ValueError("Existing splits contain leakage or unknown split names")
        mapping.update(old.drop_duplicates("game").set_index("game").split.to_dict())
    splits = rows.gameKey.map(mapping)
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Compatible game assignments do not yield three nonempty splits")
    return splits


def sample_training(rows, ratio=5, hard_fraction=.5, spacing=9, seed=42):
    if ratio < 0 or not 0 <= hard_fraction <= 1 or spacing < 1:
        raise ValueError("Invalid sampling settings")
    positives = rows.loc[rows.target.eq(1)]
    if positives.empty:
        raise ValueError("Training split contains no definitive positives")
    candidates = rows.loc[rows.target.eq(0)].sort_values(["video", "label", "frame"])
    if candidates.empty or ratio == 0:
        raise ValueError("Binary training requires negatives; review blank_is_negative and negative_ratio")
    # One randomly chosen example per player/time bin, rather than dense neighbors.
    candidates = candidates.assign(time_bin=(candidates.frame - 1) // spacing)
    candidates = candidates.sample(frac=1, random_state=seed).drop_duplicates(["video", "label", "time_bin"])
    count = min(len(candidates), int(len(positives) * ratio))
    hard = candidates.loc[candidates.hard_negative].sample(
        n=min(int(count * hard_fraction), int(candidates.hard_negative.sum())), random_state=seed)
    rest = candidates.drop(hard.index).sample(n=count-len(hard), random_state=seed)
    return pd.concat([positives, hard, rest]).drop(columns="time_bin", errors="ignore").sort_index()


def prepare(config):
    cfg, paths = config["data"], config["paths"]
    destination = project_path(paths["manifest"])
    if destination.exists():
        raise FileExistsError(destination)
    rows = read_labels(project_path(paths["labels"]))
    source_quality = dict(rows.attrs)
    rows["target"] = classify_labels(rows, cfg["blank_is_negative"])
    existing = project_path(paths["existing_splits"]) if paths.get("existing_splits") else None
    rows["split"] = split_games(rows, existing, config["seed"])
    rows = mark_events(rows, cfg["event_gap"])
    rows["hard_negative"] = mark_hard_negatives(rows, cfg["hard_distance"]) & rows.target.eq(0)
    missing = [v for v in rows.video.unique() if not (project_path(paths["videos"]) / v).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} training videos: {missing[:5]}")
    eligible = rows.loc[rows.target.ge(0)]
    train = sample_training(eligible.loc[eligible.split.eq("train")], cfg["negative_ratio"],
                            cfg["hard_fraction"], cfg["negative_spacing"], config["seed"])
    selected = pd.concat([train, eligible.loc[eligible.split.ne("train")]]).sort_index()
    selected["sample_id"] = selected.index
    # Each training event contributes total positive weight 1, while retaining
    # every eligible positive clip. Validation/test remain unweighted and natural.
    selected["sample_weight"] = 1.0
    pos = selected.target.eq(1) & selected.split.eq("train")
    selected.loc[pos, "sample_weight"] = 1 / selected.loc[pos].groupby("event_id").event_id.transform("size")
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination, index=False)
    report = {"raw_rows": len(rows) + source_quality.get("invalid_frame_rows", 0),
              "invalid_frame_excluded": source_quality.get("invalid_frame_rows", 0),
              "ambiguous_position_rows": source_quality.get("ambiguous_position_rows", 0),
              "ambiguous_excluded": int(rows.target.eq(-1).sum()),
              "labels_sha256": sha256(project_path(paths["labels"])), "config": config,
              "selected": selected.groupby(["split", "target"]).size().to_dict(),
              "events": selected.loc[selected.target.eq(1)].groupby("split").event_id.nunique().to_dict(),
              "game_assignments": rows[["gameKey", "split"]].drop_duplicates().to_dict("records")}
    report["selected"] = {f"{s}/{y}": int(n) for (s, y), n in report["selected"].items()}
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def frame_numbers(center, count, boundary="repeat"):
    if count < 1 or not 1 <= center <= count:
        raise ValueError("Center frame is outside the video")
    requested = np.arange(center-4, center+5)
    if boundary == "error" and ((requested < 1) | (requested > count)).any():
        raise ValueError("Nine-frame clip crosses a video boundary")
    if boundary not in {"repeat", "error"}:
        raise ValueError("Unknown boundary policy")
    return np.clip(requested, 1, count)


def read_frames(video, center, boundary="repeat"):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open {video}")
        numbers = frame_numbers(center, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), boundary)
        frames, cache = [], {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(numbers[0])-1)
        for number in numbers:
            if int(number) not in cache:
                ok, bgr = cap.read()
                if not ok:
                    raise OSError(f"Cannot decode {video} at frame {number}; clip is not silently dropped")
                cache[int(number)] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(cache[int(number)])
        return frames, numbers
    finally:
        cap.release()


def aligned_boxes(positions, center, numbers, fallback="center"):
    """Follow known player centers, holding target size fixed to prevent zoom jitter.

    Unknown positions use target center (default) or nearest annotated position
    within this nine-frame window. No lookups into other players or views.
    """
    target = np.asarray(positions[int(center)], dtype=float)
    result, missing = [], []
    for number in numbers:
        box = positions.get(int(number))
        missing.append(box is None)
        if box is None:
            box = target
            if fallback == "nearest":
                available = [f for f in positions if abs(f-int(number)) <= 4]
                if available:
                    box = positions[min(available, key=lambda f: (abs(f-int(number)), f))]
        box = np.asarray(box, dtype=float)
        center_xy = box[:2] + box[2:] / 2
        result.append(np.r_[center_xy - target[2:]/2, target[2:]])
    return result, missing


def crop_rgb(frame, box, size=112, context=2):
    x, y, w, h = map(float, box)
    # Warp with a constant black border preserves the helmet center and scale
    # even when context extends outside the source image.
    transform = np.array([[w*context/size, 0, x+w/2-w*context/2],
                          [0, h*context/size, y+h/2-h*context/2]], dtype=np.float32)
    return cv2.warpAffine(frame, transform, (size, size),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def clip_tensor(frames, boxes, size=112, context=2):
    crops = np.stack([crop_rgb(f, b, size, context) for f, b in zip(frames, boxes)])
    return normalize_crops(crops)


def normalize_crops(crops):
    """Identical normalization for decoded and cached uint8 THWC RGB crops."""
    values = (crops.astype(np.float32)/255 - MEAN) / STD
    return torch.from_numpy(values.transpose(3, 0, 1, 2).copy())


def epoch_validation_rows(rows, negatives=500, seed=42):
    """All definitive positives plus a fixed seeded set of negatives."""
    if negatives < 1:
        raise ValueError("Validation negatives must be positive")
    positive, negative = rows.loc[rows.target.eq(1)], rows.loc[rows.target.eq(0)]
    if positive.empty or negative.empty:
        raise ValueError("Epoch validation requires both classes")
    return pd.concat([positive, negative.sample(n=min(negatives, len(negative)), random_state=seed)]).sort_index()


class ImpactDataset(Dataset):
    def __init__(self, manifest, labels, videos, split, settings, cache=None, validation_negatives=None, selection_seed=42):
        self.rows = pd.read_csv(manifest, keep_default_na=False)
        if self.rows.groupby("gameKey").split.nunique().gt(1).any():
            raise ValueError("Game leakage")
        self.rows = self.rows.loc[self.rows.split.eq(split)].reset_index(drop=True)
        if validation_negatives is not None:
            if split != "val":
                raise ValueError("Epoch sampling is only allowed on validation")
            self.rows = epoch_validation_rows(self.rows, validation_negatives, selection_seed).reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Empty split: {split}")
        self.videos, self.settings = Path(videos), settings
        self.labels_path = labels
        self.cache = None
        if cache and cache.get("enabled", False):
            from src.impact.cache import CropCache
            self.cache = CropCache(project_path(cache["path"]), manifest, labels, settings, split)
            if split == "val" and cache.get("validation_subset", False) and validation_negatives is None:
                self.rows = self.rows.loc[self.rows.sample_id.isin(self.cache.indices)].reset_index(drop=True)
                if self.rows.empty:
                    raise ValueError("Cache has no fixed validation subset")
            if cache.get("require_all", False) and not self.rows.sample_id.isin(self.cache.indices).all():
                missing = int((~self.rows.sample_id.isin(self.cache.indices)).sum())
                raise ValueError(f"{split}: {missing} selected clips are not cached. Prepare the matching cache; MP4 fallback is disabled.")
            if self.rows.sample_id.isin(self.cache.indices).all():
                self.positions = None
                return
        self._load_positions()

    def _load_positions(self):
        labels = self.labels_path
        raw = read_labels(labels)
        # Anonymous labels (e.g. H00) and annotation errors may have multiple
        # helmets per player/frame. Never choose an arbitrary neighbor position.
        raw = raw.loc[~raw.duplicated(["video", "frame", "label"], keep=False)]
        wanted = set(zip(self.rows.video, self.rows.label))
        self.positions = {(video, player): dict(zip(group.frame.astype(int), group[BOX].to_numpy(float)))
                          for (video, player), group in raw.groupby(["video", "label"], sort=False)
                          if (video, player) in wanted}

    def __len__(self):
        return len(self.rows)

    def raw_clip(self, index):
        if self.positions is None:
            self._load_positions()
        row = self.rows.iloc[index]
        frames, numbers = read_frames(self.videos / row.video, int(row.frame), self.settings["boundary"])
        positions = dict(self.positions.get((row.video, row.label), {}))
        positions[int(row.frame)] = row[BOX].to_numpy(float)
        boxes, missing = aligned_boxes(positions, int(row.frame), numbers,
                                       self.settings["missing_position"])
        return frames, boxes, {"actual_frames": numbers.tolist(), "missing_positions": missing}

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        cached = self.cache is not None and row.sample_id in self.cache.indices
        if cached:
            clip = normalize_crops(self.cache.get(row.sample_id))
        else:
            frames, boxes, _ = self.raw_clip(index)
            clip = clip_tensor(frames, boxes, self.settings["size"], self.settings["context"])
        return {"clip": clip,
                "target": torch.tensor(float(row.target)), "weight": torch.tensor(float(row.sample_weight)),
                "index": index}


def dataset_for(config, split, purpose=None):
    p = config["paths"]
    cache = dict(config.get("cache") or {})
    negatives, seed = None, config["seed"]
    if split == "val" and purpose == "epoch":
        validation = config.get("validation", {})
        negatives, seed = validation.get("negatives", 500), validation.get("seed", seed)
        cache = {"enabled": True, "path": validation.get("cache_path", "data/processed/impact/val_epoch_500"),
                 "require_all": True, "validation_subset": False}
    elif purpose == "full":
        cache["validation_subset"] = False  # final evaluation must never inherit epoch sampling
    return ImpactDataset(project_path(p["manifest"]), project_path(p["labels"]),
                         project_path(p["videos"]), split, config["data"], cache, negatives, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/impact.yaml")
    args = parser.parse_args()
    print(json.dumps(prepare(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
