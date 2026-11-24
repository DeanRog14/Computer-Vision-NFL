"""Build immutable uint8 crop arrays by decoding each video sequentially once."""
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.prepare_yolo_data import sha256
from src.impact.common import load_config, project_path
from src.impact.dataset import BOX, aligned_boxes, crop_rgb, dataset_for, frame_numbers, epoch_validation_rows


def crop_settings(settings):
    return {key: settings[key] for key in ("frames", "size", "context", "boundary", "missing_position")}


def validation_rows(rows, count, seed):
    """Fixed proportional subset, retaining both classes when present."""
    if count >= len(rows):
        return rows
    positive = rows.loc[rows.target.eq(1)]
    negative = rows.loc[rows.target.eq(0)]
    if len(positive) and len(negative):
        if count < 2:
            raise ValueError("Validation subset needs at least two rows for both classes")
        npos = min(len(positive), max(1, round(count*len(positive)/len(rows))))
        nneg = min(len(negative), count-npos)
        npos = count-nneg
        return pd.concat([positive.sample(n=npos, random_state=seed),
                          negative.sample(n=nneg, random_state=seed)]).sort_index()
    return rows.sample(n=count, random_state=seed).sort_index()


def sequential_clips(dataset, selected):
    """Yield row index and uint8 crops, O(9 decoded frames) memory per video."""
    for video, group in selected.groupby("video", sort=True):
        cap = cv2.VideoCapture(str(dataset.videos / video))
        try:
            if not cap.isOpened():
                raise OSError(video)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            buffer, current = deque(maxlen=9), 0
            for index, row in group.sort_values("frame", kind="stable").iterrows():
                numbers = frame_numbers(int(row.frame), count, dataset.settings["boundary"])
                while current < numbers[-1]:
                    ok, bgr = cap.read()
                    current += 1
                    if not ok:
                        raise OSError(f"Cannot decode {video} frame {current}")
                    buffer.append((current, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                frames = dict(buffer)
                positions = dict(dataset.positions.get((row.video, row.label), {}))
                positions[int(row.frame)] = row[BOX].to_numpy(float)
                boxes, _ = aligned_boxes(positions, int(row.frame), numbers, dataset.settings["missing_position"])
                yield index, np.stack([crop_rgb(frames[int(n)], b, dataset.settings["size"], dataset.settings["context"])
                                       for n, b in zip(numbers, boxes)])
        finally:
            cap.release()


def build_cache(config, destination, val_size=1024, verify=16, train_limit=None, reuse_train_from=None,
                validation_only=False, val_negatives=500):
    destination = project_path(destination)
    if destination.exists():
        raise FileExistsError(f"Use a new cache directory: {destination}")
    if val_size < 1 or verify < 1 or (train_limit is not None and train_limit < 1):
        raise ValueError("Sample counts must be positive")
    cfg = copy.deepcopy(config)
    cfg["cache"] = {"enabled": False}
    destination.mkdir(parents=True)
    metadata = {"format": 1, "settings": crop_settings(cfg["data"]),
                "manifest_sha256": sha256(project_path(cfg["paths"]["manifest"])),
                "labels_sha256": sha256(project_path(cfg["paths"]["labels"])), "splits": {},
                "seed": cfg["seed"], "complete": False}
    for split in (["val"] if validation_only else ["train", "val"]):
        if split == "train" and reuse_train_from:
            source = project_path(reuse_train_from)
            cached = CropCache(source, project_path(cfg["paths"]["manifest"]),
                               project_path(cfg["paths"]["labels"]), cfg["data"], "train")
            expected = pd.read_csv(project_path(cfg["paths"]["manifest"]))
            expected = expected.loc[expected.split.eq("train")]
            if set(expected.sample_id) != set(cached.indices) or train_limit is not None:
                raise ValueError("Reused training cache must contain all training samples")
            cached.get(expected.sample_id.iloc[0])  # validate array header
            cached.close()
            for name in ["train.npy", "train.csv"]:
                shutil.copyfile(source / name, destination / name)
            old = json.loads((source / "cache.json").read_text())
            metadata["splits"]["train"] = old["splits"]["train"]
            metadata["reused_training_cache"] = str(source)
            continue
        ds = dataset_for(cfg, split)
        selected = ds.rows
        limit = train_limit if split == "train" else val_size
        if validation_only:
            selection_seed = cfg.get("validation", {}).get("seed", cfg["seed"])
            selected = epoch_validation_rows(selected, val_negatives, selection_seed)
            metadata["validation_selection"] = {"all_positives": True, "negatives": val_negatives, "seed": selection_seed}
        elif split == "val":
            selected = validation_rows(selected, val_size, cfg["seed"])
        elif limit is not None and limit < len(selected):
            selected = selected.sample(n=limit, random_state=cfg["seed"]).sort_index()
        if selected.sample_id.duplicated().any():
            raise ValueError("sample_id must be unique")
        slots = {index: slot for slot, index in enumerate(selected.index)}
        array = np.lib.format.open_memmap(destination / f"{split}.npy", mode="w+", dtype=np.uint8,
                                         shape=(len(selected), 9, cfg["data"]["size"], cfg["data"]["size"], 3))
        for index, crops in tqdm(sequential_clips(ds, selected), total=len(selected), desc=f"Caching {split}", unit="clip", mininterval=1):
            array[slots[index]] = crops
        array.flush()
        # Include both views/classes/hard negatives and boundary extremes where present.
        representatives = set(selected.sample(n=min(verify, len(selected)), random_state=cfg["seed"]).index)
        for _, group in selected.groupby(["view", "target", "hard_negative"]):
            representatives.add(group.index[0])
        representatives.update([selected.frame.idxmin(), selected.frame.idxmax()])
        for index in tqdm(sorted(representatives), desc=f"Verifying {split}", unit="clip", mininterval=1):
            frames, boxes, _ = ds.raw_clip(index)
            reference = np.stack([crop_rgb(f, b, cfg["data"]["size"], cfg["data"]["context"]) for f, b in zip(frames, boxes)])
            np.testing.assert_array_equal(array[slots[index]], reference)
        selected.to_csv(destination / f"{split}.csv", index=False)
        del array
        metadata["splits"][split] = {"count": len(selected), "verified": len(representatives),
                                     "positives": int(selected.target.eq(1).sum()), "negatives": int(selected.target.eq(0).sum()),
                                     "rows_sha256": sha256(destination / f"{split}.csv")}
    metadata["complete"] = True
    (destination / "cache.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class CropCache:
    def __init__(self, root, manifest, labels, settings, split):
        self.root, self.split, self.array = Path(root), split, None
        metadata = json.loads((self.root / "cache.json").read_text())
        if not metadata.get("complete") or metadata["settings"] != crop_settings(settings):
            raise ValueError("Incomplete cache or changed crop settings")
        if metadata["manifest_sha256"] != sha256(manifest) or metadata["labels_sha256"] != sha256(labels):
            raise ValueError("Cache source labels/manifest have changed")
        self.indices = {}
        if split in metadata["splits"]:
            path = self.root / f"{split}.csv"
            if sha256(path) != metadata["splits"][split]["rows_sha256"]:
                raise ValueError("Cached row metadata has changed")
            rows = pd.read_csv(path)
            self.indices = dict(zip(rows.sample_id, range(len(rows))))
            self.count, self.size = len(rows), settings["size"]

    def get(self, sample_id):
        if self.array is None:
            self.array = np.load(self.root / f"{self.split}.npy", mmap_mode="r", allow_pickle=False)
            if self.array.dtype != np.uint8 or self.array.shape != (self.count, 9, self.size, self.size, 3):
                raise ValueError("Invalid cache shape/dtype")
        return self.array[self.indices[sample_id]]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["array"] = None  # reopen mmap in workers; never pickle the crop array
        return state

    def close(self):
        if self.array is not None:
            self.array._mmap.close()
            self.array = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/impact.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--verify", type=int, default=16)
    parser.add_argument("--train-limit", type=int, help="Small smoke cache only; omit to cache all training rows")
    parser.add_argument("--reuse-train-from", help="Copy a complete compatible training cache when changing validation size")
    parser.add_argument("--validation-only", action="store_true", help="Cache all val positives plus fixed negatives; leave training cache untouched")
    parser.add_argument("--val-negatives", type=int, default=None, help="Epoch validation negatives (default config validation.negatives or 500)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    count = args.val_negatives if args.val_negatives is not None else cfg.get("validation", {}).get("negatives", 500)
    if args.validation_only and (args.train_limit is not None or args.reuse_train_from):
        parser.error("--validation-only cannot be combined with training cache options")
    print(json.dumps(build_cache(cfg, args.output, args.val_size, args.verify,
                                args.train_limit, args.reuse_train_from, args.validation_only, count), indent=2))


if __name__ == "__main__":
    main()
