"""Convert NFL still-image boxes into a reproducible, game-disjoint YOLO dataset."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from src.eda import media_index, image_dimensions, draw_boxes

SPLITS = ("train", "val", "test")
RATIOS = np.array([.8, .1, .1])
HELMET_LABELS = {"Helmet", "Helmet-Blurred", "Helmet-Sideline", "Helmet-Partial", "Helmet-Difficult"}
FILENAME = re.compile(r"^(\d+)_(\d+)_(Sideline|Endzone)_frame\d+\.(?:jpg|jpeg|png)$", re.I)


def project_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_game_splits(games, seed=42):
    """80/10/10 allocation of whole games; image fractions are approximate."""
    unique = np.array(sorted(set(map(str, games))))
    if len(unique) < 3:
        raise ValueError("At least three games are needed for disjoint train/val/test sets.")
    counts = np.floor(len(unique) * RATIOS).astype(int)
    remainder = len(unique) - counts.sum()
    order = np.argsort(-(len(unique) * RATIOS - counts), kind="stable")
    counts[order[:remainder]] += 1
    # Keep all splits nonempty for very small fixture datasets.
    for index in range(3):
        if counts[index] == 0:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    shuffled = np.random.default_rng(seed).permutation(unique)
    result, start = {}, 0
    for split, count in zip(SPLITS, counts):
        result.update({str(game): split for game in shuffled[start:start + count]})
        start += count
    return result


def convert_boxes(boxes, image_width, image_height):
    """Strict conversion; no silent dropping, clipping, or invented background."""
    values = np.asarray(boxes, dtype=float).reshape(-1, 4)  # left, top, width, height
    if not np.isfinite([image_width, image_height]).all() or min(image_width, image_height) <= 0:
        raise ValueError("Image dimensions must be positive and finite.")
    if (not np.isfinite(values).all() or (values[:, :2] < 0).any()
            or (values[:, 2:] <= 0).any()):
        raise ValueError("Boxes must be finite with nonnegative origins and positive sizes.")
    if ((values[:, 0] + values[:, 2] > image_width).any()
            or (values[:, 1] + values[:, 3] > image_height).any()):
        raise ValueError("Box extends beyond image boundaries; review the original annotation.")
    normalized = values.copy()
    normalized[:, :2] += values[:, 2:] / 2
    normalized /= np.array([image_width, image_height, image_width, image_height])
    if (normalized < 0).any() or (normalized > 1).any():
        raise ValueError("Normalized coordinates must lie between zero and one.")
    return normalized


def read_yolo_labels(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return np.empty((0, 5))
    values = np.array([[float(item) for item in line.split()] for line in text.splitlines()])
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError(f"Expected five columns in {path}")
    if (not np.isfinite(values).all() or (values[:, 0] != 0).any()
            or (values[:, 1:] < 0).any() or (values[:, 1:] > 1).any()
            or (values[:, 3:] <= 0).any()):
        raise ValueError(f"Invalid class IDs or normalized boxes in {path}")
    centers, sizes = values[:, 1:3], values[:, 3:5]
    if (centers - sizes / 2 < -1e-9).any() or (centers + sizes / 2 > 1 + 1e-9).any():
        raise ValueError(f"Normalized box edges are outside the image in {path}")
    return values


def validate_dataset(output, verify_hashes=True):
    output = Path(output).resolve()
    manifest = pd.read_csv(output / "manifest.csv", dtype={"game": str, "play": str})
    if manifest.empty or manifest.image.duplicated().any():
        raise ValueError("Manifest must contain unique images.")
    if set(manifest.split) != set(SPLITS):
        raise ValueError("All three nonempty splits must be present.")
    if manifest.groupby("game").split.nunique().gt(1).any():
        raise ValueError("Game leakage between splits.")
    if manifest.groupby(["game", "play"]).split.nunique().gt(1).any():
        raise ValueError("Play leakage between splits.")
    config = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    if config.get("names") != {0: "helmet"} or config.get("nc") != 1:
        raise ValueError("Dataset must contain exactly class 0: helmet.")
    label_rows = 0
    for split in SPLITS:
        if config.get(split) != f"images/{split}":
            raise ValueError(f"Unexpected dataset YAML path for {split}.")
        rows = manifest.loc[manifest.split.eq(split)]
        actual_images = {p.name for p in (output / "images" / split).iterdir() if p.is_file()}
        actual_labels = {p.name for p in (output / "labels" / split).glob("*.txt")}
        if actual_images != set(rows.image) or actual_labels != {Path(n).stem + ".txt" for n in rows.image}:
            raise ValueError(f"Image/label inventory does not match the {split} manifest.")
        for row in rows.itertuples():
            image_path = output / "images" / split / row.image
            with Image.open(image_path) as image:
                if image.size != (row.width, row.height):
                    raise ValueError(f"Dimension mismatch: {image_path}")
                image.verify()
            if verify_hashes and sha256(image_path) != row.sha256:
                raise ValueError(f"Copied image checksum mismatch: {image_path}")
            labels = read_yolo_labels(output / "labels" / split / (Path(row.image).stem + ".txt"))
            if len(labels) != row.boxes:
                raise ValueError(f"Box-count mismatch: {row.image}")
            label_rows += len(labels)
    summary = {split: {"images": int((manifest.split == split).sum()),
                       "games": int(manifest.loc[manifest.split.eq(split), "game"].nunique()),
                       "plays": int(len(manifest.loc[manifest.split.eq(split), ["game", "play"]].drop_duplicates())),
                       "boxes": int(manifest.loc[manifest.split.eq(split), "boxes"].sum()),
                       "image_percent": round(100 * manifest.split.eq(split).mean(), 3)} for split in SPLITS}
    return {"status": "passed", "images": len(manifest), "annotations": label_rows,
            "game_overlap": 0, "play_overlap": 0, "splits": summary,
            "checks": ["image headers", "copied image SHA-256", "label inventories", "class IDs",
                       "normalized coordinates and edges", "per-image annotation counts", "group leakage"]}


def visualize_dataset(output, seed=42):
    """Display decoded, converted labels (not original CSV coordinates)."""
    output = Path(output)
    manifest = pd.read_csv(output / "manifest.csv")
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), layout="constrained")
    for split, row_axes in zip(SPLITS, axes):
        selected = manifest.loc[manifest.split.eq(split)].sample(
            min(3, int(manifest.split.eq(split).sum())), random_state=seed)
        for row, ax in zip(selected.itertuples(), row_axes):
            with Image.open(output / "images" / split / row.image) as image:
                rgb = np.array(image.convert("RGB"))
            labels = read_yolo_labels(output / "labels" / split / (Path(row.image).stem + ".txt"))
            boxes = labels[:, 1:] * np.array([row.width, row.height, row.width, row.height])
            boxes[:, :2] -= boxes[:, 2:] / 2
            frame = pd.DataFrame(boxes, columns=["left", "top", "width", "height"])
            draw_boxes(ax, rgb, frame, f"{split} · {row.image}\n{len(frame)} converted helmets")
        for ax in row_axes[len(selected):]:
            ax.axis("off")
    fig.suptitle("YOLO conversion verification · labels decoded back to pixel boxes", fontsize=16)
    path = output / "annotation_preview.jpg"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def prepare_dataset(csv_path, images_dir, output, seed=42):
    csv_path, images_dir, output = map(lambda p: Path(p).resolve(), (csv_path, images_dir, output))
    # Only generated data is written. An existing destination is never erased.
    if output == images_dir or output in images_dir.parents or images_dir in output.parents or output == csv_path.parent:
        raise ValueError("Output must not replace a raw dataset directory.")
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}. Use --validate-only or a new --output.")
    labels = pd.read_csv(csv_path)
    required = {"image", "label", "left", "top", "width", "height"}
    if required - set(labels):
        raise ValueError(f"Missing CSV columns: {sorted(required - set(labels))}")
    if labels.empty or labels[list(required)].isna().any().any():
        raise ValueError("Annotation CSV is empty or has missing required values.")
    unknown = set(labels.label) - HELMET_LABELS
    if unknown:
        raise ValueError(f"Unexpected labels (not silently converted to helmets): {sorted(unknown)}")
    if labels.duplicated().any():
        raise ValueError("Duplicate annotation rows found; review before conversion.")
    index, duplicates = media_index(images_dir, {".jpg", ".jpeg", ".png"})
    if duplicates:
        raise ValueError(f"Ambiguous image basenames: {duplicates[:5]}")
    dimensions = image_dimensions(labels.image.unique(), index)
    unavailable = dimensions.loc[dimensions.status.ne("available"), "image"].tolist()
    if unavailable:
        raise FileNotFoundError(f"Missing or unreadable labeled images ({len(unavailable)}): {unavailable[:5]}")
    dims = dimensions.set_index("image")
    records, converted = [], {}
    stems = set()
    for name, group in labels.groupby("image", sort=True):
        match = FILENAME.fullmatch(name)
        if not match:
            raise ValueError(f"Cannot derive game/play/view from filename: {name}")
        game, play, view = match.groups()
        if Path(name).stem in stems:
            raise ValueError(f"Multiple images share label-file stem: {name}")
        stems.add(Path(name).stem)
        width, height = int(dims.loc[name, "image_width"]), int(dims.loc[name, "image_height"])
        values = convert_boxes(group[["left", "top", "width", "height"]], width, height)
        converted[name] = values
        # Validate serialization itself, including pixel round-trip precision.
        serialized = np.array([[float(f"{v:.10f}") for v in row] for row in values])
        decoded = serialized * np.array([width, height, width, height])
        decoded[:, :2] -= decoded[:, 2:] / 2
        if not np.allclose(decoded, group[["left", "top", "width", "height"]], atol=1e-6, rtol=0):
            raise ValueError(f"Box serialization round trip failed: {name}")
        records.append({"image": name, "game": game, "play": play, "view": view,
                        "width": width, "height": height, "boxes": len(values)})
    manifest = pd.DataFrame(records)
    assignments = assign_game_splits(manifest.game, seed)
    manifest["split"] = manifest.game.map(assignments)
    output.mkdir(parents=True)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    hashes = []
    for i, row in enumerate(manifest.itertuples(), start=1):
        source = index[row.image]
        hashes.append(sha256(source))
        shutil.copy2(source, output / "images" / row.split / row.image)
        text = "\n".join("0 " + " ".join(f"{v:.10f}" for v in box) for box in converted[row.image]) + "\n"
        (output / "labels" / row.split / (Path(row.image).stem + ".txt")).write_text(text, encoding="utf-8")
        if i % 1000 == 0:
            print(f"Copied and labeled {i:,}/{len(manifest):,} images", flush=True)
    manifest["sha256"] = hashes
    manifest.to_csv(output / "manifest.csv", index=False)
    # Omitting 'path' makes Ultralytics resolve the root from the YAML's own directory.
    yaml_data = {split: f"images/{split}" for split in SPLITS}
    yaml_data.update(nc=1, names={0: "helmet"})
    (output / "dataset.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    provenance = {"seed": seed, "group_by": "game", "target_ratios": dict(zip(SPLITS, RATIOS.tolist())),
                  "annotation_csv_sha256": sha256(csv_path), "label_categories": labels.label.value_counts().to_dict(),
                  "bbox_policy": "strict: reject invalid/out-of-bounds boxes", "image_storage": "independent copies",
                  "ratio_note": "80/10/10 of games; exact image fractions vary with group sizes"}
    (output / "preparation.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    report = validate_dataset(output)
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    visualize_dataset(output, seed)
    return report


def main(argv=None):
    default_raw = ROOT / "data" / "raw" if (ROOT / "data" / "raw" / "image_labels.csv").exists() else ROOT / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(default_raw / "image_labels.csv"))
    parser.add_argument("--images-dir", default=str(default_raw / "images"))
    parser.add_argument("--output", default="data/processed/yolo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true", help="Recheck an existing dataset without rewriting labels/images.")
    args = parser.parse_args(argv)
    try:
        output = project_path(args.output)
        report = validate_dataset(output) if args.validate_only else prepare_dataset(
            project_path(args.csv), project_path(args.images_dir), output, args.seed)
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        parser.exit(2, f"Dataset preparation failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
