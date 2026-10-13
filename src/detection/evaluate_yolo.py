"""Evaluate an existing detector checkpoint on the held-out test set."""
from pathlib import Path
import argparse
import json
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml

from src.detection.common import (DEFAULT_CONFIG, load_config, dataset_path, checkpoint_path,
                                  project_path, configure_runtime, resolve_device, print_json,
                                  require_helmet_checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="An existing trained .pt checkpoint (never auto-downloaded).")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default="runs/detect/helmet_test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="Check existing paths; do not load or evaluate a model.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        checkpoint = checkpoint_path(args.checkpoint)
        data = dataset_path(config["data"])
        output = project_path(args.output)
        if output.exists():
            raise FileExistsError(f"Output already exists; choose a new --output: {output}")
        if args.examples < 0:
            raise ValueError("examples must be nonnegative")
        configure_runtime()
        device = resolve_device(args.device)
        imgsz = args.imgsz if args.imgsz is not None else config["train"]["imgsz"]
        batch = args.batch if args.batch is not None else config["train"]["batch"]
        settings = dict(data=str(data), split="test", imgsz=imgsz, batch=batch, device=device,
                        plots=True, project=str(output.parent), name=output.name, exist_ok=True)
        from ultralytics.cfg import get_cfg
        get_cfg(overrides={**settings, "task": "detect", "mode": "val"})
        print_json({"checkpoint": checkpoint, "settings": settings})
        if args.dry_run:
            return
        from ultralytics import YOLO
        model = YOLO(str(checkpoint), task="detect")
        require_helmet_checkpoint(model)
        metrics = model.val(**settings)
        summary = {"split": "test", "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                   "mAP@0.5": float(metrics.box.map50), "mAP@0.5:0.95": float(metrics.box.map),
                   "checkpoint": str(checkpoint), "data": str(data)}
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if args.examples:
            descriptor = yaml.safe_load(data.read_text(encoding="utf-8"))
            test_dir = data.parent / descriptor["test"]
            images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            sample = np.random.default_rng(config["train"]["seed"]).choice(len(images), min(args.examples, len(images)), replace=False)
            for _ in model.predict(source=[str(images[i]) for i in sample], imgsz=imgsz, device=device,
                                   conf=.25, save=True, stream=True, project=str(output), name="prediction_examples", exist_ok=True):
                pass
        print_json(summary)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"Evaluation setup failed: {error}\n")


if __name__ == "__main__":
    main()
