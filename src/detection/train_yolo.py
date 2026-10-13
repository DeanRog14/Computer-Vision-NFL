"""Train the helmet detector only when this entry point is explicitly executed."""
from pathlib import Path
import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.detection.common import DEFAULT_CONFIG, load_config, training_plan, print_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="auto, cpu, or GPU index, e.g. 0")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--require-gpu", action="store_true", help="Fail before loading weights if CUDA is unavailable.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/settings; never load weights or start training.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        plan = training_plan(config, {key: getattr(args, key) for key in ["device", "batch", "imgsz", "epochs"]}, args.require_gpu)
        print_json(plan)
        if args.dry_run:
            return
        # Model construction/download and training exist only below the explicit dry-run guard.
        from ultralytics import YOLO
        model = YOLO(plan["model"], task="detect")
        model.train(**plan["settings"])
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"Training setup failed: {error}\n")


if __name__ == "__main__":
    main()
