"""Save annotated helmet predictions from images, directories, videos, or one video frame."""
from pathlib import Path
import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.detection.common import (DEFAULT_CONFIG, load_config, checkpoint_path, project_path,
                                  configure_runtime, resolve_device, print_json, require_helmet_checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True, help="Existing image, image directory, or video; paths are repository-relative.")
    parser.add_argument("--frame", type=int, help="One-based CSV frame number for a video (omit to predict the whole video).")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default="runs/detect/helmet_predictions")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Check paths/options only; never load weights or run inference.")
    args = parser.parse_args(argv)
    try:
        checkpoint = checkpoint_path(args.checkpoint)
        source = project_path(args.source)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        if args.frame is not None and (not source.is_file() or source.suffix.lower() not in {".mp4", ".avi", ".mov"} or args.frame < 1):
            raise ValueError("--frame requires a video file and a positive one-based frame number.")
        if not 0 <= args.conf <= 1 or args.vid_stride < 1:
            raise ValueError("conf must be in [0, 1] and vid-stride must be positive.")
        output = project_path(args.output)
        if output.exists():
            raise FileExistsError(f"Output already exists; choose a new --output: {output}")
        config = load_config(args.config)
        configure_runtime()
        device = resolve_device(args.device)
        imgsz = args.imgsz if args.imgsz is not None else config["train"]["imgsz"]
        settings = dict(imgsz=imgsz, device=device, conf=args.conf, vid_stride=args.vid_stride,
                        save=True, save_txt=True, save_conf=True, project=str(output.parent), name=output.name, exist_ok=True)
        from ultralytics.cfg import get_cfg
        get_cfg(overrides={**settings, "task": "detect", "mode": "predict"})
        print_json({"checkpoint": checkpoint, "source": source, "frame": args.frame, "settings": settings})
        if args.dry_run:
            return
        from ultralytics import YOLO
        if args.frame is not None:
            import cv2
            from src.eda import read_video_frame
            rgb = read_video_frame(source.name, args.frame, {source.name: source})
            if rgb is None:
                raise ValueError(f"Cannot decode frame {args.frame} in {source}")
            prediction_source = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            prediction_source = str(source)
        model = YOLO(str(checkpoint), task="detect")
        require_helmet_checkpoint(model)
        detections, items = 0, 0
        # Stream results so complete videos do not accumulate in memory.
        for result in model.predict(source=prediction_source, stream=True, **settings):
            items += 1
            detections += len(result.boxes) if result.boxes is not None else 0
        print_json({"items_processed": items, "helmet_detections": detections, "output": output})
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"Prediction setup failed: {error}\n")


if __name__ == "__main__":
    main()
