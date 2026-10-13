"""Shared path and configuration handling; importing this module never loads a model."""
from pathlib import Path
import json
import os

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/helmet_yolo.yaml"


def project_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def configure_runtime():
    # Keep framework settings local to this project, with no remote integrations.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts" / "ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def load_config(value=DEFAULT_CONFIG):
    path = project_path(value)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {"model", "data", "train"}:
        raise ValueError("Config must have exactly model, data, and train keys.")
    if not isinstance(config["train"], dict):
        raise ValueError("train must be a mapping of Ultralytics training settings.")
    return config


def dataset_path(value):
    path = project_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML missing: {path}. Run prepare_yolo_data first.")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("names") != {0: "helmet"} or config.get("nc") != 1:
        raise ValueError("Expected a single helmet class (0).")
    # Project-generated descriptors omit 'path'; root is the YAML parent.
    root = Path(config.get("path") or path.parent)
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    for split in ("train", "val", "test"):
        relative = config.get(split)
        if not isinstance(relative, str) or not (root / relative).is_dir():
            raise FileNotFoundError(f"Missing {split} image directory in {path}.")
    return path


def model_source(value):
    path = project_path(value)
    if path.is_file():
        return str(path)
    # Only named official weights can be fetched at the explicit training entry point.
    if value == "yolo11n.pt":
        return value
    raise FileNotFoundError(f"Model file not found: {path}. Set model to yolo11n.pt or an existing file.")


def checkpoint_path(value):
    path = project_path(value)
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise FileNotFoundError(f"An existing .pt checkpoint is required: {path}")
    return path


def require_helmet_checkpoint(model):
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != {0: "helmet"}:
        raise ValueError(f"Expected a trained single-class helmet checkpoint, got class names: {names}")


def resolve_device(value, require_gpu=False):
    import torch
    if value == "auto":
        value = "0" if torch.cuda.is_available() else "cpu"
    if require_gpu and (str(value) in {"cpu", "mps"} or not torch.cuda.is_available()):
        raise RuntimeError("A CUDA GPU is required but unavailable to PyTorch. Check the CUDA wheel and driver.")
    return value


def training_plan(config, overrides=None, require_gpu=False):
    configure_runtime()
    settings = dict(config["train"])
    settings.update({k: v for k, v in (overrides or {}).items() if v is not None})
    settings["device"] = resolve_device(settings.get("device", "auto"), require_gpu)
    settings["project"] = str(project_path(settings.get("project", "runs/detect")))
    settings["data"] = str(dataset_path(config["data"]))
    model = model_source(config["model"])
    from ultralytics.cfg import get_cfg
    get_cfg(overrides={**settings, "model": model, "task": "detect", "mode": "train"})
    if settings.get("epochs", 0) <= 0 or settings.get("imgsz", 0) <= 0:
        raise ValueError("epochs and imgsz must be positive.")
    batch = settings.get("batch", 8)
    if batch != -1 and batch <= 0:
        raise ValueError("batch must be positive or -1 for Ultralytics auto-batch.")
    return {"model": model, "settings": settings,
            "weights_download_needed": not project_path(config["model"]).is_file(),
            "operation": "train only when invoked without --dry-run"}


def print_json(value):
    print(json.dumps(value, indent=2, default=str))
