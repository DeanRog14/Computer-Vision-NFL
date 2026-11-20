"""Configuration and runtime helpers shared with the detector where possible."""
import random

import numpy as np
import torch
import yaml

from src.detection.common import project_path


def load_config(path="configs/impact.yaml"):
    config = yaml.safe_load(project_path(path).read_text(encoding="utf-8"))
    if config["data"]["frames"] != 9:
        raise ValueError("This pipeline requires exactly nine frames.")
    if config["data"]["size"] < 16 or config["data"]["context"] <= 0:
        raise ValueError("Invalid crop dimensions.")
    if config["data"]["boundary"] not in {"repeat", "error"}:
        raise ValueError("boundary must be repeat or error")
    if config["data"]["missing_position"] not in {"center", "nearest"}:
        raise ValueError("missing_position must be center or nearest")
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(value):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
