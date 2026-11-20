"""Explicit training entry point. --dry-run loads one batch without training."""
import argparse
import json
import shutil
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.impact.common import load_config, project_path, device_for, seed_everything
from src.impact.dataset import dataset_for
from src.impact.evaluate import predict_dataset, tune_threshold, binary_metrics
from src.impact.model import build_model


def backup_outputs(output, destination):
    """Update only this run's dedicated backup folder; publish complete copies."""
    for name in ("best.pt", "threshold.json", "history.json"):
        source = output / name
        if source.is_file():
            temporary = destination / f".{name}.{uuid.uuid4().hex}.tmp"
            shutil.copyfile(source, temporary)
            temporary.replace(destination / name)


def train(config, dry_run=False, backup_dir=None):
    seed_everything(config["seed"])
    settings = config["train"]
    device = device_for(settings["device"])
    train_ds, val_ds = dataset_for(config, "train"), dataset_for(config, "val", purpose="epoch")
    if train_ds.cache is None or not train_ds.rows.sample_id.isin(train_ds.cache.indices).all():
        raise ValueError("Training cache must cover all training clips. Enable cache.enabled and prepare the complete training cache.")
    for name, dataset in [("train", train_ds), ("val", val_ds)]:
        if set(dataset.rows.target) != {0, 1}:
            raise ValueError(f"{name} split requires both positive and negative examples")
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(train_ds, batch_size=settings["batch_size"], shuffle=True,
                        num_workers=settings["workers"], generator=generator,
                        pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=settings["batch_size"], shuffle=False,
                            num_workers=settings["workers"], pin_memory=device.type == "cuda",
                            persistent_workers=settings["workers"] > 0)
    print(json.dumps({"epoch_validation": {"clips": len(val_ds), "positives": int(val_ds.rows.target.eq(1).sum()),
                      "negatives": int(val_ds.rows.target.eq(0).sum()), "cached": len(val_ds), "mp4_fallback": False}}), flush=True)
    if dry_run:
        batch = next(iter(loader))
        val_batch = next(iter(val_loader))
        return {"train_clips": len(train_ds), "validation_clips": len(val_ds),
                "validation_batch_shape": list(val_batch["clip"].shape),
                "batch_shape": list(batch["clip"].shape), "device": str(device),
                "training_performed": False}
    output = project_path(config["paths"]["output"])
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new --output to preserve existing results.")
    output.mkdir(parents=True)
    backup = None
    if backup_dir:
        backup = project_path(backup_dir) / f"{output.name}-{uuid.uuid4().hex[:12]}"
        backup.mkdir(parents=True, exist_ok=False)
        print(f"Drive/file backup directory: {backup}", flush=True)
    model = build_model(config["model"]["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=.5)
    amp = settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(settings["pos_weight"], device=device), reduction="none")
    best, stale, history = -np.inf, 0, []
    for epoch in range(settings["epochs"]):
        model.train()
        total, count = 0., 0
        for batch in tqdm(loader, desc=f"Train {epoch+1}", mininterval=1):
            clips, targets, weights = (batch[key].to(device) for key in ["clip", "target", "weight"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                losses = loss_fn(model(clips).flatten(), targets)
                loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()*len(targets)
            count += len(targets)
        validation_timing = {}
        rows = predict_dataset(model, val_ds, device, settings["batch_size"], settings["workers"],
                               timings=validation_timing, loader=val_loader)
        threshold = tune_threshold(rows.target, rows.probability)
        metrics = binary_metrics(rows.target, rows.probability, threshold)
        history.append({"epoch": epoch+1, "loss": total/count, "threshold": threshold,
                        "validation_scope": "epoch_subset", "validation_timing": validation_timing, **metrics})
        scheduler.step(metrics["f1"])
        if metrics["f1"] > best:
            best, stale = metrics["f1"], 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch+1,
                        "threshold": threshold, "selected_on": "val_epoch", "metrics": metrics}, output / "best.pt")
            (output / "threshold.json").write_text(json.dumps({"threshold": threshold, "selected_on": "val_epoch",
                                                               "scope": "epoch_subset"}), encoding="utf-8")
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if backup is not None:
            backup_outputs(output, backup)
        print(json.dumps(history[-1]), flush=True)
        if stale >= settings["patience"]:
            break
    return {"best_validation_f1": best, "epochs": len(history)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/impact.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--val-negatives", type=int, help="Fixed epoch validation negatives (default 500); cache must cover selection")
    parser.add_argument("--backup-dir", help="Copy best checkpoint, history and threshold after each epoch into a unique run subfolder")
    parser.add_argument("--output", help="New local run directory; existing results are never overwritten")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.val_negatives is not None:
        cfg.setdefault("validation", {})["negatives"] = args.val_negatives
    if args.output:
        cfg["paths"]["output"] = args.output
    print(json.dumps(train(cfg, args.dry_run, args.backup_dir), indent=2))


if __name__ == "__main__":
    main()
