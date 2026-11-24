"""Bounded diagnostic benchmarks; --compute explicitly enables disposable steps.

Never loads pretrained weights, saves checkpoints, or runs a full epoch.
"""
import argparse
import copy
import json
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.impact.common import load_config, project_path, seed_everything, device_for
from src.impact.dataset import dataset_for, read_frames, clip_tensor
from src.impact.cache import sequential_clips
from src.impact.model import build_model


def timed(function, device=None):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = function()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter()-start


def summarize(records):
    return {key: {"mean_seconds": float(np.mean(values)), "p50_seconds": float(np.median(values)),
                  "p95_seconds": float(np.percentile(values, 95))}
            for key, values in records.items()}


def benchmark(config, cache_path=None, steps=8, warmup=2, samples=16, compute=False):
    if min(steps, samples) < 1 or warmup < 0:
        raise ValueError("Invalid benchmark lengths")
    cfg = copy.deepcopy(config)
    cfg["cache"] = {"enabled": False}
    seed_everything(cfg["seed"])
    raw = dataset_for(cfg, "train")
    indices = np.random.default_rng(cfg["seed"]).choice(len(raw), min(samples, len(raw)), replace=False).tolist()
    io, preprocessing = [], []
    for index in indices:
        row = raw.rows.iloc[index]
        # Raw clip includes alignment; read_frames separately isolates actual I/O.
        _, elapsed = timed(lambda: read_frames(raw.videos / row.video, int(row.frame), cfg["data"]["boundary"]))
        io.append(elapsed)
        frames, boxes, _ = raw.raw_clip(index)
        _, elapsed = timed(lambda: clip_tensor(frames, boxes, cfg["data"]["size"], cfg["data"]["context"]))
        preprocessing.append(elapsed)
    # Fair end-to-end crop comparison on the same centers, without normalization.
    start = time.perf_counter()
    for index in indices:
        frames, boxes, _ = raw.raw_clip(index)
        from src.impact.dataset import crop_rgb
        np.stack([crop_rgb(f, b, cfg["data"]["size"], cfg["data"]["context"]) for f, b in zip(frames, boxes)])
    random_crop_seconds = time.perf_counter()-start
    start = time.perf_counter()
    for _ in sequential_clips(raw, raw.rows.loc[indices]):
        pass
    sequential_seconds = time.perf_counter()-start
    device = device_for(cfg["train"]["device"])
    report = {"environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "samples": len(indices), "io_preprocessing": summarize({"video_open_seek_decode": io, "crop_normalize": preprocessing}),
              "sample_indices": indices, "batch_size": cfg["train"]["batch_size"],
              "workers": cfg["train"]["workers"], "amp": cfg["train"]["amp"] and device.type == "cuda",
              "random_crop_seconds": random_crop_seconds, "sequential_crop_seconds": sequential_seconds,
              "probe_note": "Sequential scan includes intervening frames; sparse small probes may favor seeking.",
              "compute_enabled": compute, "full_training_performed": False, "modes": {}}
    modes = {"uncached": raw}
    if cache_path:
        cached_cfg = copy.deepcopy(cfg)
        cached_cfg["cache"] = {"enabled": True, "path": str(cache_path), "validation_subset": False}
        cached = dataset_for(cached_cfg, "train")
        if not cached.rows.sample_id.isin(cached.cache.indices).all():
            raise ValueError("Training benchmark requires a complete training cache, not a smoke cache")
        modes["cached"] = cached
    # Same shuffled sample order, seed, initial model and settings for each mode.
    order = np.random.default_rng(cfg["seed"]).permutation(len(raw)).tolist()
    for name, dataset in modes.items():
        seed_everything(cfg["seed"])
        loader = DataLoader(Subset(dataset, order), batch_size=cfg["train"]["batch_size"],
                            num_workers=cfg["train"]["workers"], pin_memory=device.type == "cuda")
        if compute and warmup+steps >= len(loader):
            raise ValueError("Compute diagnostics must be shorter than one epoch; reduce steps/warmup")
        iterator, startup = timed(lambda: iter(loader))
        stages = {key: [] for key in ["batch_wait", "transfer", "forward_loss", "backward", "optimizer", "total"]}
        if compute:
            model = build_model(False).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
            amp = cfg["train"]["amp"] and device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg["train"]["pos_weight"], device=device), reduction="none")
        measured = 0
        for iteration in range(warmup+steps):
            try:
                batch, wait = timed(lambda: next(iterator), device)
            except StopIteration:
                break
            values, transfer = timed(lambda: [batch[k].to(device) for k in ["clip", "target", "weight"]], device)
            forward = backward = update = 0.
            if compute:
                clips, targets, weights = values
                optimizer.zero_grad(set_to_none=True)
                def run_forward():
                    with torch.autocast(device_type=device.type, enabled=amp):
                        return (loss_fn(model(clips).flatten(), targets)*weights).sum()/weights.sum().clamp_min(1e-8)
                loss, forward = timed(run_forward, device)
                _, backward = timed(lambda: scaler.scale(loss).backward(), device)
                def step():
                    scaler.step(optimizer)
                    scaler.update()
                _, update = timed(step, device)
                del loss
            if iteration >= warmup:
                measured += 1
                for key, value in zip(stages, [wait, transfer, forward, backward, update, wait+transfer+forward+backward+update]):
                    stages[key].append(value)
        if not measured:
            raise ValueError("No measured batches; lower warmup or batch size")
        report["modes"][name] = {"stages": summarize(stages), "batches": measured, "iterator_startup_seconds": startup,
                                  "estimated_train_epoch_seconds": float(np.mean(stages["total"]))*len(loader),
                                  "estimate_note": "Excludes validation, checkpointing and startup; synchronized phases serialize GPU work."}
        del iterator, loader
        if compute:
            del model, optimizer, scaler
    if "cached" in report["modes"]:
        report["batch_speedup"] = report["modes"]["uncached"]["stages"]["total"]["mean_seconds"] / report["modes"]["cached"]["stages"]["total"]["mean_seconds"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/impact.yaml")
    parser.add_argument("--cache")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--compute", action="store_true", help="Run bounded disposable forward/backward/AdamW steps")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = project_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = benchmark(load_config(args.config), args.cache, args.steps, args.warmup, args.samples, args.compute)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
