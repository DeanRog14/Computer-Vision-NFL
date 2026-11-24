"""Independent clip metrics and one-to-one spatiotemporal event metrics."""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.impact.common import load_config, project_path, device_for
from src.impact.dataset import BOX, dataset_for
from src.impact.model import load_model


def binary_metrics(target, probability, threshold):
    tn, fp, fn, tp = confusion_matrix(target, np.asarray(probability) >= threshold, labels=[0, 1]).ravel()
    precision = tp / max(tp+fp, 1)
    recall = tp / max(tp+fn, 1)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(2*tp/max(2*tp+fp+fn, 1)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def tune_threshold(target, probability):
    if set(np.asarray(target).tolist()) != {0, 1}:
        raise ValueError("Validation threshold selection requires both classes")
    precision, recall, thresholds = precision_recall_curve(target, probability)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
    # Prefer the higher threshold on equal scores.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(intersection / max(a[2:].prod()+b[2:].prod()-intersection, 1e-12))


def group_predictions(rows, threshold=.5, gap=4, min_iou=.35):
    """Score-ordered temporal NMS; retains actual observed boxes and frames.

    Compare directly to retained peaks rather than transitively chaining a long
    sequence. Camera views are never merged. This is postprocessing, not GT.
    """
    kept = []
    for _, row in rows.loc[rows.probability.ge(threshold)].sort_values("probability", ascending=False).iterrows():
        if not any(row.video == other.video and abs(row.frame-other.frame) <= gap
                   and iou(row[BOX], other[BOX]) >= min_iou for other in kept):
            kept.append(row)
    return pd.DataFrame(kept, columns=rows.columns).reset_index(drop=True)


def event_metrics(predictions, ground_truth, tolerance=4, min_iou=.35):
    """Maximum-cardinality assignment across same-video, +/-4, IoU>=.35 edges.

    Definitive annotation rows are the competition targets; event_id does NOT
    collapse GT rows for this score. Keeping targets separate avoids silently
    changing the competition metric when annotations occur on adjacent frames.
    """
    tp = 0
    for video, gt in ground_truth.groupby("video"):
        pred = predictions.loc[predictions.video.eq(video)]
        if pred.empty:
            continue
        valid = np.zeros((len(gt), len(pred)), dtype=bool)
        for i, (_, g) in enumerate(gt.iterrows()):
            for j, (_, p) in enumerate(pred.iterrows()):
                valid[i, j] = abs(int(g.frame)-int(p.frame)) <= tolerance and iou(g[BOX], p[BOX]) >= min_iou
        a, b = linear_sum_assignment(~valid)
        tp += int(valid[a, b].sum())
    fp, fn = len(predictions)-tp, len(ground_truth)-tp
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp/max(tp+fp, 1),
            "recall": tp/max(tp+fn, 1), "f1": 2*tp/max(2*tp+fp+fn, 1)}


@torch.inference_mode()
def predict_dataset(model, dataset, device, batch_size=8, workers=0, timings=None, loader=None, progress=True):
    model.eval()
    device = torch.device(device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    measured = {"loading_seconds": 0., "transfer_seconds": 0., "inference_seconds": 0., "output_seconds": 0.}
    started = time.perf_counter()
    rows = dataset.rows.copy()
    probabilities = np.zeros(len(rows), dtype=float)
    if loader is None:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
                            pin_memory=device.type == "cuda")
    start = time.perf_counter()
    iterator = iter(loader)
    measured["loading_seconds"] += time.perf_counter()-start
    for _ in tqdm(range(len(loader)), desc="Validation", disable=not progress, mininterval=1):
        start = time.perf_counter()
        batch = next(iterator)
        measured["loading_seconds"] += time.perf_counter()-start
        sync()
        start = time.perf_counter()
        clips = batch["clip"].to(device, non_blocking=device.type == "cuda")
        sync()
        measured["transfer_seconds"] += time.perf_counter()-start
        start = time.perf_counter()
        scores = model(clips).flatten().sigmoid()
        sync()
        measured["inference_seconds"] += time.perf_counter()-start
        start = time.perf_counter()
        scores = scores.cpu().numpy()
        probabilities[np.asarray(batch["index"])] = scores
        measured["output_seconds"] += time.perf_counter()-start
    rows["probability"] = probabilities
    measured.update(total_seconds=time.perf_counter()-started, clips=len(rows), batches=len(loader),
                    device=str(device), scope="full_or_caller_selected_rows")
    cache = getattr(dataset, "cache", None)
    measured["cached_clips"] = int(rows.sample_id.isin(cache.indices).sum()) if cache is not None else 0
    measured["uncached_clips"] = len(rows)-measured["cached_clips"]
    if timings is not None:
        timings.update(measured)
    rows.attrs["timing"] = measured
    return rows


def save_clip_report(rows, threshold, output):
    result = binary_metrics(rows.target, rows.probability, threshold)
    result["threshold"] = threshold
    precision, recall, thresholds = precision_recall_curve(rows.target, rows.probability)
    pd.DataFrame({"precision": precision, "recall": recall,
                  "threshold": np.r_[thresholds, np.nan]}).to_csv(output / "precision_recall.csv", index=False)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    fig.savefig(output / "precision_recall.png", dpi=150)
    plt.close(fig)
    rows.to_csv(output / "clip_predictions.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/impact.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold-file", help="Validation threshold JSON; required for test")
    parser.add_argument("--event-predictions", help="Optional end-to-end detection CSV (probability column)")
    parser.add_argument("--epoch-validation", action="store_true", help="Diagnose/timestamp the cached epoch subset only; not final threshold selection")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.split == "test" and not args.threshold_file:
        parser.error("Test requires a threshold selected on validation")
    if args.epoch_validation and args.split != "val":
        parser.error("--epoch-validation is only valid with --split val")
    selection = None
    if args.threshold_file:
        selection = json.loads(project_path(args.threshold_file).read_text())
        if selection.get("selected_on") != "val" or selection.get("scope", "full") != "full":
            raise ValueError("Final thresholds must be selected on full validation, not the epoch subset")
    output = project_path(args.output)
    if output.exists():
        raise FileExistsError(output)
    ds = dataset_for(cfg, args.split, purpose="epoch" if args.epoch_validation else "full")
    device = device_for(cfg["train"]["device"])
    model, saved = load_model(project_path(args.checkpoint), device)
    if saved["config"]["data"] != cfg["data"]:
        raise ValueError("Evaluation preprocessing differs from training checkpoint")
    rows = predict_dataset(model, ds, device, cfg["train"]["batch_size"], cfg["train"]["workers"])
    if selection is not None:
        threshold = float(selection["threshold"])
    else:
        threshold = tune_threshold(rows.target, rows.probability)
    output.mkdir(parents=True)
    scope = "epoch_subset" if args.epoch_validation else "full"
    result = {"clip": save_clip_report(rows, threshold, output), "timing": rows.attrs["timing"], "scope": scope}
    (output / "threshold.json").write_text(json.dumps({"threshold": threshold,
        "selected_on": "val_epoch" if args.epoch_validation else "val", "scope": scope}), encoding="utf-8")
    predictions = rows
    event_kind = "classifier_with_ground_truth_boxes"
    if args.event_predictions:
        predictions = pd.read_csv(project_path(args.event_predictions))
        if set(predictions.video) - set(ds.rows.video):
            raise ValueError("Event predictions include videos outside the selected split")
        event_kind = "end_to_end_with_predicted_boxes"
    events = group_predictions(predictions, threshold, cfg["inference"]["event_gap"], cfg["inference"]["event_iou"])
    result["event"] = {"pipeline": event_kind, **event_metrics(events, rows.loc[rows.target.eq(1)])}
    events.to_csv(output / "events.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
