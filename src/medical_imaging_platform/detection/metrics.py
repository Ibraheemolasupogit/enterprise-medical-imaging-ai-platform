"""Detection-specific metrics for 2D synthetic lesion boxes."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def box_iou_xyxy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for half-open xyxy boxes."""
    a = np.asarray(first, dtype=np.float32).reshape((-1, 4))
    b = np.asarray(second, dtype=np.float32).reshape((-1, 4))
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    left_top = np.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(right_bottom - left_top, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    result = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return cast(np.ndarray, result)


def compute_detection_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    *,
    iou_thresholds: tuple[float, ...] = (0.5, 0.75),
) -> dict[str, Any]:
    """Compute deterministic AP/precision/recall for one-class detection."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have equal length.")
    image_count = len(ground_truths)
    positive_images = sum(1 for gt in ground_truths if len(_boxes(gt)) > 0)
    negative_images = image_count - positive_images
    gt_box_count = sum(len(_boxes(gt)) for gt in ground_truths)
    pred_box_count = sum(len(_boxes(pred)) for pred in predictions)
    threshold_results = {
        f"{threshold:.2f}": _metrics_at_threshold(predictions, ground_truths, threshold)
        for threshold in iou_thresholds
    }
    ap_values = [result["average_precision"] for result in threshold_results.values()]
    primary = threshold_results[f"{iou_thresholds[0]:.2f}"]
    return {
        "image_count": image_count,
        "positive_image_count": positive_images,
        "negative_image_count": negative_images,
        "ground_truth_box_count": gt_box_count,
        "predicted_box_count": pred_box_count,
        "precision": primary["precision"],
        "recall": primary["recall"],
        "ap50": threshold_results.get("0.50", primary)["average_precision"],
        "ap75": threshold_results.get("0.75", primary)["average_precision"],
        "map": float(np.mean(ap_values)) if ap_values else 0.0,
        "false_positives_per_image": primary["false_positives"] / max(1, image_count),
        "iou_thresholds": threshold_results,
    }


def _metrics_at_threshold(
    predictions: list[dict[str, Any]], ground_truths: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    scored: list[tuple[float, int, np.ndarray]] = []
    total_gt = 0
    for image_index, (prediction, ground_truth) in enumerate(
        zip(predictions, ground_truths, strict=True)
    ):
        gt_boxes = _boxes(ground_truth)
        total_gt += len(gt_boxes)
        pred_boxes = _boxes(prediction)
        scores = _scores(prediction, len(pred_boxes))
        for score, box in sorted(
            zip(scores, pred_boxes, strict=True),
            key=lambda item: (-float(item[0]), item[1].tolist()),
        ):
            scored.append((float(score), image_index, box.astype(np.float32)))
    matched: dict[int, set[int]] = {index: set() for index in range(len(ground_truths))}
    tp: list[int] = []
    fp: list[int] = []
    for _, image_index, box in sorted(scored, key=lambda item: -item[0]):
        gt_boxes = _boxes(ground_truths[image_index])
        if gt_boxes.size == 0:
            tp.append(0)
            fp.append(1)
            continue
        ious = box_iou_xyxy(box.reshape(1, 4), gt_boxes).reshape(-1)
        order = np.argsort(-ious)
        selected = None
        for gt_index in order.tolist():
            if float(ious[gt_index]) >= threshold and gt_index not in matched[image_index]:
                selected = int(gt_index)
                break
        if selected is None:
            tp.append(0)
            fp.append(1)
        else:
            matched[image_index].add(selected)
            tp.append(1)
            fp.append(0)
    tp_cum = np.cumsum(np.asarray(tp, dtype=np.float32))
    fp_cum = np.cumsum(np.asarray(fp, dtype=np.float32))
    if total_gt == 0:
        precision = 1.0 if not scored else 0.0
        recall = 1.0 if not scored else 0.0
        average_precision = precision
    elif not scored:
        precision = 0.0
        recall = 0.0
        average_precision = 0.0
    else:
        precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
        recall_curve = tp_cum / float(total_gt)
        precision = float(precision_curve[-1])
        recall = float(recall_curve[-1])
        average_precision = _average_precision(recall_curve, precision_curve)
    return {
        "iou_threshold": threshold,
        "true_positives": int(sum(tp)),
        "false_positives": int(sum(fp)),
        "false_negatives": int(total_gt - sum(tp)),
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision,
    }


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def _boxes(payload: dict[str, Any]) -> np.ndarray:
    boxes = np.asarray(payload.get("boxes", []), dtype=np.float32)
    return boxes.reshape((-1, 4))


def _scores(payload: dict[str, Any], count: int) -> np.ndarray:
    scores = np.asarray(payload.get("scores", np.ones(count, dtype=np.float32)), dtype=np.float32)
    return scores.reshape((-1,))
