"""Detection model factory and common adapters."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from medical_imaging_platform.detection.metrics import box_iou_xyxy
from medical_imaging_platform.detection.models import DetectionConfig


class DetectionModelError(RuntimeError):
    """Raised when a requested detector cannot be built."""


class TinyDetectionBackbone(nn.Module):  # type: ignore[misc]
    """Small 1-channel 2D CNN backbone for CPU smoke detection."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = hidden_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchvisionFasterRCNNAdapter(nn.Module):  # type: ignore[misc]
    """Torchvision FasterRCNN wrapper producing the common prediction contract."""

    detector_family = "faster_rcnn"
    uses_torchvision = True

    @staticmethod
    def _prepare_images(images: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            image.squeeze(0).mean(dim=0, keepdim=True)
            if image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 3
            else image.mean(dim=0, keepdim=True)
            if image.ndim == 3 and image.shape[0] == 3
            else image
            for image in images
        ]

    def __init__(self, config: DetectionConfig) -> None:
        super().__init__()
        try:
            from torchvision.models.detection import FasterRCNN  # type: ignore[import-untyped]
            from torchvision.models.detection.anchor_utils import (  # type: ignore[import-untyped]
                AnchorGenerator,
            )
            from torchvision.ops import MultiScaleRoIAlign  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - exercised when dependency absent
            raise DetectionModelError(
                "torchvision is required for the Faster R-CNN detector. Install the "
                "repository dependency set; no pretrained weights are downloaded."
            ) from exc
        self.config = config
        backbone = TinyDetectionBackbone(config.hidden_channels)
        anchor_generator = AnchorGenerator(
            sizes=(tuple(config.faster_rcnn_anchor_sizes),),
            aspect_ratios=(tuple(config.faster_rcnn_aspect_ratios),),
        )
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=3, sampling_ratio=1)
        self.model = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=config.image_shape[0],
            max_size=max(config.image_shape),
            box_score_thresh=config.score_threshold,
            box_nms_thresh=config.nms_iou_threshold,
            image_mean=[0.0],
            image_std=[1.0],
        )

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ) -> object:
        images = [
            image.mean(dim=0, keepdim=True) if image.shape[0] == 3 else image for image in images
        ]
        return self.model(self._prepare_images(images), targets)

    def predict(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        self.eval()
        with torch.no_grad():
            output = self.model(self._prepare_images(images))
        if not isinstance(output, list):
            raise DetectionModelError("Faster R-CNN inference output must be a list.")
        return output

    def loss(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        self.train()
        output = self.model(self._prepare_images(images), targets)
        if not isinstance(output, dict):
            raise DetectionModelError("Faster R-CNN training output must be a loss dictionary.")
        return output


class TinyDETRDetector(nn.Module):  # type: ignore[misc]
    """Native tiny DETR-style detector for one synthetic lesion class plus no-object."""

    detector_family = "tiny_detr"
    uses_torchvision = False

    def __init__(self, config: DetectionConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = TinyDetectionBackbone(config.hidden_channels)
        self.input_proj = nn.Conv2d(self.backbone.out_channels, config.embedding_dim, 1)
        layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embedding_dim * config.mlp_ratio),
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.transformer_depth)
        self.query_embed = nn.Embedding(config.num_queries, config.embedding_dim)
        self.class_head = nn.Linear(config.embedding_dim, 2)
        self.box_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.embedding_dim, 4),
        )

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        batch = torch.stack(images)
        features = self.input_proj(self.backbone(batch))
        memory = features.flatten(2).transpose(1, 2)
        memory = memory + _positional_encoding(memory.shape[1], memory.shape[2], memory.device)
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch.shape[0], 1, 1)
        decoded = self.decoder(queries, memory)
        logits = self.class_head(decoded)
        boxes = torch.sigmoid(self.box_head(decoded))
        if targets is not None and self.training:
            return self.loss_from_outputs(logits, boxes, targets)
        return self.predictions_from_outputs(logits, boxes)

    def loss(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        output = self.forward(images, targets)
        if not isinstance(output, dict):
            raise DetectionModelError("Tiny DETR training output must be a loss dictionary.")
        return output

    def predict(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
        if not isinstance(output, list):
            raise DetectionModelError("Tiny DETR inference output must be predictions.")
        return output

    def loss_from_outputs(
        self, logits: torch.Tensor, boxes_norm: torch.Tensor, targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        cls_losses = []
        box_losses = []
        iou_losses = []
        for index, target in enumerate(targets):
            target_boxes = target["boxes"].to(logits.device)
            target_norm = _xyxy_to_cxcywh_norm(target_boxes, self.config.image_shape)
            assignment = _match_queries(logits[index], boxes_norm[index], target_norm)
            labels = torch.zeros((self.config.num_queries,), dtype=torch.long, device=logits.device)
            for query_index, target_index in assignment:
                labels[query_index] = 1
                pred = boxes_norm[index, query_index : query_index + 1]
                expected = target_norm[target_index : target_index + 1]
                box_losses.append(functional.l1_loss(pred, expected))
                pred_xyxy = _cxcywh_norm_to_xyxy(pred, self.config.image_shape)
                expected_xyxy = target_boxes[target_index : target_index + 1].to(logits.device)
                iou = _torch_box_iou(pred_xyxy, expected_xyxy)
                iou_losses.append(1.0 - iou.reshape(-1)[0])
            weights = torch.tensor(
                [self.config.no_object_weight, 1.0], dtype=torch.float32, device=logits.device
            )
            cls_losses.append(functional.cross_entropy(logits[index], labels, weight=weights))
        cls_loss = torch.stack(cls_losses).mean()
        box_loss = torch.stack(box_losses).mean() if box_losses else cls_loss * 0.0
        iou_loss = torch.stack(iou_losses).mean() if iou_losses else cls_loss * 0.0
        return {
            "loss_classifier": cls_loss,
            "loss_box_reg": self.config.box_loss_weight * box_loss,
            "loss_iou": self.config.iou_loss_weight * iou_loss,
        }

    def predictions_from_outputs(
        self, logits: torch.Tensor, boxes_norm: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        probabilities = torch.softmax(logits, dim=-1)[..., 1]
        boxes = _cxcywh_norm_to_xyxy(boxes_norm, self.config.image_shape)
        outputs = []
        for sample_index in range(logits.shape[0]):
            scores = probabilities[sample_index]
            keep = scores >= self.config.score_threshold
            kept_boxes = boxes[sample_index][keep]
            kept_scores = scores[keep]
            if kept_boxes.numel():
                kept = _nms(kept_boxes, kept_scores, self.config.nms_iou_threshold)
                kept_boxes = kept_boxes[kept]
                kept_scores = kept_scores[kept]
            outputs.append(
                {
                    "boxes": kept_boxes.reshape((-1, 4)),
                    "labels": torch.ones(
                        (kept_boxes.shape[0],), dtype=torch.int64, device=logits.device
                    ),
                    "scores": kept_scores.reshape((-1,)),
                }
            )
        return outputs


def build_detector(config: DetectionConfig) -> nn.Module:
    if config.model_family == "tiny_detr":
        return TinyDETRDetector(config)
    if config.model_family == "faster_rcnn":
        return TorchvisionFasterRCNNAdapter(config)
    raise ValueError(f"Unsupported detection model family: {config.model_family}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def architecture_metadata(
    config: DetectionConfig, model: nn.Module | None = None
) -> dict[str, Any]:
    uses_torchvision = bool(
        getattr(model, "uses_torchvision", config.model_family == "faster_rcnn")
    )
    return {
        "model_family": config.model_family,
        "image_shape": config.image_shape,
        "coordinate_semantics": "2D half-open xyxy boxes on axial slices",
        "uses_torchvision": uses_torchvision,
        "pretrained_weights": None,
        "weights": None,
        "weights_backbone": None,
        "hidden_channels": config.hidden_channels,
        "embedding_dim": config.embedding_dim,
        "transformer_depth": config.transformer_depth
        if config.model_family == "tiny_detr"
        else None,
        "num_heads": config.num_heads if config.model_family == "tiny_detr" else None,
        "num_queries": config.num_queries if config.model_family == "tiny_detr" else None,
        "anchor_sizes": config.faster_rcnn_anchor_sizes
        if config.model_family == "faster_rcnn"
        else None,
    }


def predictions_to_common(predictions: list[dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    common = []
    for prediction in predictions:
        common.append(
            {
                "boxes": prediction["boxes"].detach().cpu().numpy().astype(float).tolist(),
                "labels": prediction["labels"].detach().cpu().numpy().astype(int).tolist(),
                "scores": prediction["scores"].detach().cpu().numpy().astype(float).tolist(),
            }
        )
    return common


def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape((0, 4))
    height, width = image_shape
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return torch.stack((cx, cy, w, h), dim=-1).clamp(0.0, 1.0)


def _cxcywh_norm_to_xyxy(boxes: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    height, width = image_shape
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w) * width
    y1 = (cy - 0.5 * h) * height
    x2 = (cx + 0.5 * w) * width
    y2 = (cy + 0.5 * h) * height
    output = torch.stack((x1, y1, x2, y2), dim=-1)
    mins = torch.tensor([0.0, 0.0, 0.0, 0.0], device=boxes.device)
    maxs = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes.device)
    return torch.maximum(torch.minimum(output, maxs), mins)


def _positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-np.log(10000.0) / dim)
    )
    encoding = torch.zeros((1, length, dim), dtype=torch.float32, device=device)
    encoding[0, :, 0::2] = torch.sin(positions * div)
    encoding[0, :, 1::2] = torch.cos(positions * div[: encoding[0, :, 1::2].shape[-1]])
    return encoding


def _match_queries(
    logits: torch.Tensor, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
) -> list[tuple[int, int]]:
    if target_boxes.numel() == 0:
        return []
    probs = torch.softmax(logits, dim=-1)[:, 1]
    cls_cost = -probs[:, None]
    box_cost = torch.cdist(pred_boxes, target_boxes, p=1)
    cost = (cls_cost + box_cost).detach().cpu().numpy()
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return [(int(row), int(col)) for row, col in zip(rows, cols, strict=True)]
    except Exception:
        assignments = []
        used_rows: set[int] = set()
        for col in range(cost.shape[1]):
            row = min(
                (row for row in range(cost.shape[0]) if row not in used_rows),
                key=lambda item: (float(cost[item, col]), item),
            )
            used_rows.add(row)
            assignments.append((row, col))
        return assignments


def _torch_box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    values = box_iou_xyxy(first.detach().cpu().numpy(), second.detach().cpu().numpy())
    return torch.as_tensor(values, dtype=torch.float32, device=first.device)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        ious = _torch_box_iou(boxes[current : current + 1], boxes[order[1:]]).reshape(-1)
        order = order[1:][ious <= threshold]
    return torch.stack(keep) if keep else torch.zeros((0,), dtype=torch.long, device=boxes.device)


def _empty_prediction(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
    }
