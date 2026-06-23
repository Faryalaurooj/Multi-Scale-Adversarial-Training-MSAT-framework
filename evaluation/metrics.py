import torch
import numpy as np


# -------------------------
# IoU computation
# -------------------------
def iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


# -------------------------
# Precision / Recall
# -------------------------
def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall


# -------------------------
# mAP (simplified IoU-based)
# -------------------------
def compute_map(preds, targets, iou_threshold=0.5):
    """
    preds: list of predictions per image
    targets: list of GT boxes per image
    """

    TP, FP, FN = 0, 0, 0

    for pred_boxes, gt_boxes in zip(preds, targets):

        matched = set()

        for p in pred_boxes:
            matched_iou = 0
            match_idx = -1

            for i, g in enumerate(gt_boxes):
                score = iou(p, g)

                if score > matched_iou:
                    matched_iou = score
                    match_idx = i

            if matched_iou >= iou_threshold and match_idx not in matched:
                TP += 1
                matched.add(match_idx)
            else:
                FP += 1

        FN += len(gt_boxes) - len(matched)

    precision, recall = precision_recall(TP, FP, FN)

    map_score = (precision + recall) / 2  # simplified proxy mAP

    return {
        "precision": precision,
        "recall": recall,
        "mAP@0.5": map_score
    }
