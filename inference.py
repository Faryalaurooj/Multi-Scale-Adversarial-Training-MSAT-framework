import os
import yaml
import cv2
import torch
import numpy as np

from models.msat import MSAT
from models.detector import Detector


# -------------------------
# Config loader
# -------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Load model
# -------------------------
def load_model(cfg, ckpt_path, device):

    backbone = MSAT(
        in_channels=3,
        base=cfg["model"]["backbone"]["base_channels"],
        layers=cfg["model"]["backbone"]["num_blocks"]
    )

    model = Detector(
        backbone=backbone,
        num_classes=cfg["model"]["detector"]["num_classes"]
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()

    return model


# -------------------------
# Preprocess image
# -------------------------
def preprocess(img, img_size=1024):

    img = cv2.resize(img, (img_size, img_size))
    img = img[:, :, ::-1]  # BGR -> RGB

    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0)


# -------------------------
# Simple NMS
# -------------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes, iou_thresh=0.5):

    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda x: x[-1], reverse=True)

    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)

        boxes = [
            b for b in boxes
            if iou(best[:4], b[:4]) < iou_thresh
        ]

    return keep


# -------------------------
# Decode predictions
# -------------------------
def decode(preds, conf_thresh=0.25, iou_thresh=0.5):

    results = []

    for p in preds:

        obj = torch.sigmoid(p["obj"])
        cls = torch.softmax(p["cls"], dim=1)
        box = p["box"]

        batch_boxes = []

        B = obj.shape[0]

        for i in range(B):

            score = obj[i].mean().item()

            if score < conf_thresh:
                continue

            b = box[i].mean(dim=(1, 2)).detach().cpu().numpy()

            x, y, w, h = b.tolist()

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            cls_id = cls[i].argmax().item()

            batch_boxes.append([x1, y1, x2, y2, cls_id, score])

        batch_boxes = nms(batch_boxes, iou_thresh)

        results.append(batch_boxes)

    return results


# -------------------------
# Draw boxes
# -------------------------
def visualize(img, boxes):

    img = img.copy()

    for b in boxes:
        x1, y1, x2, y2, cls_id, score = b

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"{cls_id}:{score:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    return img


# -------------------------
# Single image inference
# -------------------------
def run_image(model, img_path, device, cfg):

    img = cv2.imread(img_path)
    orig = img.copy()

    inp = preprocess(img, cfg["data"]["img_size"]).to(device)

    with torch.no_grad():
        preds = model(inp)

    boxes = decode(preds)[0]

    vis = visualize(orig, boxes)

    return vis


# -------------------------
# Folder inference
# -------------------------
def run_folder(model, folder, device, cfg, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    for f in os.listdir(folder):

        if not f.endswith((".jpg", ".png", ".tif")):
            continue

        path = os.path.join(folder, f)

        result = run_image(model, path, device, cfg)

        out_path = os.path.join(save_dir, f)

        cv2.imwrite(out_path, result)

        print(f"[INFO] Saved: {out_path}")


# -------------------------
# Main
# -------------------------
def main():

    cfg = load_config("configs/msat.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        cfg,
        cfg["inference"]["weight_path"],
        device
    )

    print("🚀 MSAT Inference Started")

    test_folder = cfg["data"]["val_img_dir"]
    save_dir = cfg["inference"]["output_dir"]

    run_folder(model, test_folder, device, cfg, save_dir)

    print("✅ Inference Completed")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()
