import os
import yaml
import torch
from torch.utils.data import DataLoader

from models.msat import MSAT
from models.detector import Detector

from datasets.dota import DOTADataset
from evaluation.metrics import compute_map


# -------------------------
# Config loader
# -------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Load checkpoint
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
# Collate
# -------------------------
def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), labels


# -------------------------
# Convert model output → boxes (simplified)
# -------------------------
def decode_predictions(preds, conf_threshold=0.25):
    """
    Convert raw detector output into boxes
    (lightweight placeholder for MSAT framework)
    """

    all_boxes = []

    for p in preds:

        obj = p["obj"].sigmoid()
        box = p["box"]
        cls = p["cls"].softmax(dim=1)

        batch_boxes = []

        for i in range(obj.shape[0]):

            score = obj[i].mean().item()

            if score < conf_threshold:
                continue

            # simplified box: center format → xyxy
            bx = box[i].mean(dim=(1, 2)).detach().cpu()

            x, y, w, h = bx.tolist()
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2

            cls_id = cls[i].argmax().item()

            batch_boxes.append([x1, y1, x2, y2, cls_id, score])

        all_boxes.append(batch_boxes)

    return all_boxes


# -------------------------
# Evaluation loop
# -------------------------
def evaluate(model, loader, device):

    preds_all = []
    targets_all = []

    with torch.no_grad():

        for imgs, targets in loader:

            imgs = imgs.to(device)

            preds = model(imgs)

            decoded_preds = decode_predictions(preds)

            preds_all.extend(decoded_preds)
            targets_all.extend(targets)

    results = compute_map(preds_all, targets_all)

    return results


# -------------------------
# Inference on single image batch
# -------------------------
def inference(model, loader, device, save=False):

    results = []

    with torch.no_grad():

        for i, (imgs, _) in enumerate(loader):

            imgs = imgs.to(device)
            preds = model(imgs)

            decoded = decode_predictions(preds)

            results.append(decoded)

            if save:
                print(f"[INFO] Processed batch {i}")

    return results


# -------------------------
# Main
# -------------------------
def main():

    cfg = load_config("configs/msat.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = DOTADataset(
        cfg["data"]["val_img_dir"],
        cfg["data"]["val_label_dir"]
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # model
    model = load_model(
        cfg,
        cfg["inference"]["weight_path"],
        device
    )

    print("🚀 Running MSAT Evaluation...")

    results = evaluate(model, loader, device)

    print("\n📊 Evaluation Results:")
    print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")


if __name__ == "__main__":
    main()
