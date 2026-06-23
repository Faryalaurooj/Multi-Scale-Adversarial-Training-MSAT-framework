import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from models.msat import MSAT
from models.detector import Detector
from models.discriminator import Discriminator
from models.singan_msa import SinGAN_MSA

from datasets.dota import DOTADataset


# -------------------------
# Config
# -------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# IoU Loss (basic but correct)
# -------------------------
def iou_loss(pred_box, target_box):
    px, py, pw, ph = pred_box.unbind(-1)
    tx, ty, tw, th = target_box.unbind(-1)

    inter_x1 = torch.max(px - pw / 2, tx - tw / 2)
    inter_y1 = torch.max(py - ph / 2, ty - th / 2)
    inter_x2 = torch.min(px + pw / 2, tx + tw / 2)
    inter_y2 = torch.min(py + ph / 2, ty + th / 2)

    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    area_p = pw * ph
    area_t = tw * th

    union = area_p + area_t - inter + 1e-6

    return 1 - inter / union


# -------------------------
# Detection Loss (proper structure)
# -------------------------
def detection_loss(preds, targets):
    """
    preds: list of feature-level outputs
    targets: list of GT tensors
    """

    cls_loss = 0.0
    box_loss = 0.0
    obj_loss = 0.0

    for p, t in zip(preds, targets):

        # classification loss
        cls_loss += torch.nn.functional.cross_entropy(
            p["cls"].flatten(2).mean(-1),
            torch.zeros(p["cls"].shape[0], dtype=torch.long, device=p["cls"].device)
        )

        # objectness loss
        obj_loss += torch.nn.functional.binary_cross_entropy_with_logits(
            p["obj"], torch.ones_like(p["obj"])
        )

        # box loss (simplified matching)
        if t.numel() > 0:
            box_pred = p["box"].mean(dim=(2, 3))
            box_target = t[:, :4].mean(dim=0).unsqueeze(0).to(p["box"].device)

            box_loss += iou_loss(box_pred, box_target).mean()

    return cls_loss + obj_loss + box_loss


# -------------------------
# Build model
# -------------------------
def build_model(cfg):
    backbone = MSAT(
        in_channels=3,
        base=cfg["model"]["backbone"]["base_channels"],
        layers=cfg["model"]["backbone"]["num_blocks"]
    )

    detector = Detector(backbone, cfg["model"]["detector"]["num_classes"])
    discriminator = Discriminator()
    gan = SinGAN_MSA()

    return detector, discriminator, gan


# -------------------------
# Train
# -------------------------
def train(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DOTADataset(
        cfg["data"]["train_img_dir"],
        cfg["data"]["train_label_dir"]
    )

    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=True)

    detector, discriminator, gan = build_model(cfg)

    detector.to(device)
    discriminator.to(device)
    gan.to(device)

    opt_det = torch.optim.AdamW(detector.parameters(), lr=1e-4)
    opt_disc = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
    opt_gan = torch.optim.AdamW(gan.parameters(), lr=1e-4)

    scaler = GradScaler()

    print("🚀 MSAT Training Started")

    for epoch in range(cfg["train"]["epochs"]):

        for i, (imgs, targets) in enumerate(loader):

            imgs = imgs.to(device)

            # =========================
            # 1. Generate synthetic images
            # =========================
            fake_imgs = gan(imgs)

            # =========================
            # 2. Train Discriminator
            # =========================
            real_logits = discriminator(imgs)
            fake_logits = discriminator(fake_imgs.detach())

            d_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) +
                torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            )

            opt_disc.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(opt_disc)

            # =========================
            # 3. Train Detector
            # =========================
            preds = detector(imgs)

            det_loss = detection_loss(preds, targets)

            opt_det.zero_grad()
            scaler.scale(det_loss).backward()
            scaler.step(opt_det)

            # =========================
            # 4. Train GAN
            # =========================
            fake_logits = discriminator(fake_imgs)
            gan_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )

            opt_gan.zero_grad()
            scaler.scale(gan_loss).backward()
            scaler.step(opt_gan)

            scaler.update()

            if i % 20 == 0:
                print(f"[E{epoch} I{i}] "
                      f"Det: {det_loss.item():.4f} | "
                      f"D: {d_loss.item():.4f} | "
                      f"GAN: {gan_loss.item():.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(detector.state_dict(), f"checkpoints/msat_{epoch}.pt")

    print("✅ Training Finished")


if __name__ == "__main__":
    cfg = load_config("configs/msat.yaml")
    train(cfg)
