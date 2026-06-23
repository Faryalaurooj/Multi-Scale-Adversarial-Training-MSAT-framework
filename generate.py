import os
import yaml
import torch
import torchvision.utils as vutils

from models.singan_msa import SinGAN_MSA
from models.discriminator import Discriminator


# -------------------------
# Config loader
# -------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Set seed
# -------------------------
def set_seed(seed=42):
    import random
    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Load models
# -------------------------
def load_models(cfg, device):

    gan = SinGAN_MSA(
        channels=cfg["singan"]["channels"],
        num_layers=cfg["singan"]["num_layers"]
    ).to(device)

    disc = Discriminator().to(device)

    return gan, disc


# -------------------------
# Generate synthetic images
# -------------------------
def generate_images(gan, batch_size, channels, img_size, device):

    # random noise / latent seed
    z = torch.randn(batch_size, channels, img_size, img_size).to(device)

    with torch.no_grad():
        fake = gan(z)

    return fake


# -------------------------
# Optional filtering using discriminator
# -------------------------
def filter_by_discriminator(fake_imgs, discriminator, threshold=0.5):

    with torch.no_grad():
        logits = discriminator(fake_imgs)
        probs = torch.sigmoid(logits)

    mask = probs.mean(dim=(1, 2, 3)) > threshold

    return fake_imgs[mask]


# -------------------------
# Save images
# -------------------------
def save_images(images, out_dir, prefix="gen"):

    os.makedirs(out_dir, exist_ok=True)

    for i, img in enumerate(images):
        path = os.path.join(out_dir, f"{prefix}_{i}.png")
        vutils.save_image(img, path, normalize=True)


# -------------------------
# Main generation pipeline
# -------------------------
def main():

    cfg = load_config("configs/msat.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg["project"]["seed"])

    gan, disc = load_models(cfg, device)

    gan.eval()
    disc.eval()

    print("🚀 MSAT Image Generation Started")

    # -------------------------
    # generate batch
    # -------------------------
    fake_imgs = generate_images(
        gan,
        batch_size=cfg["data"]["batch_size"],
        channels=cfg["singan"]["channels"],
        img_size=cfg["data"]["img_size"],
        device=device
    )

    # -------------------------
    # optional filtering
    # -------------------------
    if cfg["discriminator"]["enabled"]:
        fake_imgs = filter_by_discriminator(fake_imgs, disc)

    # -------------------------
    # save outputs
    # -------------------------
    save_images(
        fake_imgs,
        cfg["logging"]["save_dir"] + "/generated",
        prefix="msat_gen"
    )

    print(f"✅ Generated {len(fake_imgs)} images")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()
