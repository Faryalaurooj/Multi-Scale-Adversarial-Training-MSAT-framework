# 6. File: main.py (Bringing It All Together)

from dataset import HybridDataset
from train_gan import train_discriminator
from models.discriminator import RealismDiscriminator
from models.msa_module import MultiScaleAttention
from utils.singan_wrapper import SinGANAugmentor

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

def main():
    # Paths to real and synthetic image folders
    real_dir = "./data/real"
    synthetic_dir = "./data/synthetic"

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load hybrid dataset
    dataset = HybridDataset(real_dir, synthetic_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Instantiate Discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = RealismDiscriminator().to(device)

    # Split into real and synthetic loaders
    real_loader = DataLoader(dataset.real_subset(), batch_size=4, shuffle=True)
    fake_loader = DataLoader(dataset.synthetic_subset(), batch_size=4, shuffle=True)

    # Train Discriminator
    train_discriminator(discriminator, real_loader, fake_loader, epochs=10, device=device)

    # SinGAN Augmentation Example
    singan_aug = SinGANAugmentor(config_path="config.yaml")
    example_image = os.path.join(real_dir, os.listdir(real_dir)[0])
    Gs, Zs, reals, NoiseAmp = singan_aug.train_singan(example_image)
    samples = singan_aug.generate_samples(Gs, Zs, reals, NoiseAmp, num_samples=5)

    print(f"Generated {len(samples)} synthetic images.")

if __name__ == "__main__":
    main()

