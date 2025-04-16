import torch
import torch.nn as nn
import torch.optim as optim
from models.discriminator import RealismDiscriminator

def train_discriminator(discriminator, real_data_loader, fake_data_loader, epochs=10, device='cuda'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_imgs, fake_imgs in zip(real_data_loader, fake_data_loader):
            real_imgs = real_imgs.to(device)
            fake_imgs = fake_imgs.to(device)

            real_labels = torch.ones(real_imgs.size(0), 1, 1, 1, device=device)
            fake_labels = torch.zeros(fake_imgs.size(0), 1, 1, 1, device=device)

            optimizer.zero_grad()
            output_real = discriminator(real_imgs)
            loss_real = criterion(output_real, real_labels)

            output_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(output_fake, fake_labels)

            loss = loss_real + loss_fake
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}/{epochs}] Discriminator Loss: {loss.item():.4f}")

