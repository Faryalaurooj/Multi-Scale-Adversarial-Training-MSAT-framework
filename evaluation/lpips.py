import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips


class LPIPSMetric(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity)
    Full research-grade wrapper for MSAT framework.

    Supports:
    - batch inference
    - safe device handling
    - image normalization
    - evaluation + training usage
    """

    def __init__(self, net="alex", device=None, normalize=True):
        super().__init__()

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        # pretrained LPIPS network
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    # -----------------------------
    # Internal normalization
    # -----------------------------
    def _normalize(self, x):
        """
        LPIPS expects inputs in [-1, 1]
        If input is [0, 1], convert it.
        """
        if self.normalize:
            return x * 2.0 - 1.0
        return x

    # -----------------------------
    # Forward (batch mode)
    # -----------------------------
    def forward(self, img1, img2):
        """
        Args:
            img1: (B, 3, H, W)
            img2: (B, 3, H, W)

        Returns:
            scalar LPIPS distance
        """

        img1 = self._normalize(img1).to(self.device)
        img2 = self._normalize(img2).to(self.device)

        with torch.no_grad():
            dist = self.model(img1, img2)

        return dist.mean()

    # -----------------------------
    # Pairwise single-image API
    # -----------------------------
    def pair(self, img1, img2):
        """
        Args:
            img1, img2: (3, H, W)
        Returns:
            float LPIPS distance
        """

        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        return self.forward(img1, img2).item()

    # -----------------------------
    # Feature-wise evaluation
    # -----------------------------
    def batch_score(self, real_batch, fake_batch):
        """
        Useful for GAN evaluation loops.

        Args:
            real_batch: (B, 3, H, W)
            fake_batch: (B, 3, H, W)
        """
        return self.forward(real_batch, fake_batch).item()
