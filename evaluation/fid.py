import torch
import numpy as np
from torchvision.models import inception_v3
from scipy import linalg
import torch.nn.functional as F


class InceptionFeatureExtractor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)

    def get_features(self, x):
        with torch.no_grad():
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            feats = self.model(x)
        return feats.cpu().numpy()


def calculate_fid(real_feats, fake_feats):
    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid
