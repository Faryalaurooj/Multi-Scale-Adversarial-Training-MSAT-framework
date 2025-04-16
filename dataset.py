from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random

class HybridDataset(Dataset):
    def __init__(self, real_dir, synthetic_dir, transform=None):
        self.real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
        self.synthetic_images = [os.path.join(synthetic_dir, f) for f in os.listdir(synthetic_dir)]
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.real_images) + len(self.synthetic_images)

    def __getitem__(self, idx):
        if idx < len(self.real_images):
            img_path = self.real_images[idx]
            label = 1
        else:
            img_path = self.synthetic_images[idx - len(self.real_images)]
            label = 0

        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label

    def real_subset(self):
        return ImageLabelSubset(self.real_images, label=1, transform=self.transform)

    def synthetic_subset(self):
        return ImageLabelSubset(self.synthetic_images, label=0, transform=self.transform)

class ImageLabelSubset(Dataset):
    def __init__(self, image_paths, label, transform):
        self.image_paths = image_paths
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.label

