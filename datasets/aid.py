import os
import cv2
import torch
from torch.utils.data import Dataset


class AIDDataset(Dataset):
    """
    AID dataset for scene classification
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, img), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, torch.tensor(label, dtype=torch.long)
