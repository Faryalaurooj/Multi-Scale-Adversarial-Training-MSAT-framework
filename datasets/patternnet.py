import os
import cv2
import torch
from torch.utils.data import Dataset


class PatternNetDataset(Dataset):
    """
    PatternNet / UC Merced-style dataset loader
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))
        self.class_map = {c: i for i, c in enumerate(self.classes)}

        self.data = []

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img in os.listdir(cls_path):
                self.data.append((os.path.join(cls_path, img), self.class_map[cls]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, torch.tensor(label, dtype=torch.long)
