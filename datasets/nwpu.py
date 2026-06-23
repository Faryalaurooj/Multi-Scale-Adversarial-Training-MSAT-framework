import os
import cv2
import torch
from torch.utils.data import Dataset


class NWPUVHR10Dataset(Dataset):
    """
    NWPU VHR-10 dataset loader
    """

    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform

        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(self.ann_dir, img_name.replace(".jpg", ".txt"))

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x1, y1, x2, y2 = map(float, line.split())
                    boxes.append([x1, y1, x2, y2, cls])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))

        if self.transform:
            img = self.transform(image=img)["image"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, boxes
