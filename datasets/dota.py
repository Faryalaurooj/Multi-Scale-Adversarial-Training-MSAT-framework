import os
import cv2
import torch
from torch.utils.data import Dataset


class DOTADataset(Dataset):
    """
    DOTA dataset loader for object detection
    Assumes:
    images/
    labels/ (YOLO or DOTA txt format converted)
    """

    def __init__(self, img_dir, label_dir, img_size=1024, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform

        self.images = [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".tif"))]

    def __len__(self):
        return len(self.images)

    def load_label(self, label_path):
        boxes = []

        if not os.path.exists(label_path):
            return torch.zeros((0, 5))

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])

                boxes.append([x, y, w, h, cls])

        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = self.load_label(label_path)

        if self.transform:
            img = self.transform(image=img)["image"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, labels
