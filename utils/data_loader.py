import cv2
import torch
import numpy as np

def preprocess_image(image, input_size=640):
    # Resize and normalize image to feed to YOLOv10
    h, w = image.shape[:2]
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    image_resized = cv2.resize(image, (nw, nh))
    image_padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    image_padded[:nh, :nw] = image_resized

    # Convert BGR to RGB
    image_rgb = image_padded[:, :, ::-1].astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, 640, 640]

    return image_tensor

