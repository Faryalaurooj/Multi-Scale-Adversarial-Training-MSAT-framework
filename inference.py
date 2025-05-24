import os
import argparse
import torch
import cv2
import numpy as np

from models.yolov10.yolov10 import YOLOv10
from utils.data_loader import preprocess_image  # We'll write this helper

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with YOLOv10 MSAT model")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv10 model weights')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to save output image with detections')
    return parser.parse_args()

def postprocess(preds, conf_thresh=0.5):
    # Placeholder: Filter detections by confidence threshold
    # preds assumed shape: [N, 6] with columns [x1, y1, x2, y2, conf, class]
    mask = preds[:, 4] >= conf_thresh
    return preds[mask]

def draw_boxes(image, detections):
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"Class {int(cls)}: {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return image

def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = YOLOv10().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Load and preprocess image
    img = cv2.imread(args.image)
    input_tensor = preprocess_image(img).to(device)

    # Forward pass
    with torch.no_grad():
        preds = model(input_tensor)
        # Assuming preds is a tensor of shape [N, 6] (x1,y1,x2,y2,conf,class)
        preds = preds.cpu().numpy()

    # Postprocess and filter by confidence
    detections = postprocess(preds, args.conf_thresh)

    # Draw boxes
    output_img = draw_boxes(img, detections)

    # Save output image
    cv2.imwrite(args.output, output_img)
    print(f"Output saved to {args.output}")

if __name__ == '__main__':
    main()

