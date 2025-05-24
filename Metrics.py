# utils/metrics.py
import torch

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.cuda()
            targets = targets.cuda()
            preds = model(images)
            loss = model.loss_fn(preds, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[EVAL] Average Loss: {avg_loss:.4f}")

