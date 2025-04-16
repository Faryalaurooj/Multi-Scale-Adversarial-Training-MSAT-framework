from models.msat_net import MSATModel
import torch

model = MSATModel(num_classes=20)
dummy_input = torch.randn(2, 3, 256, 256)
output = model(dummy_input)
print("Output shape:", output.shape)

