'''import torch
import torch.nn as nn
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load state dict from Hugging Face
state_dict = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/resolve/main/efficientnet_b0_ffpp_c23.pth",
    map_location=device
)

# Rebuild architecture
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(state_dict)
model.to(device)
model.eval()
torch.save(model, "efficientnet-b0-ffpp-c23.pt")'''

from transformers import ViTForImageClassification, ViTImageProcessor

model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")

# save locally
model.save_pretrained("vit_deepfake_model")
processor.save_pretrained("vit_deepfake_model")

print("Model saved offline.")