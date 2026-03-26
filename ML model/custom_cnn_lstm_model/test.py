import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN_LSTM")
from model_efficientnet_b1 import DeepFakeDetector
from train_model import build_samples, DeepFakeDataset

TRAIN_TEST_ROOT = r"E:\dataset\Deepfake Detection Dataset\3. frames_dataset"

all_samples  = build_samples(TRAIN_TEST_ROOT)
real_samples = [s for s in all_samples if s[1] == 0]
fake_samples = [s for s in all_samples if s[1] == 1]
random.seed(42)
random.shuffle(real_samples)
random.shuffle(fake_samples)
train_samples = real_samples[:900] + fake_samples[:900]
random.shuffle(train_samples)

loader = DataLoader(DeepFakeDataset(train_samples), batch_size=4,
                    shuffle=False,    # ← no shuffle
                    num_workers=0)

device = torch.device("cuda")
model  = DeepFakeDetector().to(device)
opt    = optim.Adam(model.parameters(), lr=1e-4)
crit   = nn.CrossEntropyLoss()

# grab fixed 10 batches once
fixed_batches = []
for i, (videos, labels) in enumerate(loader):
    if i >= 10:
        break
    fixed_batches.append((videos, labels))

# check label distribution in these fixed batches
all_labels = torch.cat([l for _, l in fixed_batches])
print(f"Fixed batches — real: {(all_labels==0).sum().item()}, fake: {(all_labels==1).sum().item()}")

print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Acc':>8}")
print("-" * 28)

for epoch in range(10):
    total_loss, correct, total = 0, 0, 0

    for videos, labels in fixed_batches:
        videos, labels = videos.to(device), labels.to(device)
        opt.zero_grad()
        out  = model(videos)
        loss = crit(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        total_loss += loss.item()
        _, pred = torch.max(out, 1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)

    print(f"{epoch+1:>6}  {total_loss/10:>8.4f}  {100*correct/total:>7.1f}%")