import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights


# =========================
# DATASET PATH
# =========================

DATASET_PATH = r"E:\dataset\Deepfake Detection Dataset\Preprocessed_Dataset"


# =========================
# CNN + LSTM MODEL
# =========================

class CNNLSTMDeepfakeDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.classifier = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,2)
        )

    def forward(self,x):

        batch, seq, c, h, w = x.shape

        x = x.view(batch*seq, c, h, w)

        features = self.cnn(x)

        features = features.view(batch, seq, 512)

        lstm_out,_ = self.lstm(features)

        final = lstm_out[:,-1,:]

        output = self.classifier(final)

        return output


# =========================
# DATASET
# =========================

class DeepfakeVideoDataset(Dataset):

    def __init__(self, dataset_path, seq_len=8):

        self.seq_len = seq_len

        self.videos = []
        self.labels = []

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

        real_path = os.path.join(dataset_path,"real")
        fake_path = os.path.join(dataset_path,"fake")

        # Load real videos
        for v in os.listdir(real_path):
            if v.endswith((".mp4",".avi",".mov",".mkv")):
                self.videos.append(os.path.join(real_path,v))
                self.labels.append(0)

        # Load fake videos
        for v in os.listdir(fake_path):
            if v.endswith((".mp4",".avi",".mov",".mkv")):
                self.videos.append(os.path.join(fake_path,v))
                self.labels.append(1)

        print("Videos loaded:",len(self.videos))


    def __len__(self):
        return len(self.videos)


    def extract_frames(self, video_path):

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        indices = np.linspace(
            0,
            max(total_frames-1,0),
            self.seq_len,
            dtype=int
        )

        frames = []
        frame_id = 0
        index_set = set(indices)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id in index_set:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = Image.fromarray(frame)

                frame = self.transform(frame)

                frames.append(frame)

            frame_id += 1

        cap.release()

        if len(frames) == 0:
            frames = [torch.zeros(3,224,224)]

        while len(frames) < self.seq_len:
            frames.append(frames[-1])

        frames = torch.stack(frames)

        return frames


    def __getitem__(self, idx):

        video_path = self.videos[idx]

        label = self.labels[idx]

        frames = self.extract_frames(video_path)

        return frames, torch.tensor(label)


# =========================
# TRAINING
# =========================

def train():

    batch_size = 2
    epochs = 1
    lr = 1e-4

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:",device)

    dataset = DeepfakeVideoDataset(DATASET_PATH)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    model = CNNLSTMDeepfakeDetector().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=lr)

    for epoch in range(epochs):

        print("\n==============================")
        print(f"Starting Epoch {epoch+1}/{epochs}")
        print("==============================\n")

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for batch_idx,(frames,labels) in enumerate(dataloader):

            print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)}")

            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(frames)

            loss = criterion(outputs,labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            _,predicted = torch.max(outputs,1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            print("Batch Loss:",round(loss.item(),4))
            print("--------------------------")

        acc = 100 * correct / total

        print("\nEpoch Completed")
        print("Average Loss:",round(total_loss,4))
        print("Accuracy:",round(acc,2),"%\n")

    torch.save(model.state_dict(),"deepfake_detector.pth")

    print("Model saved as deepfake_detector.pth")


# =========================
# RUN
# =========================

if __name__ == "__main__":

    train()