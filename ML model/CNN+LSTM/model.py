import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import os
from PIL import Image
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Define CNN+LSTM Model
class CNNLSTMDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTMDeepfakeDetector, self).__init__()
        
        # CNN backbone (ResNet18)
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove final layer
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        # Classification layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, 3, 224, 224)
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract CNN features for each frame
        cnn_out = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            features = self.cnn(frame)
            cnn_out.append(features)
        
        # Stack features: (batch_size, seq_len, 512)
        cnn_out = torch.stack(cnn_out, dim=1)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # Use last hidden state
        lstm_features = lstm_out[:, -1, :]
        
        # Classification
        output = self.fc(lstm_features)
        return output


# Custom Dataset
class DeepfakeDataset(Dataset):
    def __init__(self, dataset_path, seq_len=10, img_size=224):
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self.labels = []
        
        # Load real videos
        real_path = os.path.join(dataset_path, 'real')
        if os.path.exists(real_path):
            for video_folder in os.listdir(real_path):
                video_path = os.path.join(real_path, video_folder)
                if os.path.isdir(video_path):
                    self.samples.append(video_path)
                    self.labels.append(0)
        
        # Load fake videos
        fake_path = os.path.join(dataset_path, 'fake')
        if os.path.exists(fake_path):
            for video_folder in os.listdir(fake_path):
                video_path = os.path.join(fake_path, video_folder)
                if os.path.isdir(video_path):
                    self.samples.append(video_path)
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]
        
        # Get frame files
        frames = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
        
        # Select seq_len frames uniformly
        if len(frames) >= self.seq_len:
            indices = np.linspace(0, len(frames) - 1, self.seq_len, dtype=int)
        else:
            indices = list(range(len(frames))) + [len(frames) - 1] * (self.seq_len - len(frames))
        
        # Load and transform frames
        frame_tensors = []
        for i in indices:
            frame_path = os.path.join(video_path, frames[i])
            img = Image.open(frame_path).convert('RGB')
            img_tensor = self.transform(img)
            frame_tensors.append(img_tensor)
        
        # Stack frames: (seq_len, 3, 224, 224)
        frames_tensor = torch.stack(frame_tensors)
        
        return frames_tensor, torch.tensor(label, dtype=torch.long)


# Training
def train():
    # Hyperparameters
    dataset_path = './datasets'  # Update with your dataset path
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 1
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = CNNLSTMDeepfakeDetector(num_classes=2)
    model = model.to(device)
    
    # Dataset and DataLoader
    dataset = DeepfakeDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (frames, labels) in enumerate(dataloader):
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'deepfake_detector.pth')
    print('Model saved as deepfake_detector.pth')


if __name__ == '__main__':
    train()