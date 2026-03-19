import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import DeepFakeDetector


# ===============================
# Dataset Class
# ===============================

class DeepFakeDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        video_path, label = self.samples[idx]

        frame_files = sorted(os.listdir(video_path))

        frames = []

        for frame_file in frame_files:

            frame_path = os.path.join(video_path, frame_file)

            image = Image.open(frame_path).convert("RGB")

            image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)

        return frames, label


# ===============================
# Main Function
# ===============================

def main():
    
    os.makedirs("saved_models", exist_ok=True)

    # ===============================
    # Dataset Path
    # ===============================

    dataset_root = r"E:\dataset\Deepfake Detection Dataset\3. frames_dataset"

    samples = []

    for label_name in ["real", "fake"]:

        label_path = os.path.join(dataset_root, label_name)

        label = 0 if label_name == "real" else 1

        for video_folder in os.listdir(label_path):

            video_path = os.path.join(label_path, video_folder)

            if os.path.isdir(video_path):

                samples.append((video_path, label))

    print(f"Total videos found: {len(samples)}")


    # ===============================
    # Train/Test Split
    # ===============================

    random.shuffle(samples)

    train_samples = samples[:1800]
    test_samples  = samples[1800:2000]

    print(f"Training videos: {len(train_samples)}")
    print(f"Testing videos : {len(test_samples)}")


    # ===============================
    # Dataset Objects
    # ===============================

    train_dataset = DeepFakeDataset(train_samples)
    test_dataset  = DeepFakeDataset(test_samples)


    # ===============================
    # DataLoaders
    # ===============================

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # ===============================
    # Device
    # ===============================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)


    # ===============================
    # Model
    # ===============================

    model = DeepFakeDetector().to(device)


    # ===============================
    # Loss + Optimizer
    # ===============================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )


    # ===============================
    # Training Loop
    # ===============================

    EPOCHS = 15

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for videos, labels in progress_bar:

            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(videos)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

            accuracy = 100 * correct / total

            progress_bar.set_postfix(
                loss=loss.item(),
                acc=f"{accuracy:.2f}%"
            )

        train_acc = 100 * correct / total

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Training Loss: {running_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")

    # ===============================
    # Save Model Checkpoint
    # ===============================

    model_path = f"saved_models/deepfake_model_epoch_{EPOCHS}.pth"

    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
    }, model_path)

    print(f"Model saved to {model_path}")

    # ===============================
    # Evaluation
    # ===============================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for videos, labels in test_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    test_accuracy = 100 * correct / total

    print("\n============================")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("============================")


# ===============================
# Entry Point (IMPORTANT for Windows)
# ===============================

if __name__ == "__main__":
    main()