import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

sys.path.append(r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN_LSTM")
# sys.path.append(r"C:\Users\ASUS\OneDrive\Desktop\kkkkk\Deepfake-video-Detection\ML model\custom_cnn_lstm_model")
from model_efficientnet_b1 import DeepFakeDetector


# ===============================
# Config
# ===============================

TRAIN_TEST_ROOT = r"E:\dataset\Deepfake Detection Dataset\3. frames_dataset"
VAL_ROOT        = r"E:\dataset\Deepfake Detection Dataset\5. Validation_Dataset_Frames"
SAVE_DIR        = "saved_models"
EPOCHS          = 20
BATCH_SIZE      = 4
LR              = 1e-4
NUM_WORKERS     = 4
PATIENCE        = 5


# ===============================
# Dataset
# ===============================

class DeepFakeDataset(Dataset):

    def __init__(self, samples, expected_frames=20):
        self.samples         = samples
        self.expected_frames = expected_frames
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frame_files = sorted([
            f for f in os.listdir(video_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # pad or trim to exactly expected_frames
        if len(frame_files) < self.expected_frames:
            frame_files += [frame_files[-1]] * (self.expected_frames - len(frame_files))
        elif len(frame_files) > self.expected_frames:
            frame_files = frame_files[:self.expected_frames]

        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            image = Image.open(frame_path).convert("RGB")
            image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames)  # (20, 3, 224, 224)
        return frames, label


# ===============================
# Early Stopping
# ===============================

class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.001, save_path="saved_models/best_model.pth"):
        self.patience  = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss, model, optimizer, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save({
                "epoch"               : epoch,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss"            : val_loss,
            }, self.save_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            self.counter += 1
            print(f"  No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True


# ===============================
# Data Helpers
# ===============================

def build_samples(root_path):
    samples = []
    for label_name in ["real", "fake"]:
        label      = 0 if label_name == "real" else 1
        label_path = os.path.join(root_path, label_name)
        for video_folder in os.listdir(label_path):
            video_path = os.path.join(label_path, video_folder)
            if os.path.isdir(video_path):
                samples.append((video_path, label))
    return samples


def verify_paths():
    all_ok = True
    for name, path in [("Train/Test Root", TRAIN_TEST_ROOT), ("Validation Root", VAL_ROOT)]:
        exists = os.path.exists(path)
        print(f"{name}: {'✓ Found' if exists else '✗ NOT FOUND'} → {path}")
        if not exists:
            all_ok = False
        elif exists:
            for split in ["real", "fake"]:
                split_path = os.path.join(path, split)
                count = len(os.listdir(split_path)) if os.path.exists(split_path) else 0
                print(f"   {split}: {count} video folders")
    return all_ok


# ===============================
# Evaluation
# ===============================

def evaluate(model, loader, loader_name, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for videos, labels in tqdm(loader, desc=f"Evaluating {loader_name}"):
            videos  = videos.to(device)
            labels  = labels.to(device)
            outputs = model(videos)
            probs   = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())  # prob of fake class

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ===============================
# Plots
# ===============================

def plot_confusion_matrix(y_true, y_pred, title, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{title.replace(' ', '_').replace('—', '-')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_training_curves(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss",     markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",       markersize=4)
    axes[0].set_title("Loss vs Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"],  "b-o", label="Train Accuracy", markersize=4)
    axes[1].plot(epochs, history["val_acc"],    "r-o", label="Val Accuracy",   markersize=4)
    axes[1].set_title("Accuracy vs Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_roc(y_true, y_prob, title, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{title.replace(' ', '_').replace('—', '-')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    return roc_auc


# ===============================
# Main
# ===============================

def main():

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Verify paths ──────────────────────────────────────────
    print("\n=== Path Verification ===")
    if not verify_paths():
        print("Aborting — fix paths above.")
        return

    # ── Build splits ──────────────────────────────────────────
    print("\n=== Building Splits ===")
    all_samples  = build_samples(TRAIN_TEST_ROOT)
    real_samples = [s for s in all_samples if s[1] == 0]
    fake_samples = [s for s in all_samples if s[1] == 1]

    random.seed(42)
    random.shuffle(real_samples)
    random.shuffle(fake_samples)

    train_samples = real_samples[:900] + fake_samples[:900]
    test_samples  = real_samples[900:] + fake_samples[900:]

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    val_samples = build_samples(VAL_ROOT)

    print(f"Train : {len(train_samples)} videos")
    print(f"Test  : {len(test_samples)}  videos")
    print(f"Val   : {len(val_samples)}   videos")

    for name, samples in [("Train", train_samples), ("Test", test_samples), ("Val", val_samples)]:
        real = sum(1 for _, l in samples if l == 0)
        fake = sum(1 for _, l in samples if l == 1)
        print(f"  {name} → real: {real}, fake: {fake}")

    # ── DataLoaders ───────────────────────────────────────────
    train_loader = DataLoader(
        DeepFakeDataset(train_samples), batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        DeepFakeDataset(test_samples),  batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        DeepFakeDataset(val_samples),   batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Test batches  : {len(test_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # ── Device & Model ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device : {device}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    model     = DeepFakeDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,} / {total:,}")

    # ── Training Loop ─────────────────────────────────────────
    print("\n=== Training ===")
    history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    early_stopping = EarlyStopping(patience=PATIENCE, save_path=os.path.join(SAVE_DIR, "best_model.pth"))

    for epoch in range(EPOCHS):

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

        for videos, labels in train_bar:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            _, predicted   = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total   += labels.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*train_correct/train_total:.2f}%")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = 100 * train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ", leave=False)

        with torch.no_grad():
            for videos, labels in val_bar:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)
                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*val_correct/val_total:.2f}%")

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = 100 * val_correct / val_total

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train → Loss: {avg_train_loss:.4f}  Acc: {avg_train_acc:.2f}%")
        print(f"  Val   → Loss: {avg_val_loss:.4f}  Acc: {avg_val_acc:.2f}%")

        # Epoch checkpoint
        torch.save({
            "epoch"               : epoch + 1,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss"          : avg_train_loss,
            "val_loss"            : avg_val_loss,
        }, os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))

        # Early stopping
        early_stopping.step(avg_val_loss, model, optimizer, epoch + 1)
        if early_stopping.triggered:
            print(f"\n⚡ Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\n✓ Training complete — best val loss: {early_stopping.best_loss:.4f}")

    # ── Load Best Model ───────────────────────────────────────
    print("\n=== Loading Best Model ===")
    checkpoint = torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded from epoch {checkpoint['epoch']}  (val_loss: {checkpoint['val_loss']:.4f})")

    # ── Evaluation ────────────────────────────────────────────
    print("\n=== Evaluation ===")
    y_true_test, y_pred_test, y_prob_test = evaluate(model, test_loader, "Test Set", device)
    y_true_val,  y_pred_val,  y_prob_val  = evaluate(model, val_loader,  "Val Set",  device)

    print("\n" + "=" * 50)
    print("TEST SET — Classification Report")
    print("=" * 50)
    print(classification_report(y_true_test, y_pred_test, target_names=["Real", "Fake"]))

    print("=" * 50)
    print("VALIDATION SET — Classification Report")
    print("=" * 50)
    print(classification_report(y_true_val, y_pred_val, target_names=["Real", "Fake"]))

    # ── Plots ─────────────────────────────────────────────────
    print("\n=== Saving Plots ===")
    plot_training_curves(history, SAVE_DIR)
    plot_confusion_matrix(y_true_test, y_pred_test, "Confusion Matrix - Test Set", SAVE_DIR)
    plot_confusion_matrix(y_true_val,  y_pred_val,  "Confusion Matrix - Val Set",  SAVE_DIR)

    auc_test = plot_roc(y_true_test, y_prob_test, "ROC Curve - Test Set", SAVE_DIR)
    auc_val  = plot_roc(y_true_val,  y_prob_val,  "ROC Curve - Val Set",  SAVE_DIR)

    print(f"\nTest AUC : {auc_test:.4f}")
    print(f"Val AUC  : {auc_val:.4f}")
    print("\n✓ All plots saved to", SAVE_DIR)


# ===============================
# Entry Point
# ===============================

if __name__ == "__main__":
    main()