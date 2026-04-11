import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from pathlib import Path
from PIL import Image

sys.path.append(r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN_LSTM")
from model_efficientnet_b1 import DeepFakeDetector


# ===============================
# Config
# ===============================

MODEL_PATH     = r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN_LSTM\saved_models\best_model.pth"
SEQUENCE_LEN   = 20       # must match training (20 frames per video)
LABELS         = {0: "REAL", 1: "FAKE"}


# ===============================
# Setup
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(model_path):
    model = DeepFakeDetector()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}  "
          f"(val_loss: {checkpoint['val_loss']:.4f})")
    return model


# ===============================
# Frame Extraction
# ===============================

def extract_frames(video_path, n_frames=20):
    """
    Uniformly sample exactly n_frames from a video file.
    Returns a list of PIL Images.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video → {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0

    print(f"\nVideo Info:")
    print(f"  Path          : {video_path}")
    print(f"  Total frames  : {total_frames}")
    print(f"  FPS           : {fps:.2f}")
    print(f"  Duration      : {duration:.2f}s")

    # uniformly spaced indices
    if total_frames >= n_frames:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    else:
        # fewer frames than needed — repeat last frame to pad
        indices = list(range(total_frames))
        indices += [total_frames - 1] * (n_frames - total_frames)
        indices = np.array(indices)
        print(f"  ⚠ Only {total_frames} frames found, padding to {n_frames}")

    frames      = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        frame_count += 1

    cap.release()

    print(f"  Frames extracted : {len(frames)}")
    return frames


# ===============================
# Predict
# ===============================

def predict(model, video_path):
    """
    Predict whether a video is real or fake.
    Returns a result dict with label, confidence, and probabilities.
    """
    frames = extract_frames(video_path, n_frames=SEQUENCE_LEN)

    if frames is None or len(frames) == 0:
        return None

    # preprocess each frame and stack into sequence tensor
    frame_tensors = [transform(f) for f in frames]
    sequence      = torch.stack(frame_tensors)           # (20, 3, 224, 224)
    sequence      = sequence.unsqueeze(0).to(device)     # (1, 20, 3, 224, 224)

    with torch.no_grad():
        output = model(sequence)                         # (1, 2)
        probs  = torch.softmax(output, dim=1)            # (1, 2)
        confidence, predicted_class = torch.max(probs, dim=1)

    predicted_class = predicted_class.item()
    confidence      = confidence.item()
    prob_real       = probs[0][0].item()
    prob_fake       = probs[0][1].item()

    return {
        "video_path"      : video_path,
        "prediction"      : LABELS[predicted_class],
        "confidence"      : confidence * 100,
        "prob_real"       : prob_real * 100,
        "prob_fake"       : prob_fake * 100,
    }


def print_result(result):
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"  Video       : {Path(result['video_path']).name}")
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Confidence  : {result['confidence']:.2f}%")
    print(f"  Prob Real   : {result['prob_real']:.2f}%")
    print(f"  Prob Fake   : {result['prob_fake']:.2f}%")
    print("=" * 50)


# ===============================
# Main
# ===============================

if __name__ == "__main__":

    print("=" * 50)
    print("DeepFake Detector — Prediction Script")
    print("=" * 50)

    # load model once
    model = load_model(MODEL_PATH)

    while True:
        print("\n--- Video Selection ---")
        print("1. Select from videos in current directory")
        print("2. Enter custom video path")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            video_extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov"]
            all_videos = []
            for ext in video_extensions:
                all_videos += list(Path(".").glob(ext))

            if not all_videos:
                print("No video files found in current directory.")
                continue

            print(f"\nFound {len(all_videos)} video(s):")
            for idx, v in enumerate(all_videos, 1):
                print(f"  {idx}. {v.name}")

            try:
                video_idx = int(input(f"\nEnter video number (1-{len(all_videos)}): ")) - 1
                if 0 <= video_idx < len(all_videos):
                    video_path = str(all_videos[video_idx])
                else:
                    print("Invalid selection.")
                    continue
            except ValueError:
                print("Invalid input.")
                continue

        elif choice == "2":
            video_path = input("Enter full video path: ").strip().strip('"')
            if not Path(video_path).exists():
                print(f"Error: File not found → {video_path}")
                continue

        elif choice == "3":
            print("Exiting.")
            break

        else:
            print("Invalid choice.")
            continue

        # run prediction
        print("\nProcessing...")
        print("-" * 50)
        result = predict(model, video_path)

        if result:
            print_result(result)

        again = input("\nPredict another video? (y/n): ").strip().lower()
        if again != "y":
            print("Done.")
            break