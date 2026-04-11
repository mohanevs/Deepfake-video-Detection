import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# =============================
# Device configuration
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================
# Load ViT model and processor (offline)
# =============================
model = ViTForImageClassification.from_pretrained("vit_deepfake_model").to(device)
processor = ViTImageProcessor.from_pretrained("vit_deepfake_model")

model.eval()

print("ViT model loaded successfully!")


# =============================
# Video prediction function
# =============================
def predict_video(video_path, sample_frames=10):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")

    # choose evenly spaced frames
    if total_frames > sample_frames:
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    else:
        frame_indices = np.arange(total_frames)

    predictions = []
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count in frame_indices:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            inputs = processor(images=frame_pil, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            prediction = probs.cpu().numpy()[0]

            predictions.append(prediction)

            print(f"Frame {frame_count}: {prediction}")

        frame_count += 1

    cap.release()

    predictions = np.array(predictions)

    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)

    label_idx = np.argmax(mean_pred)
    label = model.config.id2label[label_idx]
    confidence = mean_pred[label_idx] * 100

    results = {
        "video_path": video_path,
        "total_frames": total_frames,
        "frames_analyzed": len(predictions),
        "mean_prediction": mean_pred,
        "std_prediction": std_pred,
        "label": label,
        "confidence": confidence
    }

    return results


# =============================
# Main menu
# =============================
if __name__ == "__main__":

    print("=" * 60)
    print("ViT Deepfake Video Detector")
    print("=" * 60)

    while True:

        print("\n--- Video Selection ---")
        print("1. Select from current directory")
        print("2. Enter custom path")
        print("3. Exit")

        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":

            videos = list(Path(".").glob("*.mp4")) + \
                     list(Path(".").glob("*.avi")) + \
                     list(Path(".").glob("*.mkv"))

            if not videos:
                print("No videos found.")
                continue

            for i, v in enumerate(videos, 1):
                print(f"{i}. {v.name}")

            try:
                idx = int(input("Select video number: ")) - 1
                video_path = str(videos[idx])
            except:
                print("Invalid selection.")
                continue

        elif choice == "2":

            video_path = input("Enter full path: ").strip()

            if not Path(video_path).exists():
                print("File not found.")
                continue

        elif choice == "3":
            break

        else:
            continue

        try:
            sample_frames = int(input("Frames to sample (default 10): ") or "10")
        except:
            sample_frames = 10

        print("\nProcessing...")
        print("-" * 60)

        results = predict_video(video_path, sample_frames)

        if results:

            print("\n================ RESULT ================")
            print(f"Video: {results['video_path']}")
            print(f"Frames analyzed: {results['frames_analyzed']}")
            print(f"Mean prediction: {results['mean_prediction']}")
            print(f"Std deviation: {results['std_prediction']}")
            print(f"\nFINAL RESULT: {results['label']}")
            print(f"Confidence: {results['confidence']:.2f}%")
            print("========================================")

        again = input("\nCheck another video? (y/n): ").lower()

        if again != "y":
            break

print("Done.")