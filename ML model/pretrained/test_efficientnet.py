import torch
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model = torch.load("efficientnet-b0-ffpp-c23.pt", map_location=device, weights_only=False)
model.eval()
model.to(device)
print("Model loaded successfully!")

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_video(video_path, sample_frames=10):
    """
    Process video and make predictions on sampled frames
    
    Args:
        video_path: Path to the video file
        sample_frames: Number of frames to sample from the video
    
    Returns:
        Dictionary with predictions and statistics
    """
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nVideo: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Calculate frame indices to sample
    if total_frames > sample_frames:
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    else:
        frame_indices = np.arange(total_frames)
    
    predictions = []
    frame_count = 0
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Sample specific frames
        if frame_count in frame_indices:
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = transforms.ToPILImage()(frame_rgb)
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(frame_tensor)
                # Get probability (if output has softmax/sigmoid)
                if len(output.shape) > 1:
                    probs = torch.nn.functional.softmax(output, dim=1)
                    prediction = probs.cpu().numpy()
                else:
                    prediction = output.cpu().numpy()
                
                predictions.append(prediction)
                print(f"Frame {frame_count}: {prediction}")
        
        frame_count += 1
    
    cap.release()
    
    # Aggregate results
    predictions = np.array(predictions)
    
    results = {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "frames_analyzed": len(predictions),
        "predictions": predictions,
        "mean_prediction": np.mean(predictions, axis=0),
        "std_prediction": np.std(predictions, axis=0)
    }
    
    print(f"Mean prediction: {results['mean_prediction']}")
    print(f"Std prediction: {results['std_prediction']}")
    
    return results

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("EfficientNet-B0 Video Prediction Model")
    print("=" * 60)
    
    while True:
        print("\n--- Video Selection ---")
        print("1. Select from videos in current directory")
        print("2. Enter custom video file path")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            # Find all video files
            all_videos = list(Path(".").glob("*.mp4")) + list(Path(".").glob("*.avi")) + list(Path(".").glob("*.mkv"))
            
            if not all_videos:
                print("No video files found in current directory.")
                print("Please add .mp4, .avi, or .mkv files.")
                continue
            
            # Display available videos
            print(f"\nFound {len(all_videos)} video file(s):")
            for idx, video in enumerate(all_videos, 1):
                print(f"{idx}. {video.name}")
            
            # Get user selection
            try:
                video_idx = int(input(f"\nEnter video number (1-{len(all_videos)}): ")) - 1
                if 0 <= video_idx < len(all_videos):
                    video_path = str(all_videos[video_idx])
                else:
                    print("Invalid selection!")
                    continue
            except ValueError:
                print("Invalid input!")
                continue
        
        elif choice == "2":
            video_path = input("Enter video file path: ").strip()
            if not Path(video_path).exists():
                print(f"Error: File not found - {video_path}")
                continue
        
        elif choice == "3":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice!")
            continue
        
        # Get number of frames to sample
        try:
            sample_frames = int(input("Enter number of frames to sample (default 10): ") or "10")
        except ValueError:
            sample_frames = 10
        
        # Process video
        print("\nProcessing video...")
        print("-" * 60)
        results = predict_video(video_path, sample_frames=sample_frames)
        
        if results:
            print("-" * 60)
            print("\n=== RESULTS ===")
            print(f"Video: {results['video_path']}")
            print(f"Total frames: {results['total_frames']}")
            print(f"Frames analyzed: {results['frames_analyzed']}")
            print(f"Mean prediction: {results['mean_prediction']}")
            print(f"Std deviation: {results['std_prediction']}")
            print("=" * 60)
        
        # Ask if user wants to process another video
        another = input("\nProcess another video? (y/n): ").strip().lower()
        if another != 'y':
            print("Thank you for using the model!")
            break