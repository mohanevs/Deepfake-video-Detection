import sys
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
sys.path.append(r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN_LSTM")
from model_efficientnet_b1 import DeepFakeDetector
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device : {device}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")

path = r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\Ensemble_prediction_system\models\custom_trained.pth"

def load_models():
    
    device_id = 0 if torch.cuda.is_available() else -1

    # Model 1
    model_name1 = "dima806/deepfake_vs_real_image_detection"
    processor1 = AutoImageProcessor.from_pretrained(model_name1, cache_dir="models/", local_files_only=True)
    model1 = AutoModelForImageClassification.from_pretrained(model_name1, cache_dir="models/", local_files_only=True)
    pipe1 = pipeline("image-classification", model=model1, feature_extractor=processor1, device=device_id)

    # Model 2
    model_name2 = "Wvolf/ViT_Deepfake_Detection"
    processor2 = AutoImageProcessor.from_pretrained(model_name2, cache_dir="models/", local_files_only=True)
    model2 = AutoModelForImageClassification.from_pretrained(model_name2, cache_dir="models/", local_files_only=True)
    pipe2 = pipeline("image-classification", model=model2, feature_extractor=processor2, device=device_id)

    # Model 3
    model_name3 = "ashish-001/deepfake-detection-using-ViT"
    processor3 = AutoImageProcessor.from_pretrained(model_name3, cache_dir="models/", local_files_only=True)
    model3 = AutoModelForImageClassification.from_pretrained(model_name3, cache_dir="models/", local_files_only=True)
    pipe3 = pipeline("image-classification", model=model3, feature_extractor=processor3, device=device_id)

    # Model 4
    model_name4 = "Hemg/Deepfake-Detection"
    processor4 = AutoImageProcessor.from_pretrained(model_name4, cache_dir="models/", local_files_only=True)
    model4 = AutoModelForImageClassification.from_pretrained(model_name4, cache_dir="models/", local_files_only=True)
    pipe4 = pipeline("image-classification", model=model4, feature_extractor=processor4, device=device_id)

    # Model 5 (Custom)
    model5 = DeepFakeDetector()
    checkpoint = torch.load(path, map_location=device)
    model5.load_state_dict(checkpoint["model_state_dict"])
    model5.to(device)
    model5.eval()

    return pipe1, pipe2, pipe3, pipe4, model5
    

if __name__ == "__main__":
    load_models()