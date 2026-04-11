from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

def download_model(model_name):
    print(f"\nDownloading: {model_name}")

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir="models/",
        local_files_only=False   # IMPORTANT
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        cache_dir="models/",
        local_files_only=False
    )

    pipe = pipeline(
        "image-classification",
        model=model,
        feature_extractor=processor
    )

    # Dummy run to ensure full download
    try:
        result = pipe("frame_01.jpg")
        print("Downloaded & working ✅")
    except:
        print("Downloaded (image test skipped)")



# Download all models
download_model("dima806/deepfake_vs_real_image_detection")
download_model("Wvolf/ViT_Deepfake_Detection")
download_model("ashish-001/deepfake-detection-using-ViT")
download_model("Hemg/Deepfake-Detection")