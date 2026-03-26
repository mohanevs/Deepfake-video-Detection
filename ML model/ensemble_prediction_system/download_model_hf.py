from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

# Model 1  dima806
model_name = "dima806/deepfake_vs_real_image_detection"

processor = AutoImageProcessor.from_pretrained(model_name, cache_dir="models/", local_files_only=True)

model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir="models/", local_files_only=True )

model1 = pipeline("image-classification", model=model, image_processor=processor)

result = model1("frame_01.jpg")
print(result)




# Model 2  Wvolf
model_name = "Wvolf/ViT_Deepfake_Detection"

processor = AutoImageProcessor.from_pretrained(model_name, cache_dir="models/", local_files_only=True)

model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir="models/", local_files_only=True)

model2 = pipeline("image-classification", model=model, image_processor=processor)

result = model2("frame_01.jpg")
print(result)




# Model 3  ashish-001
model_name = "ashish-001/deepfake-detection-using-ViT"

processor = AutoImageProcessor.from_pretrained(model_name, cache_dir="models/", local_files_only=True)

model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir="models/", local_files_only=True)

model2 = pipeline("image-classification", model=model, image_processor=processor)

result = model2("frame_01.jpg")
print(result)




# Model 4  Hemg
model_name = "Hemg/Deepfake-Detection"

processor = AutoImageProcessor.from_pretrained(model_name, cache_dir="models/", local_files_only=True)

model = AutoModelForImageClassification.from_pretrained(model_name, cache_dir="models/", local_files_only=True)

model2 = pipeline("image-classification", model=model, image_processor=processor)

result = model2("frame_01.jpg")
print(result)