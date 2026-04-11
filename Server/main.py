from fastapi import FastAPI, File, UploadFile, Response, status, HTTPException, Depends
import tempfile
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(r"C:\Users\ASUS\OneDrive\Desktop\kkkkk\Deepfake-video-Detection\ML model\ensemble_prediction_system")
from EnsemCNet import detectDeepfake, extractFrame, metaLearner

app = FastAPI()

origins = ["*"] #the website which can access

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_VIDEO_TYPES = [
    "video/mp4",
    "video/avi",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo"
]


def to_json_serializable(value):
    try:
        import numpy as np
    except ImportError:
        np = None

    if isinstance(value, dict):
        return {k: to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_serializable(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    return value


async def validate_video(file: UploadFile):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only video files are allowed"
        )
    return file


UPLOAD_FOLDER = r"C:/Users/ASUS/OneDrive/Desktop/kkkkk/Deepfake-video-Detection/Server/user_videos"


@app.post("/predict", status_code=status.HTTP_201_CREATED)
async def predict_video(file: UploadFile = Depends(validate_video)):
    # Validate filename exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a valid filename"
        )
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save file to your custom folder
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    result = analyze(file_path)

    

    return {
        "status": "success",
        "result": to_json_serializable(result)
    }


def analyze(video_path):
    output_path = r"C:/Users/ASUS/OneDrive/Desktop/kkkkk/Deepfake-video-Detection/Server/user_video"
    frame_folder = extractFrame(video_path,output_path)
    model_output = detectDeepfake(frame_folder)
    lr = metaLearner()
    pred = lr.predict(model_output)
    return(pred)



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# WATCH_FOLDER = r"C:\Users\ASUS\OneDrive\Desktop\Mini Project\Deepfake-video-Detection\Server\user_videos"

# # Your ML model function
# def run_model(video_path):
#     print(f"Processing video: {video_path}")
#     result = model.predict(video_path)
#     print(result)


# class VideoHandler(FileSystemEventHandler):
#     def on_created(self, event):
#         if not event.is_directory:
#             file_path = event.src_path

#             # Check if it's a video file
#             if file_path.endswith((".mp4", ".avi", ".mov")):
#                 print("New video detected:", file_path)

#                 # Small delay to ensure file is fully written
#                 time.sleep(2)

#                 # run_model(file_path)


# if __name__ == "__main__":
#     event_handler = VideoHandler()
#     observer = Observer()
#     observer.schedule(event_handler, WATCH_FOLDER, recursive=False)

#     print("Watching folder for new videos...")
#     observer.start()

#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()