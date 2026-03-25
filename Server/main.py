from fastapi import FastAPI, File, UploadFile, Response, status, HTTPException, Depends
import tempfile
from pydantic import BaseModel

app = FastAPI()


ALLOWED_VIDEO_TYPES = [
    "video/mp4",
    "video/avi",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo"
]


async def validate_video(file: UploadFile):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only video files are allowed"
        )
    return file

@app.post("/predict", status_code= status.HTTP_201_CREATED)
async def predict_video(file: UploadFile = File(...)):

    
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        
        temp_video.write(await file.read())
        temp_video.flush()

        # Pass temp file path to model
        # result = run_model(temp_video.name)


    return {
        "status": "success"
        # "result": result
    }