import sys
sys.path.append(r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\ensemble_prediction_system")
from EnsemCNet import detectDeepfake , extractFrame , metaLearner

if __name__ == "__main__" :
    
    video_path = r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\Ensemble_prediction_system\Realistic_Deepfake_Video_Generation.mp4"
    output_path = r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\Ensemble_prediction_system\user_video"
    frame_folder = extractFrame(video_path,output_path)
    model_output = detectDeepfake(frame_folder)
    lr = metaLearner()
    pred = lr.predict(model_output)
    print(pred)