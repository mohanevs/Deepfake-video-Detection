import joblib
from predict import video_to_predictions, extract_conf, avg_confidence, predict_lstm_model
from loadModel import load_models
import torch
import cv2
import os
from PIL import Image


lr = joblib.load("meta_model_lr.pkl")

def detect_deepfake(frames_folder):

    pipe1, pipe2, pipe3, pipe4, model5 = load_models()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model5.to(device)
    model5.eval()

    p1_list, p2_list, p3_list, p4_list = [], [], [], []
    frame_list_for_lstm = []
    final_list = []

    frame_files = sorted(os.listdir(frames_folder))

    for frame_file in frame_files:

        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        frame_list_for_lstm.append(frame)

        # ---- PIPELINES ----
        out1 = pipe1(frame)[0]
        out2 = pipe2(frame)[0]
        out3 = pipe3(frame)[0]
        out4 = pipe4(frame)[0]

        p1_list.append(extract_conf(out1))
        p2_list.append(extract_conf(out2))
        p3_list.append(extract_conf(out3))
        p4_list.append(extract_conf(out4))

    # ---- AVERAGE ----
    p1_real, p1_fake = avg_confidence(p1_list)
    p2_real, p2_fake = avg_confidence(p2_list)
    p3_real, p3_fake = avg_confidence(p3_list)
    p4_real, p4_fake = avg_confidence(p4_list)

    # ---- LSTM ----
    if len(frame_list_for_lstm) > 0:
        m5_real, m5_fake = predict_lstm_model(model5, frame_list_for_lstm, device)
    else:
        m5_real, m5_fake = 0.0, 0.0

    # ---- FINAL DECISION ----
    # ---- STORE ----
    video_result = [
        p1_real, p1_fake,
        p2_real, p2_fake,
        p3_real, p3_fake,
        p4_real, p4_fake,
        m5_real, m5_fake
    ]

    final_list.append(video_result)
    
    return final_list
    
def extract_frame(video_path, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print("Error: FPS not detected")
        return

    for i in range(20):
        frame_id = int(i * fps)   # 1 frame per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()
        if not ret:
            break

        frame_name = os.path.join(output_folder, f"frame_{i:02d}.jpg")
        cv2.imwrite(frame_name, frame)

    cap.release()
    print("Saved 20 frames (time-based)")
    
    return output_folder
    

if __name__ == "__main__" :
    
    video_path = r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\Ensemble_prediction_system\Realistic_Deepfake_Video_Generation.mp4"
    output_path = r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\Ensemble_prediction_system\user_video"
    frame_folder = extract_frame(video_path,output_path)
    model_output = detect_deepfake(frame_folder)
    pred = lr.predict(model_output)
    print(pred)
    

