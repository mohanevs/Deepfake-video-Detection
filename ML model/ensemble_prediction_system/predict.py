import os
import cv2
import numpy as np
from PIL import Image
from loadModel import load_models
import torch
import torch.nn.functional as F
from torchvision import transforms


# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_lstm_model(model, frame_list, device):
    processed_frames = []

    for frame in frame_list:
        frame = transform(frame)   # (3,224,224)
        processed_frames.append(frame)

    # stack → (seq_len, C, H, W)
    x = torch.stack(processed_frames)

    # add batch dim → (1, seq_len, C, H, W)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)   # (1,2)
        probs = F.softmax(output, dim=1)

    real = probs[0][0].item()
    fake = probs[0][1].item()

    return real, fake


# ---- AVG FUNCTION ----
def avg_confidence(pred_list):
    real_avg = np.mean([p[0] for p in pred_list])
    fake_avg = np.mean([p[1] for p in pred_list])
    return real_avg, fake_avg


# ---- CONVERT PIPE OUTPUT ----
def extract_conf(output):
    label = output['label'].lower()
    score = output['score']

    # ---- CASE 1: Normal labels ----
    if "real" in label:
        return score, 1 - score
    elif "fake" in label:
        return 1 - score, score

    # ---- CASE 2: LABEL_0 / LABEL_1 ----
    elif "label_1" in label:
        # you verified: LABEL_1 = REAL
        return score, 1 - score
    elif "label_0" in label:
        return 1 - score, score

    else:
        raise ValueError(f"Unknown label format: {label}")


# ---- MAIN FUNCTION ----
def video_to_predictions(dataset_path):

    pipe1, pipe2, pipe3, pipe4, model5 = load_models()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model5.to(device)
    model5.eval()  

    final_list = []

    # ---- OUTER LOOP (videos) ----
    for label_name in os.listdir(dataset_path):

        label_path = os.path.join(dataset_path, label_name)
        if not os.path.isdir(label_path):
            continue

        # assign label
        label = 0 if label_name.lower() == "real" else 1

        for video_folder in os.listdir(label_path):

            video_path = os.path.join(label_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            print(f"Processing: {video_folder}")

            frame_files = sorted(os.listdir(video_path))

            # lists for each model
            p1_list, p2_list, p3_list, p4_list = [], [], [], []
            frame_list_for_lstm = []

            # ---- INNER LOOP (frames) ----
            for frame_file in frame_files:

                frame_path = os.path.join(video_path, frame_file)
                frame = cv2.imread(frame_path)

                if frame is None:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame_list_for_lstm.append(frame)

                # predictions
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
            
            # ---- LSTM MODEL PREDICTION ----
            if len(frame_list_for_lstm) > 0:
                m5_real, m5_fake = predict_lstm_model(model5, frame_list_for_lstm, device)
            else:
                m5_real, m5_fake = 0.0, 0.0

            # ---- STORE ----
            video_result = [
                label,   # 0 = real, 1 = fake

                p1_real, p1_fake,
                p2_real, p2_fake,
                p3_real, p3_fake,
                p4_real, p4_fake,
                m5_real, m5_fake
            ]

            final_list.append(video_result)

    return final_list


# if __name__ == "__main__":
    # path = r"E:\dataset\Deepfake Detection Dataset\3. frames_dataset"
    # path = r"E:\dataset\Deepfake Detection Dataset\6. Test_frames_dataset"      
    # result = video_to_predictions(path)
    # print(result)