from predict import video_to_predictions
import pandas as pd 

def save_to_csv(result, save_path):

    columns = [
        "label",

        "m1_real", "m1_fake",
        "m2_real", "m2_fake",
        "m3_real", "m3_fake",
        "m4_real", "m4_fake",
        "m5_real", "m5_fake"
    ]

    df = pd.DataFrame(result, columns=columns)

    df.to_csv(save_path, index=False)

    print(f"CSV saved at: {save_path}")

    
if __name__ == "__main__":

    dataset_path = r"E:\dataset\Deepfake Detection Dataset\3. frames_dataset"
    save_path = r"C:\Users\ASUS\Desktop\Mini Project\Ensemble_prediction_system\deepfake_features.csv"

    result = video_to_predictions(dataset_path)

    save_to_csv(result, save_path)