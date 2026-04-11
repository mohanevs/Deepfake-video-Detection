'''
    what i am going to do is :

    - load all five models into there isolated function 
    - and each funnction take one input parameter video and return the prediction 0 or 1 and how much confidence 
    - a model_predict function that takes input parameter as video and call rest functions of models and takes prediction from each model and store them in a List.
    - the format of list is  : [[1,0.98],[0,0.23]...likewise 5 list in list 
    - then i take this prediction and write a main function that takes this list of prediction and train asimple FCNN and outputs real/fake with softmax 
    
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load models

# efficientnet-b0-ffpp-c23
model1 = torch.load("efficientnet-b0-ffpp-c23.pt", map_location=device, weights_only=False)
model1.eval()
model1.to(device)
print("Model loaded successfully!")


def preprocess(video_path):
    frames = []

    for file in sorted(os.listdir(video_path)):
        img_path = os.path.join(video_path, file)

        img = Image.open(img_path).convert("RGB")
        img = transform(img)

        frames.append(img)

    return frames

def predict(model, video_path):

    frames = preprocess(video_path)

    confidences = []

    for frame in frames:
        frame = frame.unsqueeze(0).to(device)  # (1, C, H, W)

        with torch.no_grad():
            output = model(frame)
            probs = F.softmax(output, dim=1)

        conf = probs[0][1].item()  # probability of FAKE class
        confidences.append(conf)

    # Aggregate (average)
    avg_conf = sum(confidences) / len(confidences)

    pred = 1 if avg_conf > 0.5 else 0
    return pred, avg_conf

def model1_predict(video_path): 
    result = predict(model1,video_path)
    return result

def model2_predict(video_path):
    return 0, 0.34

def model3_predict(video_path):
    return 1, 0.81

def model4_predict(video_path):
    return 1, 0.77

def model5_predict(video_path):
    return 0, 0.29


def get_all_predictions(video_path):
    preds = []

    preds.append(model1_predict(video_path))
    preds.append(model2_predict(video_path))
    preds.append(model3_predict(video_path))
    preds.append(model4_predict(video_path))
    preds.append(model5_predict(video_path))

    return preds

def prepare_meta_input(preds):
    input_flatten = []
    for _, conf in preds:
        input_flatten.append(1 - conf)  
        input_flatten.append(conf)      
    return input_flatten

class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)
    
    
def load_dataset(dataset_path):
    data = []

    real_path = os.path.join(dataset_path, "real")
    fake_path = os.path.join(dataset_path, "fake")

    for file in os.listdir(real_path):
        data.append((os.path.join(real_path, file), 0))

    for file in os.listdir(fake_path):
        data.append((os.path.join(fake_path, file), 1))
        
    print("Size of data :", len(data))
    return data


def train_meta_model(dataset_path, epochs=10):

    data = load_dataset(dataset_path)

    model = MetaLearner()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0

        for video_path, label in data:

            # Step 1: Get predictions from all models
            preds = get_all_predictions(video_path)

            # Step 2: Convert to input vector
            meta_input = prepare_meta_input(preds)

            meta_input = torch.tensor(meta_input, dtype=torch.float32)
            meta_input = meta_input.unsqueeze(0)

            label_tensor = torch.tensor([label], dtype=torch.long)

            # Step 3: Forward
            output = model(meta_input)

            # Step 4: Loss
            loss = criterion(output, label_tensor)

            # Step 5: Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model


def save_model(model, path="meta_model.pth"):
    torch.save(model.state_dict(), path)
    print("Meta model saved!")
    
    
if __name__ == "__main__":

    dataset_path = r"E:\dataset\Deepfake Detection Dataset\3. frames_dataset"

    model = train_meta_model(dataset_path, epochs=10)

    save_model(model)