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

def preprocess(video_path):
    # Here process the video to the scale it to the shape that efficient required and all rest and returnthe preprocessed video 

def predict(model,video_path):
    
    # Preprocess the video frames to models input shape
    video = preprocess(video_path)
    #continue this
    
    return predictions

def model1_predict(video_path): 
    # efficientnet-b0-ffpp-c23
    # Load the model
    model = torch.load("efficientnet-b0-ffpp-c23.pt", map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    print("Model loaded successfully!")
    
    result = predict(model,video_path)
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