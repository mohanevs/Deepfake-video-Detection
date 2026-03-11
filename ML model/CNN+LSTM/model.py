import os 

# Loading dataset video paths
dataset_paths_real =[]
folder_real = r"E:\dataset\Deepfake Detection Dataset\Preprocessed_Dataset\real"
for real in os.listdir(folder_real):
    path_real = os.path.join(folder_real,real)
    dataset_paths_real.append(path_real)
# print(dataset_paths_real)

dataset_paths_fake = []
folder_fake = r"E:\dataset\Deepfake Detection Dataset\Preprocessed_Dataset\fake"
for fake in os.listdir(folder_fake):
    path_fake = os.path.join(folder_fake,fake)
    dataset_paths_fake.append(path_fake)
# print(dataset_paths_fake)

print("Real videos:", len(dataset_paths_real))
print("Fake videos:", len(dataset_paths_fake))
print("Total videos:", len(dataset_paths_real) + len(dataset_paths_fake))