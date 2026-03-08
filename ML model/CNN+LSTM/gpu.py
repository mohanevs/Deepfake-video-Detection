import torch

def check_gpu():
    if torch.cuda.is_available():
        print("✓ PyTorch is connected to GPU")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    else:
        print("✗ PyTorch is NOT connected to GPU")
        print("Running on CPU")

if __name__ == "__main__":
    check_gpu()