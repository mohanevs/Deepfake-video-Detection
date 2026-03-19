# why 20 frames?

## Frame Extraction Parameter Justification

### Dataset Analysis

Statistical analysis of the FaceForensics++ dataset (1,000 real and 1,000 manipulated sequences) reveals the following frame count distribution:

- **Minimum:** 287 frames
- **Median:** ~480 frames
- **Mean:** ~530 frames
- **Approximate 80th percentile range:** 295–800 frames

### Selected Parameter: 20 Frames per Video

After evaluating the dataset characteristics and the architectural requirements of the proposed CNN+LSTM model, a uniform extraction of **20 evenly-spaced frames per video** is selected as the optimal parameter.

### Justification

**1. Dataset Compatibility**
Every video in the dataset contains a minimum of 287 frames, ensuring that a uniform extraction of 20 frames is feasible across the entire dataset without special-case handling or padding. This guarantees consistent input dimensionality for all samples.

**2. Architectural Alignment**
The proposed model employs a CNN+LSTM architecture where the CNN extracts spatial features per frame and the LSTM models temporal dependencies across the sequence. Deepfake detection via this architecture relies primarily on spatial inconsistencies in facial regions — blending artifacts, texture anomalies, and boundary irregularities — which are present throughout the video rather than being temporally sparse. A sequence length of 20 provides sufficient temporal context for the LSTM to learn inter-frame patterns without requiring high frame density.

**3. Computational Feasibility**
At 20 frames per video across 2,000 videos, the total extracted frame count is 40,000. This is computationally manageable for training on consumer-grade hardware. Increasing this parameter to 40 or 60 frames would double or triple memory consumption and training time with negligible accuracy gain at the current dataset scale of 2,000 videos.

**4. Uniform Temporal Coverage**
Frames are sampled at evenly-spaced intervals across the full video duration rather than sequentially from the beginning. This approach ensures coverage of the entire temporal span of each video, accounting for cases where facial manipulation may intensify or vary throughout the sequence.

### Conclusion

A value of 20 uniformly-spaced frames per video satisfies dataset constraints, aligns with the temporal modelling requirements of the CNN+LSTM architecture, and remains computationally tractable for the scope of this project.



---

PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM> python model_1M_params.py
DeepFakeDetector(
  (cnn): CNNBlock(
    (block1): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (block2): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (block3): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
    )
    (gap): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (lstm): LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.3)
  (classifier): Sequential(
    (0): Linear(in_features=256, out_features=256, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.4, inplace=False)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.3, inplace=False)
    (6): Linear(in_features=128, out_features=64, bias=True)
    (7): ReLU()
    (8): Linear(in_features=64, out_features=2, bias=True)
  )
)

--- Parameter Count ---
Total parameters     : 1,530,562
Trainable parameters : 1,530,562

Input shape  : torch.Size([2, 20, 3, 224, 224])
Output shape : torch.Size([2, 2])
Architecture check passed.
PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM> python train_model.py
Total videos found: 2000
Training videos: 1800
Testing videos : 200
Using device: cuda
Epoch 1/1: 100%|█████████████████████████████████████████████████████████████████████████████████████| 450/450 [31:44<00:00,  4.23s/it, acc=50.06%, loss=0.694]

Epoch 1/1
Training Loss: 312.2466
Training Accuracy: 50.06%


---



PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM> python model_3LK_params.py
DeepFakeDetector(
  (cnn): CNNBlock(
    (block1): Sequential(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (block2): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (block3): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
    )
    (gap): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (lstm): LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.3)
  (classifier): Sequential(
    (0): Linear(in_features=128, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.4, inplace=False)
    (3): Linear(in_features=128, out_features=64, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.3, inplace=False)
    (6): Linear(in_features=64, out_features=32, bias=True)
    (7): ReLU()
    (8): Linear(in_features=32, out_features=2, bias=True)
  )
)

--- Parameter Count ---

--- Parameter Count ---
Total parameters     : 384,354
Trainable parameters : 384,354

Input shape  : torch.Size([2, 20, 3, 224, 224])
Output shape : torch.Size([2, 2])
Architecture check passed.
Total parameters     : 384,354
Trainable parameters : 384,354

Input shape  : torch.Size([2, 20, 3, 224, 224])
Output shape : torch.Size([2, 2])
Architecture check passed.
PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM> python train_model.py
Output shape : torch.Size([2, 2])
Architecture check passed.
PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM> python train_model.py
Total videos found: 2000
PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM> python train_model.py
Total videos found: 2000
Training videos: 1800
Total videos found: 2000
Training videos: 1800
Testing videos : 200
Using device: cuda
Training videos: 1800
Testing videos : 200
Using device: cuda
Epoch 1/15: 100%|███████████████████████████████████████████████████████████████████| 450/450 [02:36<00:00,  2.88it/s, acc=49.78%, loss=0.703] 
Testing videos : 200
Using device: cuda
Epoch 1/15: 100%|███████████████████████████████████████████████████████████████████| 450/450 [02:36<00:00,  2.88it/s, acc=49.78%, loss=0.703] 

Epoch 1/15
Epoch 1/15: 100%|███████████████████████████████████████████████████████████████████| 450/450 [02:36<00:00,  2.88it/s, acc=49.78%, loss=0.703] 

Epoch 1/15
Training Loss: 313.7062

Epoch 1/15
Training Loss: 313.7062
Training Loss: 313.7062
Training Accuracy: 49.78%


---

PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM> python model.py      
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\timm\models\_factory.py:138: UserWarning: Mapping deprecated model name xception to current legacy_xception.
  model = create_fn(
DeepFakeDetector(
  (cnn): CNNBlock(
    (xception): Xception(
      (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
      (block1): Block(
        (skip): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (skipbn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (rep): Sequential(
          (0): SeparableConv2d(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
            (pointwise): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): SeparableConv2d(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (pointwise): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
      )
      (block2): Block(
        (skip): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (skipbn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
            (pointwise): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
      )
      (block3): Block(
        (skip): Conv2d(256, 728, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (skipbn): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
            (pointwise): Conv2d(256, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
      )
      (block4): Block(
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace=True)
          (7): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (block5): Block(
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace=True)
          (7): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (block6): Block(
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace=True)
          (7): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (block7): Block(
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace=True)
          (7): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (block8): Block(
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace=True)
          (7): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (block9): Block(
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace=True)
          (7): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (block10): Block(
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace=True)
          (7): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (block11): Block(
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU(inplace=True)
          (7): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (block12): Block(
        (skip): Conv2d(728, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (skipbn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (rep): Sequential(
          (0): ReLU()
          (1): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(inplace=True)
          (4): SeparableConv2d(
            (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
            (pointwise): Conv2d(728, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
      )
      (conv3): SeparableConv2d(
        (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
        (pointwise): Conv2d(1024, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (bn3): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act3): ReLU(inplace=True)
      (conv4): SeparableConv2d(
        (conv1): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)
        (pointwise): Conv2d(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (bn4): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act4): ReLU(inplace=True)
      (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
      (fc): Linear(in_features=2048, out_features=1000, bias=True)
    )
    (gap): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Linear(in_features=2048, out_features=256, bias=True)
  )
  (lstm): LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.3)
  (classifier): Sequential(
    (0): Linear(in_features=256, out_features=256, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.4, inplace=False)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.3, inplace=False)
    (6): Linear(in_features=128, out_features=64, bias=True)
    (7): ReLU()
    (8): Linear(in_features=64, out_features=2, bias=True)
  )
)

--- Parameter Count ---
Total parameters     : 24,540,242
Trainable parameters : 1,684,290

Input shape  : torch.Size([2, 20, 3, 224, 224])
Output shape : torch.Size([2, 2])
Architecture check passed.
PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM> python train_model.py
Total videos found: 2000
Training videos: 1800
Testing videos : 200
Using device: cuda
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\timm\models\_factory.py:138: UserWarning: Mapping deprecated model name xception to current legacy_xception.
  model = create_fn(
Epoch 1/15: 100%|███████████████████████████████████████████████████████████████████| 450/450 [04:12<00:00,  1.78it/s, acc=53.17%, loss=0.637]

Epoch 1/15
Training Loss: 311.3695
Training Accuracy: 53.17%