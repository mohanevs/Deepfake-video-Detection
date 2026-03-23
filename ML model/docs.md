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
Epoch 1/10: 100%|████████████████████████████████████████████████████████████████████████████████████| 450/450 [04:11<00:00,  1.79it/s, acc=48.72%, loss=0.691] 

Epoch 1/10
Training Loss: 312.4615
Training Accuracy: 48.72%
Model saved to saved_models/deepfake_model_epoch_10.pth
Epoch 2/10: 100%|████████████████████████████████████████████████████████████████████████████████████| 450/450 [04:10<00:00,  1.80it/s, acc=48.89%, loss=0.693]

Epoch 2/10
Training Loss: 312.3502
Training Accuracy: 48.89%
Model saved to saved_models/deepfake_model_epoch_10.pth
Epoch 3/10: 100%|████████████████████████████████████████████████████████████████████████████████████| 450/450 [04:11<00:00,  1.79it/s, acc=49.17%, loss=0.692]

Epoch 3/10
Training Loss: 312.2474
Training Accuracy: 49.17%
Model saved to saved_models/deepfake_model_epoch_10.pth
Epoch 4/10: 100%|████████████████████████████████████████████████████████████████████████████████████| 450/450 [04:10<00:00,  1.80it/s, acc=48.67%, loss=0.691]

Epoch 4/10
Training Loss: 312.0824
Training Accuracy: 48.67%
Model saved to saved_models/deepfake_model_epoch_10.pth
Epoch 5/10:   3%|██▍                                                                                  | 13/450 [00:22<12:44,  1.75s/it, acc=36.54%, loss=0.701]
Traceback (most recent call last):
  File "C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM\train_model.py", line 265, in <module>
    main()
    ~~~~^^
  File "C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM\train_model.py", line 193, in main
    running_loss += loss.item()
                    ~~~~~~~~~^^
KeyboardInterrupt
PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN+LSTM> 




--- 

# XceptioNet+LSTM(freeze all) terminal :

---

PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN_LSTM> python model.py
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
PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN_LSTM> python train_model.py

=== Path Verification ===
Train/Test Root: ✓ Found → E:\dataset\Deepfake Detection Dataset\3. frames_dataset
   real: 1000 video folders
   fake: 1000 video folders
Validation Root: ✓ Found → E:\dataset\Deepfake Detection Dataset\5. Validation_Dataset_Frames
   real: 500 video folders
   fake: 500 video folders

=== Building Splits ===
Train : 1800 videos
Test  : 200  videos
Val   : 1000   videos
  Train → real: 900, fake: 900
  Test → real: 100, fake: 100
  Val → real: 500, fake: 500

Train batches : 450
Test batches  : 50
Val batches   : 250

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\timm\models\_factory.py:138: UserWarning: Mapping deprecated model name xception to current legacy_xception.
Test batches  : 50
Val batches   : 250

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\timm\models\_factory.py:138: UserWarning: Mapping deprecated model name xception to current legacy_Test batches  : 50
Val batches   : 250

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
Test batches  : 50
Val batches   : 250

Using device : cuda
Test batches  : 50
Val batches   : 250

Using device : cuda
Test batches  : 50
Val batches   : 250

Using device : cuda
Test batches  : 50
Val batches   : 250
Test batches  : 50
Val batches   : 250
Test batches  : 50
Val batches   : 250

Test batches  : 50
Val batches   : 250

Test batches  : 50
Test batches  : 50
Val batches   : 250
Test batches  : 50
Val batches   : 250
Test batches  : 50
Val batches   : 250
Test batches  : 50
Val batches   : 250

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\timm\models\_factory.py:138: UserWarning: Mapping deprecated model name xception to current legacy_xception.
  model = create_fn(
Trainable params : 1,684,290 / 24,540,242

=== Training ===
                                                                                                                                                                
Epoch 1/20
  Train → Loss: 0.6934  Acc: 50.78%
  Val   → Loss: 0.6985  Acc: 50.00%
  ✓ Best model saved (val_loss: 0.6985)

Epoch 2/20
  Train → Loss: 0.6943  Acc: 50.33%
  Val   → Loss: 0.6933  Acc: 49.70%
  ✓ Best model saved (val_loss: 0.6933)

Epoch 3/20
  Train → Loss: 0.6937  Acc: 47.61%
  Val   → Loss: 0.6930  Acc: 50.00%
  No improvement. Patience: 1/5

Epoch 4/20
  Train → Loss: 0.6936  Acc: 49.78%
  Val   → Loss: 0.6931  Acc: 50.00%
  No improvement. Patience: 2/5

Epoch 5/20
  Train → Loss: 0.6935  Acc: 50.39%
  Val   → Loss: 0.6931  Acc: 50.00%
  No improvement. Patience: 3/5

Epoch 6/20
  Train → Loss: 0.6938  Acc: 50.17%
  Val   → Loss: 0.6932  Acc: 50.00%
  No improvement. Patience: 4/5

Epoch 7/20
  Train → Loss: 0.6936  Acc: 49.11%
  Val   → Loss: 0.6931  Acc: 50.00%
  No improvement. Patience: 5/5

⚡ Early stopping triggered at epoch 7

✓ Training complete — best val loss: 0.6933

=== Loading Best Model ===
✓ Loaded from epoch 2  (val_loss: 0.6933)

=== Evaluation ===
Evaluating Test Set: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:42<00:00,  1.19it/s] 
Evaluating Val Set: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [01:49<00:00,  2.29it/s] 

==================================================
TEST SET — Classification Report
==================================================
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
              precision    recall  f1-score   support

        Real       0.50      1.00      0.67       100
        Fake       0.00      0.00      0.00       100

    accuracy                           0.50       200
   macro avg       0.25      0.50      0.33       200
weighted avg       0.25      0.50      0.33       200

==================================================
VALIDATION SET — Classification Report
==================================================
              precision    recall  f1-score   support

        Real       0.50      0.99      0.66       500
        Fake       0.00      0.00      0.00       500

    accuracy                           0.50      1000
   macro avg       0.25      0.50      0.33      1000
weighted avg       0.25      0.50      0.33      1000


=== Saving Plots ===
  Saved → saved_models\training_curves.png
  Saved → saved_models\Confusion_Matrix_-_Test_Set.png
  Saved → saved_models\Confusion_Matrix_-_Val_Set.png
  Saved → saved_models\ROC_Curve_-_Test_Set.png
  Saved → saved_models\ROC_Curve_-_Val_Set.png

Test AUC : 0.4625
Val AUC  : 0.4310

✓ All plots saved to saved_models


---
# XceptionNet+LSTM (unfreeze some layer) terminals:
PS C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN_LSTM> python train_model.py

=== Path Verification ===
Train/Test Root: ✓ Found → E:\dataset\Deepfake Detection Dataset\3. frames_dataset
   real: 1000 video folders
   fake: 1000 video folders
Validation Root: ✓ Found → E:\dataset\Deepfake Detection Dataset\5. Validation_Dataset_Frames
   real: 500 video folders
   fake: 500 video folders

=== Building Splits ===
Train : 1800 videos
Test  : 200  videos
Val   : 1000   videos
  Train → real: 900, fake: 900
  Test → real: 100, fake: 100
  Val → real: 500, fake: 500

Train batches : 450
Test batches  : 50
Val batches   : 250

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\timm\models\_factory.py:138: UserWarning: Mapping deprecated model name xception to current legacy_xception.
  model = create_fn(
Xception — frozen: 94, unfrozen: 62 / 156 params groups
Trainable params : 15,902,322 / 24,540,242

=== Training ===

Epoch 1/20
  Train → Loss: 0.6939  Acc: 49.89%
  Val   → Loss: 0.6935  Acc: 50.00%
  ✓ Best model saved (val_loss: 0.6935)
                                                                                                                                                         
Epoch 2/20
  Train → Loss: 0.6938  Acc: 49.22%
  Val   → Loss: 0.6927  Acc: 50.00%
  No improvement. Patience: 1/5

Epoch 3/20
  Train → Loss: 0.6934  Acc: 49.83%
  Val   → Loss: 0.6935  Acc: 50.00%
  No improvement. Patience: 2/5

Epoch 4/20
  Train → Loss: 0.6933  Acc: 51.28%
  Val   → Loss: 0.6938  Acc: 50.00%
  No improvement. Patience: 3/5

Epoch 5/20
  Train → Loss: 0.6938  Acc: 50.22%
  Val   → Loss: 0.6938  Acc: 50.00%
  No improvement. Patience: 4/5

Epoch 6/20
  Train → Loss: 0.6938  Acc: 48.72%
  Val   → Loss: 0.6929  Acc: 50.70%
  No improvement. Patience: 5/5

⚡ Early stopping triggered at epoch 6

✓ Training complete — best val loss: 0.6935

=== Loading Best Model ===
✓ Loaded from epoch 1  (val_loss: 0.6935)

=== Evaluation ===
Evaluating Test Set: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:51<00:00,  1.04s/it] 
Evaluating Val Set: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [01:49<00:00,  2.29it/s] 

==================================================
TEST SET — Classification Report
==================================================
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
              precision    recall  f1-score   support

        Real       0.50      1.00      0.67       100
        Fake       0.00      0.00      0.00       100

    accuracy                           0.50       200
   macro avg       0.25      0.50      0.33       200
weighted avg       0.25      0.50      0.33       200

==================================================
VALIDATION SET — Classification Report
==================================================
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
              precision    recall  f1-score   support

        Real       0.50      1.00      0.67       500
        Fake       0.00      0.00      0.00       500

    accuracy                           0.50      1000
   macro avg       0.25      0.50      0.33      1000
weighted avg       0.25      0.50      0.33      1000


=== Saving Plots ===
  Saved → saved_models\training_curves.png
  Saved → saved_models\Confusion_Matrix_-_Test_Set.png
  Saved → saved_models\Confusion_Matrix_-_Val_Set.png
  Saved → saved_models\ROC_Curve_-_Test_Set.png
  Saved → saved_models\ROC_Curve_-_Val_Set.png

Test AUC : 0.5139
Val AUC  : 0.3664

✓ All plots saved to saved_models


---
EfficientNet-B1 training terminal :-

=== Path Verification ===
Train/Test Root: ✓ Found → E:\dataset\Deepfake Detection Dataset\3. frames_dataset
   real: 1000 video folders
   fake: 1000 video folders
Validation Root: ✓ Found → E:\dataset\Deepfake Detection Dataset\5. Validation_Dataset_Frames
   real: 500 video folders
   fake: 500 video folders

=== Building Splits ===
Train : 1800 videos
Test  : 200  videos
Val   : 1000   videos
  Train → real: 900, fake: 900
  Test → real: 100, fake: 100
  Val → real: 500, fake: 500

Train batches : 450
Test batches  : 50
Val batches   : 250

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
EfficientNet-B1 — frozen: 180, unfrozen: 119 / 299
  Test → real: 100, fake: 100
  Val → real: 500, fake: 500

Train batches : 450
Test batches  : 50
Val batches   : 250

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
EfficientNet-B1 — frozen: 180, unfrozen: 119 / 299

Train batches : 450
Test batches  : 50
Val batches   : 250

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
EfficientNet-B1 — frozen: 180, unfrozen: 119 / 299
Test batches  : 50
Val batches   : 250

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
EfficientNet-B1 — frozen: 180, unfrozen: 119 / 299

Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
EfficientNet-B1 — frozen: 180, unfrozen: 119 / 299
Using device : cuda
GPU          : NVIDIA GeForce RTX 2050
EfficientNet-B1 — frozen: 180, unfrozen: 119 / 299
GPU          : NVIDIA GeForce RTX 2050
EfficientNet-B1 — frozen: 180, unfrozen: 119 / 299
EfficientNet-B1 — frozen: 180, unfrozen: 119 / 299
Trainable params : 7,136,854 / 8,000,866

=== Training ===

Epoch 1/20
  Train → Loss: 0.6965  Acc: 50.00%
  Val   → Loss: 0.6962  Acc: 50.00%
  ✓ Best model saved (val_loss: 0.6962)
                                                                                                                                                                
Epoch 2/20
  Train → Loss: 0.6949  Acc: 49.94%
  Val   → Loss: 0.6875  Acc: 50.00%
  ✓ Best model saved (val_loss: 0.6875)

Epoch 3/20
  Train → Loss: 0.6622  Acc: 64.33%
  Val   → Loss: 0.5967  Acc: 77.60%
  ✓ Best model saved (val_loss: 0.5967)

Epoch 4/20
  Train → Loss: 0.6276  Acc: 69.94%
  Val   → Loss: 0.5649  Acc: 79.30%
  ✓ Best model saved (val_loss: 0.5649)

Epoch 5/20
  Train → Loss: 0.6048  Acc: 72.11%
  Val   → Loss: 0.5258  Acc: 78.30%
  ✓ Best model saved (val_loss: 0.5258)

Epoch 6/20
  Train → Loss: 0.5571  Acc: 74.06%
  Val   → Loss: 0.4761  Acc: 79.30%
  ✓ Best model saved (val_loss: 0.4761)

Epoch 7/20
  Train → Loss: 0.5250  Acc: 76.17%
  Val   → Loss: 0.5161  Acc: 76.20%
  No improvement. Patience: 1/5

Epoch 8/20
  Train → Loss: 0.5162  Acc: 75.39%
  Val   → Loss: 0.4988  Acc: 76.00%
  No improvement. Patience: 2/5

Epoch 9/20
  Train → Loss: 0.4903  Acc: 78.56%
  Val   → Loss: 0.4555  Acc: 79.50%
  ✓ Best model saved (val_loss: 0.4555)

Epoch 10/20
  Train → Loss: 0.4533  Acc: 79.56%
  Val   → Loss: 0.5358  Acc: 73.60%
  No improvement. Patience: 1/5

Epoch 11/20
  Train → Loss: 0.4353  Acc: 81.56%
  Val   → Loss: 0.5443  Acc: 72.40%
  No improvement. Patience: 2/5

Epoch 12/20
  Train → Loss: 0.3911  Acc: 83.78%
  Val   → Loss: 0.4415  Acc: 77.60%
  ✓ Best model saved (val_loss: 0.4415)

Epoch 13/20
  Train → Loss: 0.3830  Acc: 83.17%
  Val   → Loss: 0.5870  Acc: 69.40%
  No improvement. Patience: 1/5

Epoch 14/20
  Train → Loss: 0.3429  Acc: 86.22%
  Val   → Loss: 0.7223  Acc: 64.20%
  No improvement. Patience: 2/5

Epoch 15/20
  Train → Loss: 0.3190  Acc: 86.83%
  Val   → Loss: 0.6290  Acc: 68.20%
  No improvement. Patience: 3/5

Epoch 16/20
  Train → Loss: 0.3195  Acc: 86.83%
  Val   → Loss: 0.5972  Acc: 71.40%
  No improvement. Patience: 4/5

Epoch 17/20
  Train → Loss: 0.2944  Acc: 87.50%
  Val   → Loss: 0.6218  Acc: 68.70%
  No improvement. Patience: 5/5

⚡ Early stopping triggered at epoch 17

✓ Training complete — best val loss: 0.4415

=== Loading Best Model ===
✓ Loaded from epoch 12  (val_loss: 0.4415)

=== Evaluation ===
Evaluating Test Set: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:36<00:00,  1.36it/s] 
Evaluating Val Set: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [01:17<00:00,  3.22it/s] 

==================================================
TEST SET — Classification Report
==================================================
              precision    recall  f1-score   support

        Real       0.59      0.68      0.63       100
        Fake       0.62      0.53      0.57       100

    accuracy                           0.60       200
   macro avg       0.61      0.60      0.60       200
weighted avg       0.61      0.60      0.60       200

==================================================
VALIDATION SET — Classification Report
==================================================
              precision    recall  f1-score   support

        Real       0.73      0.88      0.80       500
        Fake       0.85      0.67      0.75       500

    accuracy                           0.78      1000
   macro avg       0.79      0.78      0.77      1000
weighted avg       0.79      0.78      0.77      1000


=== Saving Plots ===
  Saved → saved_models\training_curves.png
  Saved → saved_models\Confusion_Matrix_-_Test_Set.png
  Saved → saved_models\Confusion_Matrix_-_Val_Set.png
  Saved → saved_models\ROC_Curve_-_Test_Set.png
  Saved → saved_models\ROC_Curve_-_Val_Set.png

Test AUC : 0.6472
Val AUC  : 0.8974

✓ All plots saved to saved_models



---
🔍 So What Did the Original Author Do?

There are only 3 possible ways they handled videos:

🔹 1. Frame-wise Prediction + Aggregation (MOST COMMON ✅)

Pipeline:

Video → Frames → CNN → Predictions per frame → Aggregate → Final output
Example:
Frame1 → 0.91 (fake)
Frame2 → 0.85 (fake)
Frame3 → 0.40 (real)
...

Then:

Average
Majority vote
Max confidence

👉 Final decision = video-level prediction

🔹 2. Frame Sampling (Single Frame)
Video → pick 1 frame → CNN → output

❌ Less accurate
✔ Simple

🔹 3. Feature Aggregation (Advanced)
Frames → CNN features → LSTM/Pooling → Classifier

👉 This is what YOU did with:

EfficientNet + LSTM ✅

--- 