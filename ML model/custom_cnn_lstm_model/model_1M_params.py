import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self):
        super(CNNBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                          
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)   
        return x


class DeepFakeDetector(nn.Module):

    def __init__(
        self,
        sequence_length=20,
        cnn_feature_dim=256,      
        lstm_hidden_size=256,
        lstm_num_layers=2,
        lstm_dropout=0.3,
        fc_hidden_1=256,         
        fc_hidden_2=128,        
        fc_hidden_3=64,        
        num_classes=2
    ):
        super(DeepFakeDetector, self).__init__()

        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers  = lstm_num_layers

        self.cnn = CNNBlock()
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, fc_hidden_1),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(fc_hidden_1, fc_hidden_2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(fc_hidden_2, fc_hidden_3),
            nn.ReLU(),

            nn.Linear(fc_hidden_3, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        cnn_features = self.cnn(x)                      
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        last_hidden = h_n[-1]                             
        out = self.classifier(last_hidden)                 
        return out

if __name__ == "__main__":
    model = DeepFakeDetector()
    print(model)
    print("\n--- Parameter Count ---")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")

    dummy = torch.randn(2, 20, 3, 224, 224)
    output = model(dummy)
    print(f"\nInput shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")  
    print("Architecture check passed.")