import torch
from timm import create_model
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, unfreeze_ratio=0.4):
        super(CNNBlock, self).__init__()

        # Load pre-trained Xception model
        self.xception = create_model('xception', pretrained=True)

        # First freeze all layers
        for param in self.xception.parameters():
            param.requires_grad = False

        # Collect all named parameters
        all_params = list(self.xception.named_parameters())
        total      = len(all_params)
        n_unfreeze = int(total * unfreeze_ratio)  # 40% from the end

        # Unfreeze last 40% of layers
        for name, param in all_params[total - n_unfreeze:]:
            param.requires_grad = True

        # Log what got unfrozen
        frozen    = sum(1 for _, p in all_params if not p.requires_grad)
        unfrozen  = sum(1 for _, p in all_params if p.requires_grad)
        print(f"Xception — frozen: {frozen}, unfrozen: {unfrozen} / {total} params groups")

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Linear layer to reduce to 256 features
        self.fc = nn.Linear(2048, 256)

    def forward(self, x):
        x = self.xception.forward_features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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

        self.cnn = CNNBlock(unfreeze_ratio=0.4)
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
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")

    dummy  = torch.randn(2, 20, 3, 224, 224)
    output = model(dummy)
    print(f"\nInput shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print("Architecture check passed.")