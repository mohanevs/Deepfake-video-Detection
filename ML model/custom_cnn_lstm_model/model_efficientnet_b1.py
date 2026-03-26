import torch
import torch.nn as nn
from timm import create_model


class CNNBlock(nn.Module):
    def __init__(self, unfreeze_ratio=0.4):
        super(CNNBlock, self).__init__()

        # Load EfficientNet-B1, remove classifier head
        self.efficientnet = create_model(
            'efficientnet_b1',
            pretrained=True,
            num_classes=0,        # removes the FC head automatically
            global_pool='avg'     # keeps GAP, outputs (B, 1280)
        )

        # Freeze all layers first
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Unfreeze last 40% of layers
        all_params = list(self.efficientnet.named_parameters())
        total      = len(all_params)
        n_unfreeze = int(total * unfreeze_ratio)

        for name, param in all_params[total - n_unfreeze:]:
            param.requires_grad = True

        frozen   = sum(1 for _, p in all_params if not p.requires_grad)
        unfrozen = sum(1 for _, p in all_params if p.requires_grad)
        print(f"EfficientNet-B1 — frozen: {frozen}, unfrozen: {unfrozen} / {total}")

        # Project 1280 → 256 to keep LSTM input size manageable
        self.proj = nn.Linear(1280, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.efficientnet(x)   # (B, 1280) — GAP already applied
        x = self.relu(self.proj(x)) # (B, 256)
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

        self.sequence_length  = sequence_length
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
            nn.Dropout(0.2),

            nn.Linear(fc_hidden_1, fc_hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(fc_hidden_2, fc_hidden_3),
            nn.ReLU(),

            nn.Linear(fc_hidden_3, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        x = x.view(batch_size * seq_len, C, H, W)
        cnn_features = self.cnn(x)                        # (B*20, 256)

        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # (B, 20, 256)
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        last_hidden = h_n[-1]                              # (B, 256)

        out = self.classifier(last_hidden)                 # (B, 2)
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