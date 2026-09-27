import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=117,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):

        # x:
        # B × 100 × 117

        _, (h_n, _) = self.lstm(x)

        # h_n:
        # num_layers × B × hidden_size

        # Take final LSTM layer
        last_hidden = h_n[-1]

        # B × hidden_size
        return self.fc(last_hidden)