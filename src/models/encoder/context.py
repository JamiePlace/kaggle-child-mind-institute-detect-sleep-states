import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    def __init__(
        self,
        n_classes: int,
        input_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 100
        self.dropout = 0
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.lstm1 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.fc2 = nn.Linear(self.hidden_size * 2, n_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x, _ = self.lstm1(x)
        print(x.shape)
        x = self.fc2(x)
        return self.softmax(x)
