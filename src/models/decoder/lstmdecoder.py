import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        n_classes: int,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size * 2, n_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.linear(x)
        return self.softmax(x)
