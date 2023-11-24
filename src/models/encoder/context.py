import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 100
        self.dropout = 0
        self.lstm1 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
