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
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_features, seq_len)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        # we need to combine the features to be sequential. The features
        # from one window to another are what we are trying to look at.
        # decompose batch into a single batch
        # (batch_size, n_features, seq_len) ->
        # (batch_size, seq_len, n_features)
        # we can do this by swapping the second and third dimensions
        # and declaring the first dimension is not the batch size
        x = x.view(1, x.shape[0], x.shape[1])
        x, _ = self.lstm1(x)
        x = x.squeeze()
        return x
