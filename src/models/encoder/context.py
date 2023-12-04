from collections import OrderedDict
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
    ):
        super().__init__()
        self.input_size = input_size
        hidden_size1 = 50
        hidden_size2 = 100
        self.dropout = 0
        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size1,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm2 = nn.LSTM(
            hidden_size1 * 2,
            hidden_size2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

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
        x, (h_0, c_0) = self.lstm1(x)
        # reshape due to bi directional
        bi_h_0 = torch.zeros(h_0.shape[0], h_0.shape[1], h_0.shape[2] * 2).to(
            x.device
        )
        bi_c_0 = torch.zeros(c_0.shape[0], c_0.shape[1], c_0.shape[2] * 2).to(
            x.device
        )
        bi_h_0[:, :, : int(h_0.shape[2])] = h_0
        bi_h_0[:, :, int(h_0.shape[2]) :] = torch.flip(h_0, [2])
        bi_c_0[:, :, : int(c_0.shape[2])] = c_0
        bi_c_0[:, :, int(c_0.shape[2]) :] = torch.flip(c_0, [2])
        x, _ = self.lstm2(x, (bi_h_0, bi_c_0))
        x = x.squeeze()
        return x
