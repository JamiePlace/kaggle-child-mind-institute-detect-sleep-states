from typing import Optional

import torch
import torch.nn as nn

from src.models.feature_extractor.cnn import CNNextractor
from src.models.encoder.context import ContextEncoder


class PrecTimeModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
    ):
        super().__init__()
        self.feature_extractor = CNNextractor(in_channels=in_channels)
        self.context_extractor = ContextEncoder(
            n_classes,
            input_size=200,
        )
        self.linear = nn.Linear(200, n_classes)
        self.loss_fn = nn.BCELoss()

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        features_flat, features_stacked = self.feature_extractor.forward(x)
        inter_window_context = self.context_extractor.forward(features_flat)
        linear = self.linear(inter_window_context)
        loss = self.loss_fn(linear, labels)

        output = {
            "inter_window_context": inter_window_context,
            "loss": loss,
        }

        return output
