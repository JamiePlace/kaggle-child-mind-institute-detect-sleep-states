from typing import Optional

import torch
import torch.nn as nn

from src.models.feature_extractor.cnn import CNNextractor
from src.models.encoder.context import ContextEncoder
from src.conf import TrainConfig


class PrecTimeModel(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        in_channels: int,
        n_classes: int,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.feature_extractor = CNNextractor(in_channels=in_channels)
        # self.context_extractor = ContextEncoder(
        #    input_size=cfg.window_size * 128,
        # )
        self.fc = nn.Linear(cfg.window_size * 128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        x, _ = self.feature_extractor(x)
        # x = self.context_extractor(x)
        inter_window_context = x
        x = self.fc(x)
        x = self.sigmoid(x)
        return {
            "inter_window_context": inter_window_context,
            "predictions": x.squeeze(),
        }
