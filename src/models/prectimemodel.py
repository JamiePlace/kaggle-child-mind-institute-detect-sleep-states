from typing import Optional

import torch
import torch.nn as nn

from src.models.feature_extractor.cnn import CNNextractor, CNNrefinor
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
        base_filters = 256
        self.feature_extractor = CNNextractor(
            in_channels=in_channels, base_filters=base_filters
        )
        # self.context_extractor = ContextEncoder(
        #    input_size=cfg.window_size * 128,
        # )
        self.prediction_refinor = CNNrefinor(
            in_channels=base_filters, base_filters=base_filters
        )
        self.fc_sparse = nn.Linear(cfg.window_size * base_filters, n_classes)
        self.fc_dense = nn.Linear(
            cfg.window_size * base_filters, cfg.window_size
        )
        self.sigmoid_sparse = nn.Sigmoid()
        self.sigmoid_dense = nn.Sigmoid()

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
        x1, x2 = self.feature_extractor(x)
        # x = self.context_extractor(x)
        inter_window_context = x1
        x1 = self.fc_sparse(x1)
        sparse_prediction = self.sigmoid_sparse(x1)

        x2 = self.prediction_refinor(x2)
        x2 = self.fc_dense(x2)
        dense_prediction = self.sigmoid_dense(x2)
        return {
            "sparse_predictions": sparse_prediction.squeeze(),
            "dense_predictions": dense_prediction.squeeze(),
        }
