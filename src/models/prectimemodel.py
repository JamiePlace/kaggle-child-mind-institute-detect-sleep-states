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
        self.cfg = cfg
        base_filters = 128
        self.feature_extractor = CNNextractor(
            in_channels=in_channels, base_filters=base_filters
        )
        self.context_extractor = ContextEncoder(
            input_size=cfg.window_size * 128,
        )
        if 200 > cfg.window_size:
            self.upsample_or_downsample = nn.AdaptiveAvgPool1d(cfg.window_size)
        else:
            self.upsample_or_downsample = nn.Upsample(
                size=(base_filters, cfg.window_size),
                mode="bilinear",
                align_corners=True,
            )
        self.prediction_refinor = CNNrefinor(
            cfg, in_channels=cfg.window_size * 2, base_filters=base_filters
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
        x1 = self.context_extractor(x1)
        sparse_prediction = self.sigmoid_sparse(x1[:, -1])
        x1 = self.upsample_or_downsample(x1)
        new_x = torch.zeros(x1.shape[0], x2.shape[1] * 2, x2.shape[2] * 2).to(
            x1.device
        )
        new_x[:, : x2.shape[1], : x2.shape[2]] = x2
        new_x[:, x2.shape[1] :, x2.shape[2] :] = x1
        new_x = self.prediction_refinor(new_x)
        dense_prediction = self.sigmoid_dense(new_x)
        return {
            "sparse_predictions": sparse_prediction.squeeze(),
            "dense_predictions": dense_prediction.squeeze(),
        }
