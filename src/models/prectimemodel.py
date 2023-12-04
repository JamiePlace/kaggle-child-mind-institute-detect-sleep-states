from typing import Optional

import torch
import torch.nn as nn

from src.models.feature_extractor.cnn import CNNextractor, CNNrefinor
from src.models.encoder.context import ContextEncoder
from src.conf import TrainConfig, InferenceConfig


class PrecTimeModel(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig | InferenceConfig,
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
            input_size=cfg.dataset.window_size * base_filters,
        )
        self.fc_sparse = nn.Linear(200, n_classes)
        self.upsample_or_downsample = nn.AdaptiveAvgPool1d(
            cfg.dataset.window_size
        )
        self.prediction_refinor = CNNrefinor(
            cfg,
            in_channels=base_filters + 1,
            base_filters=base_filters,
        )
        self.fc_dense = nn.Linear(
            base_filters * cfg.dataset.window_size * 2,
            cfg.dataset.window_size * n_classes,
        )
        self.softmax_sparse = nn.Softmax(dim=1)
        self.softmax_dense = nn.Softmax(dim=1)

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
        sparse_prediction = self.fc_sparse(x1)
        sparse_prediction = self.softmax_sparse(sparse_prediction)
        x1 = self.upsample_or_downsample(x1)
        x1 = x1.view(x1.shape[0], 1, x1.shape[1])
        new_x = torch.zeros(x1.shape[0], x2.shape[1] + 1, x2.shape[2]).to(
            x1.device
        )
        new_x[:, : x2.shape[1], :] = x2
        new_x[:, x2.shape[1] :, :] = x1
        new_x = self.prediction_refinor(new_x)
        new_x = self.fc_dense(new_x)
        new_x = new_x.view(
            new_x.shape[0], self.cfg.dataset.window_size, self.n_classes
        )
        dense_prediction = self.softmax_dense(new_x)
        return {
            "sparse_predictions": sparse_prediction.squeeze(),
            "dense_predictions": dense_prediction.squeeze(),
        }
