from typing import Any, Optional

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchvision.transforms.functional import resize
from transformers import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from src.conf import TrainConfig
from src.datamodule.seg import nearest_valid_size
from src.models.common import get_model
from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg


class PrecTimeModel(LightningModule):
    def __init__(
        self,
        cfg: TrainConfig,
        val_event_df: pl.DataFrame,
        feature_dim: int,
        num_classes: int,
        duration: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.val_event_df = val_event_df
        self.model = get_model(
            cfg,
            feature_dim=feature_dim,
            n_classes=num_classes,
            num_timesteps=duration,
        )
