from pathlib import Path
from typing import Union
import torch
import torch.nn as nn

from src.conf import InferenceConfig, TrainConfig
from src.models.prectimemodel import PrecTimeModel


def get_model(
    train_cfg: TrainConfig | InferenceConfig,
    feature_dim: int,
    n_classes: int,
) -> PrecTimeModel:
    model: PrecTimeModel
    model = PrecTimeModel(train_cfg, feature_dim, n_classes)
    return model
