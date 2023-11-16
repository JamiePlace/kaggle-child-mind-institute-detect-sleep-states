import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
import pickle
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize

from src.conf import InferenceConfig, TrainConfig, PrepareDataConfig
from src.utils.common import pad_if_needed


###################
# Load Functions
###################
def load_features(
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [
            series_dir.name for series_dir in (processed_dir / phase).glob("*")
        ]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        features[series_dir.name] = np.stack(this_feature, axis=1)

    return features


def load_chunk_features(
    duration: int,
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [
            series_dir.name for series_dir in (processed_dir / phase).glob("*")
        ]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        this_feature = np.stack(this_feature, axis=1)
        num_chunks = (len(this_feature) // duration) + 1
        for i in range(num_chunks):
            chunk_feature = this_feature[i * duration : (i + 1) * duration]
            chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)  # type: ignore
            features[f"{series_id}_{i:07}"] = chunk_feature

    return features  # type: ignore


###################
# Augmentation
###################
def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(
        max(0, pos - duration), min(pos, max_end - duration)
    )
    end = start + duration
    return start, end


###################
# Label
###################
def get_label(
    this_event_df: pd.DataFrame,
    num_frames: int,
    duration: int,
    start: int,
    end: int,
) -> np.ndarray:
    # # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    label = np.zeros((num_frames, 3))
    # onset, wakeup, sleepのラベルを作成
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset = int((onset - start) / duration * num_frames)
        wakeup = int((wakeup - start) / duration * num_frames)
        if onset >= 0 and onset < num_frames:
            label[onset, 1] = 1
        if wakeup < num_frames and wakeup >= 0:
            label[wakeup, 2] = 1

        onset = max(0, onset)
        wakeup = min(num_frames, wakeup)
        label[onset:wakeup, 0] = 1  # sleep

    return label


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(
            label[:, i], gaussian_kernel(offset, sigma), mode="same"
        )

    return label


def negative_sampling(this_event_df: pd.DataFrame, num_steps: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    # onsetとwakupを除いた範囲からランダムにサンプリング
    positive_positions = set(
        this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist()
    )
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]


###################
# PRE-PROCESS TRAINING DATA FOR TRAINING AND VALIDATION
###################
def pre_process_for_training(cfg: TrainConfig):
    event_df_path = Path(cfg.dir.data_dir) / "train_events.csv"
    features_path = Path(cfg.dir.processed_dir)
    num_features = len(cfg.features)
    upsampled_num_frames = nearest_valid_size(
        int(cfg.duration * cfg.upsample_rate),
        cfg.downsample_rate,
    )
    # load all of the features for all of the training data
    all_series_ids = cfg.split.train_series_ids + cfg.split.valid_series_ids
    features = load_features(
        feature_names=cfg.features,
        series_ids=all_series_ids,
        processed_dir=features_path,
        phase="train",
    )
    event_df = (
        pl.read_csv(event_df_path)
        .drop_nulls()
        .pivot(index=["series_id", "night"], columns="event", values="step")
        .drop_nulls()
        .to_pandas()
    )

    output_path = Path(cfg.dir.processed_dir) / "train/"
    for i in tqdm(range(len(event_df))):
        event = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])
        pos = event_df.at[i, event]
        series_id = event_df.at[i, "series_id"]
        this_event_df = event_df.query("series_id == @series_id").reset_index(
            drop=True
        )
        this_feature = features[series_id]
        n_rows = this_feature.shape[0]
        # should this be done such that there are an equal number of
        # negative periods as positive periods?
        # sample background
        if random.random() < cfg.dataset.bg_sampling_rate:
            pos = negative_sampling(this_event_df, n_rows)

        # crop
        start, end = random_crop(pos, cfg.duration, n_rows)
        feature = this_feature[start:end]  # (duration, num_features)
        # upsample
        # this is upsampling from 5 min data to 1 min data
        feature = torch.FloatTensor(feature.T).unsqueeze(
            0
        )  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[num_features, upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        # from hard label to gaussian label
        num_frames = upsampled_num_frames // cfg.downsample_rate
        label = get_label(this_event_df, num_frames, cfg.duration, start, end)
        label[:, [1, 2]] = gaussian_label(
            label[:, [1, 2]],
            offset=cfg.dataset.offset,
            sigma=cfg.dataset.sigma,
        )
        output = {
            "series_id": series_id,
            "feature": feature,
            "label": torch.FloatTensor(label),
        }
        file_name = f"{series_id}_{i:07}.pkl"
        fileobj = open(output_path / file_name, "wb")
        pickle.dump(output, fileobj)
        fileobj.close()


###################
# Dataset
###################
def nearest_valid_size(input_size: int, downsample_rate: int) -> int:
    """
    (x // hop_length) % 32 == 0
    を満たすinput_sizeに最も近いxを返す
    """

    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0

    return input_size


# TODO this method loads the features and determines the label on each pass
# through this is too much. Have the features for training and the labels
# determined before we get to the generation of the dataset object. This will
# greatly improve efficiency which is the main problem at the moment.
# look at updating to datapipes and pickling
# https://medium.com/deelvin-machine-learning/comparison-of-pytorch-dataset-and-torchdata-datapipes-486e03068c58
class TrainDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
    ):
        self.cfg = cfg

        self.train_data_files = [
            train_file.name
            for train_file in (Path(cfg.dir.processed_dir) / "train").glob(
                "*.pkl"
            )
            if train_file.name.split("_")[0] in cfg.split.train_series_ids
        ]

    def __len__(self):
        return len(self.train_data_files)

    def __getitem__(self, idx):
        data_path = Path(self.cfg.dir.processed_dir) / "train"
        file_name = self.train_data_files[idx]

        fileobj = open(data_path / file_name, "rb")
        output = pickle.load(fileobj)
        fileobj.close()

        return output


class ValidDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
    ):
        self.cfg = cfg
        self.valid_data_files = [
            valid_file.name
            for valid_file in (Path(cfg.dir.processed_dir) / "train").glob(
                "*.pkl"
            )
            if valid_file.name.split("_")[0] in cfg.split.valid_series_ids
        ]

    def __len__(self):
        return len(self.valid_data_files)

    def __getitem__(self, idx):
        data_path = Path(self.cfg.dir.processed_dir) / "train"
        file_name = self.valid_data_files[idx]

        fileobj = open(data_path / file_name, "rb")
        output = pickle.load(fileobj)
        fileobj.close()

        return output


class TestDataset(Dataset):
    def __init__(
        self,
        cfg: InferenceConfig,
        chunk_features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate),
            self.cfg.downsample_rate,
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(
            0
        )  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
        }


###################
# DataModule
###################
class SegDataModule(LightningDataModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.processed_dir = Path(cfg.dir.processed_dir)
        self.event_df = pl.read_csv(
            self.data_dir / "train_events.csv"
        ).drop_nulls()
        self.train_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.train_series_ids)
        )
        self.valid_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.valid_series_ids)
        )

    def train_dataloader(self):
        train_dataset = TrainDataset(
            cfg=self.cfg,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = ValidDataset(cfg=self.cfg)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
        )
        return valid_loader
