from typing import Optional
from pathlib import Path
import polars as pl
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from src.conf import TrainConfig


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


def split_array_into_chunks(
    array: np.ndarray, event_df: pl.DataFrame, window_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # clean up the array such that we don't have data passed the last event
    array = array[: event_df["wakeup"].max(), :]

    num_rows = array.shape[0]
    num_chunks = num_rows // window_size
    remaining_rows = num_rows % window_size

    # create the label from the event_df
    label = np.zeros(array.shape[0])
    # for each row of event_df find the corresponding rows in array from
    # event_df["onset"] to event_df["wakeup"] to set the label to 1
    for row in event_df.rows(named=True):
        label[row["onset"] : row["wakeup"] + 1] = 1

    array_with_label = np.concatenate((array, label[:, None]), axis=1)
    # Split the array into chunks of size window_size
    chunks = np.split(array_with_label[: num_chunks * window_size], num_chunks)

    # If there are remaining rows, create a chunk with the remaining rows and pad with zeros
    if remaining_rows > 0:
        remaining_chunk = np.concatenate(
            (
                array_with_label[num_chunks * window_size :],
                np.zeros(
                    (window_size - remaining_rows, array_with_label.shape[1])
                ),
            )
        )
        chunks.append(remaining_chunk)
    chunks = np.array(chunks)

    dense_labels = chunks[:, :, -1]
    sparse_labels = np.array(
        [
            0 if val < window_size // 2 else 1
            for val in chunks[:, :, -1].sum(axis=1)
        ]
    )

    return chunks[:, :, :-1], dense_labels, sparse_labels


# caclulate the number of positive and negative files in the training data
# directory
def calculate_class_weights(cfg: TrainConfig) -> list[float]:
    train_data_dir = Path(cfg.dir.processed_dir) / "train"
    train_data_files = [
        train_file.name for train_file in train_data_dir.glob("*.pkl")
    ]
    pos_files = [file for file in train_data_files if "pos" in file.split("_")]
    neg_files = [file for file in train_data_files if "neg" in file.split("_")]
    return [len(neg_files), len(pos_files)]


###################
# PRE-PROCESS TRAINING DATA FOR TRAINING AND VALIDATION
###################
def pre_process_for_training(cfg: TrainConfig):
    event_df_path = Path(cfg.dir.data_dir) / "train_events.csv"
    features_path = Path(cfg.dir.processed_dir)
    # load all of the features for all of the training data
    # all_series_ids = cfg.split.train_series_ids + cfg.split.valid_series_ids
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
        .filter(pl.col("series_id").is_in(all_series_ids))
        .drop_nulls()
    )

    output_path = Path(cfg.dir.processed_dir) / "train/"
    train_keys = []
    valid_keys = []
    for series_id in tqdm(all_series_ids):
        series_features = features[series_id]
        series_event_df = event_df.filter(pl.col("series_id") == series_id)
        series_chunks, dense_labels, sparse_labels = split_array_into_chunks(
            series_features,
            series_event_df,
            cfg.window_size,
        )
        # for each chunk, save the chunk and the label
        for i, (chunk, dense_label, sparse_label) in enumerate(
            zip(series_chunks, dense_labels, sparse_labels)
        ):
            # use sparse label in the file name so that we can easily filter
            sparse_name = "pos" if sparse_label == 1 else "neg"
            key = f"{series_id}_{sparse_name}_{i:07}"
            file_name = f"{series_id}_{sparse_name}_{i:07}.pkl"
            fileobj = open(output_path / file_name, "wb")
            pickle.dump(
                {
                    "key": key,
                    "feature": chunk,
                    "dense_label": dense_label,
                    "sparse_label": sparse_label,
                },
                fileobj,
            )
            fileobj.close()
            if series_id in cfg.split.train_series_ids:
                train_keys.append(key)
            else:
                valid_keys.append(key)
    # write keys to file so don't have to scan directory to get keys
    file_name = "__train_keys__.pkl"
    fileobj = open(output_path / file_name, "wb")
    pickle.dump(train_keys, fileobj)
    fileobj.close()

    file_name = "__valid_keys__.pkl"
    fileobj = open(output_path / file_name, "wb")
    pickle.dump(valid_keys, fileobj)
    fileobj.close()


class TrainDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
    ):
        self.train_data_files: list[str]
        self.cfg = cfg
        keys_file = (
            Path(cfg.dir.processed_dir) / "train" / "__train_keys__.pkl"
        )
        fileobj = open(keys_file, "rb")
        self.train_data_files = pickle.load(fileobj)
        fileobj.close()
        # filter the train_data_files to include a proportion of negative to positive examples
        # extrac the positive examples and negative examples
        pos_files = [
            file for file in self.train_data_files if "pos" in file.split("_")
        ]
        neg_files = [
            file for file in self.train_data_files if "neg" in file.split("_")
        ]
        # TODO - if subsample is true then ensure that samples are taken in order from the files
        if cfg.subsample:
            # subsample the positive examples
            # get the ratio of pos to neg from the config
            n = len(neg_files) - cfg.dataset.positive_to_negative_ratio * len(
                pos_files
            )
            neg_files = np.random.choice(neg_files, size=int(n), replace=False)
            self.train_data_files = list(pos_files) + list(neg_files)
            subsample_size = int(
                len(self.train_data_files) * cfg.subsample_rate
            )
            self.train_data_files = list(
                np.random.choice(self.train_data_files, size=subsample_size)
            )

    def __len__(self):
        return len(self.train_data_files)

    def __getitem__(self, idx):
        data_path = Path(self.cfg.dir.processed_dir) / "train"
        file_name = self.train_data_files[idx] + ".pkl"

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
        keys_file = (
            Path(cfg.dir.processed_dir) / "train" / "__valid_keys__.pkl"
        )
        fileobj = open(keys_file, "rb")
        self.valid_data_files = pickle.load(fileobj)
        fileobj.close()

    def __len__(self):
        return len(self.valid_data_files)

    def __getitem__(self, idx):
        data_path = Path(self.cfg.dir.processed_dir) / "train"
        file_name = self.valid_data_files[idx] + ".pkl"

        fileobj = open(data_path / file_name, "rb")
        output = pickle.load(fileobj)
        fileobj.close()

        return output


###################
# DataModule
###################
class PrecTimeDataModule(LightningDataModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        train_dataset = TrainDataset(
            cfg=self.cfg,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
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
            persistent_workers=True,
        )
        return valid_loader
