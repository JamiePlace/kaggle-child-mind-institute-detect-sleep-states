from typing import Optional
from pathlib import Path
import polars as pl
import numpy as np
from tqdm import tqdm
import pickle
import os
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import gc

from src.conf import TrainConfig, InferenceConfig, PrepareDataConfig


def load_features(
    feature_names: list[str],
    series_ids: Optional[list[str]],
    series_id: Optional[str],
    processed_dir: Path,
    phase: str,
) -> dict[str, np.ndarray]:
    features = {}
    # this is a hack to allow us to use the validation data as inference data

    if series_id is None:
        for series_id in tqdm(series_ids):
            series_dir = processed_dir / phase / series_id

            this_feature = []
            for feature_name in feature_names:
                this_feature.append(
                    np.load(series_dir / f"{feature_name}.npy")
                )
            features[series_dir.name] = np.stack(this_feature, axis=1)
    else:
        series_dir = processed_dir / phase / series_id  # type: ignore
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        features[series_dir.name] = np.stack(this_feature, axis=1)

    return features


def split_array_into_chunks(
    cfg: PrepareDataConfig,
    array: np.ndarray,
    event_df: Optional[pl.DataFrame],
    window_size: int,
    phase: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # clean up the array such that we don't have data passed the last event

    num_rows = array.shape[0]
    num_chunks = num_rows // window_size
    remaining_rows = num_rows % window_size
    number_of_steps = array.shape[0]
    if event_df is not None and event_df.shape[0] == 0:
        return array, np.array(None), np.array(None), number_of_steps

    # for training we need the label
    if phase == "train":
        # create the label from the event_df
        label = np.zeros((array.shape[0], 3))
        # for each row of event_df find the corresponding rows in array from
        # event_df["onset"] to event_df["wakeup"] to set the label to 1
        last_wakeup = 0
        for row in event_df.rows(named=True):  # type: ignore
            label[row["onset"], 1] = 1
            label[row["wakeup"], 2] = 1
            label[last_wakeup : row["onset"], 0] = 1
            last_wakeup = row["wakeup"]

        # convolve the label with a gaussian kernel
        label = gaussian_label(label, cfg.dataset.offset, cfg.dataset.sigma)
        array = np.concatenate((array, label), axis=1)
    # Split the array into chunks of size window_size
    chunks = np.split(array[: num_chunks * window_size], num_chunks)

    # If there are remaining rows, create a chunk with the remaining rows and pad with zeros
    if remaining_rows > 0:
        remaining_chunk = np.concatenate(
            (
                array[num_chunks * window_size :],
                np.zeros((window_size - remaining_rows, array.shape[1])),
            )
        )
        chunks.append(remaining_chunk)
    chunks = np.array(chunks)

    if phase == "train":
        dense_labels = chunks[:, :, -3:]
        sparse_labels = np.array(
            [
                np.bincount(val).argmax()
                for val in chunks[:, :, -3:].argmax(axis=2)
            ]
        )

        return chunks[:, :, :-3], dense_labels, sparse_labels, 0
    return chunks, np.array(None), np.array(None), number_of_steps


def truncate_features(
    cfg: PrepareDataConfig, features: np.ndarray, event_df: pl.DataFrame
) -> np.ndarray:
    # find the step value of the last wakeup event
    last_wakeup_step = event_df.get_column("wakeup").max()
    # truncate features
    if last_wakeup_step is not None:
        max_len = len(features) - last_wakeup_step  # type: ignore
        features = features[: (last_wakeup_step + min(max_len, 500)), :]  # type: ignore
    return features


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
        if label[:, i].max() > 0:
            label[:, i] = label[:, i] / label[:, i].max()

    return label


###################
# PRE-PROCESS TRAINING DATA FOR TRAINING AND VALIDATION
###################
def pre_process_for_training(cfg: PrepareDataConfig):
    event_df_path = Path(cfg.dir.data_dir) / "train_events.csv"
    features_path = Path(cfg.dir.processed_dir)
    # load all of the features for all of the training data
    # we seperate out train and valid data at the end
    all_series_ids = cfg.split.train_series_ids + cfg.split.valid_series_ids
    # all_series_ids = cfg.split.train_series_ids
    features = load_features(
        feature_names=cfg.features,
        series_ids=all_series_ids,
        series_id=None,
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
        series_features = truncate_features(
            cfg, series_features, series_event_df
        )
        (
            series_chunks,
            dense_labels,
            sparse_labels,
            _,
        ) = split_array_into_chunks(
            cfg,
            series_features,
            series_event_df,
            cfg.dataset.window_size,
            phase="train",
        )
        if np.array_equal(series_chunks, series_features):
            continue
        if series_chunks.shape[0] % cfg.dataset.batch_size != 0:
            # pad the series_chunks with zeros
            pad_size = cfg.dataset.batch_size - (
                series_chunks.shape[0] % cfg.dataset.batch_size
            )
            series_chunks = np.concatenate(
                (
                    series_chunks,
                    np.zeros(
                        (
                            pad_size,
                            series_chunks.shape[1],
                            series_chunks.shape[2],
                        )
                    ),
                )
            )
            dense_labels = np.concatenate(
                (
                    dense_labels,
                    np.zeros(
                        (
                            pad_size,
                            dense_labels.shape[1],
                            dense_labels.shape[2],
                        )
                    ),
                )
            )
            sparse_labels = np.concatenate(
                (
                    sparse_labels,
                    np.zeros((pad_size)),
                )
            )

        batched_chunks = np.array_split(
            series_chunks, series_chunks.shape[0] // cfg.dataset.batch_size
        )

        batched_dense_labels = np.array_split(
            dense_labels, dense_labels.shape[0] // cfg.dataset.batch_size
        )

        batched_sparse_labels = np.array_split(
            sparse_labels, sparse_labels.shape[0] // cfg.dataset.batch_size
        )

        # for each chunk, save the chunk and the label
        for i, (chunk, dense_label, sparse_label) in enumerate(
            zip(batched_chunks, batched_dense_labels, batched_sparse_labels)
        ):
            # use sparse label in the file name so that we can easily filter
            key = f"{series_id}_{i:07}"
            file_name = f"{series_id}_{i:07}.pkl"
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


# pre process data for inference. no event_df is needed
def pre_process_for_inference(cfg: PrepareDataConfig):
    features_path = Path(cfg.dir.processed_dir)
    # load all of the features for all of the training data
    # all_series_ids = cfg.split.train_series_ids + cfg.split.valid_series_ids
    if cfg.phase == "train":
        event_df_path = Path(cfg.dir.data_dir) / "train_events.csv"
        all_series_ids = cfg.split.valid_series_ids
        event_df = (
            pl.read_csv(event_df_path)
            .drop_nulls()
            .pivot(
                index=["series_id", "night"], columns="event", values="step"
            )
            .filter(pl.col("series_id").is_in(all_series_ids))
            .drop_nulls()
        )
    else:
        all_series_ids = [
            series_dir.name
            for series_dir in (features_path / cfg.phase).glob("*")
        ]
        event_df = None

    output_path = Path(cfg.dir.processed_dir) / "inference/"

    inference_keys = []
    series_length = {}
    for series_id in tqdm(all_series_ids):
        features = load_features(
            feature_names=cfg.features,
            series_ids=None,
            series_id=series_id,
            processed_dir=features_path,
            phase=cfg.phase,
        )

        series_features = features[series_id]
        if event_df is not None:
            series_event_df = event_df.filter(pl.col("series_id") == series_id)
            if series_event_df.shape[0] == 0:
                continue
        else:
            series_event_df = None
        series_chunks, _, _, number_of_steps = split_array_into_chunks(
            cfg,
            series_features,
            series_event_df,
            cfg.dataset.window_size,
            phase=cfg.phase,
        )
        if series_chunks.shape[0] % cfg.dataset.batch_size != 0:
            # pad the series_chunks with zeros
            pad_size = cfg.dataset.batch_size - (
                series_chunks.shape[0] % cfg.dataset.batch_size
            )
            series_chunks = np.concatenate(
                (
                    series_chunks,
                    np.zeros(
                        (
                            pad_size,
                            series_chunks.shape[1],
                            series_chunks.shape[2],
                        )
                    ),
                )
            )
        series_length[series_id] = number_of_steps

        batched_chunks = np.array_split(
            series_chunks, series_chunks.shape[0] // cfg.dataset.batch_size
        )
        # for each chunk, save the chunk and the label
        for i, chunk in enumerate(batched_chunks):
            key = f"{series_id}_{i:07}"
            file_name = f"{series_id}_{i:07}.pkl"
            fileobj = open(output_path / file_name, "wb")
            pickle.dump(
                {
                    "key": key,
                    "feature": chunk,
                },
                fileobj,
            )
            fileobj.close()
            inference_keys.append(key)
    # write keys to file so don't have to scan directory to get keys
    file_name = "__inference_keys__.pkl"
    fileobj = open(output_path / file_name, "wb")
    pickle.dump(inference_keys, fileobj)
    fileobj.close()

    with open(output_path / "__series_length__.pkl", "wb") as f:
        pickle.dump(series_length, f)

    gc.collect()


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

        self.train_data_files = np.random.choice(
            self.train_data_files, len(self.train_data_files), replace=False
        )  # type: ignore

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


class TestDataset(Dataset):
    def __init__(
        self,
        cfg: InferenceConfig,
    ):
        self.cfg = cfg
        keys_file = (
            Path(cfg.dir.processed_dir)
            / "inference"
            / "__inference_keys__.pkl"
        )
        fileobj = open(keys_file, "rb")
        self.valid_data_files = pickle.load(fileobj)
        fileobj.close()
        with open(
            Path(cfg.dir.processed_dir)
            / "inference"
            / "__series_length__.pkl",
            "rb",
        ) as f:
            self.series_length = pickle.load(f)

    def __len__(self):
        return len(self.valid_data_files)

    def __getitem__(self, idx):
        data_path = Path(self.cfg.dir.processed_dir) / "inference"
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
            batch_size=1,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = ValidDataset(cfg=self.cfg)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return valid_loader
