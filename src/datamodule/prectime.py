from typing import Optional
from pathlib import Path
import polars as pl
import numpy as np
from tqdm import tqdm
import pickle

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
    num_rows = array.shape[0]
    num_chunks = num_rows // window_size
    remaining_rows = num_rows % window_size

    # clean up the array such that we don't have data passed the last event
    array = array[: event_df["wakeup"].max(), :]
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
            file_name = f"{series_id}_{i:07}.pkl"
            fileobj = open(output_path / file_name, "wb")
            pickle.dump(
                {
                    "key": f"{series_id}_{i:07}",
                    "feature": chunk,
                    "dense_label": dense_label,
                    "sparse_label": sparse_label,
                },
                fileobj,
            )
            fileobj.close()
