import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import hydra
import polars as pl
from rich import print
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

from src.utils.metrics import event_detection_ap
from src.conf import InferenceConfig, TrainConfig


def get_label(cfg: TrainConfig, series_id: str):
    with open(
        Path(cfg.dir.processed_dir) / "train" / "__valid_keys__.pkl", "rb"
    ) as f:
        valid_keys = pickle.load(f)

    with open(
        Path(cfg.dir.processed_dir) / "train" / "__train_keys__.pkl", "rb"
    ) as f:
        train_keys = pickle.load(f)

    all_keys = valid_keys + train_keys

    series_df = (
        pl.scan_parquet(
            Path(cfg.dir.data_dir) / "train_series.parquet",
            low_memory=True,
        )
        .filter(pl.col("series_id") == series_id)
        .collect(streaming=True)
        .select([pl.col("timestamp"), pl.col("anglez"), pl.col("enmo")])
        .with_columns(
            pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z")
        )
    )
    timestamp = series_df.get_column("timestamp")
    max_time = (
        pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv")
        .drop_nulls()
        .filter(pl.col("series_id") == series_id)
        .drop_nulls()
        .filter(pl.col("event") == "wakeup")
        .select(pl.col("step"))
        .max()
    )

    files = [
        file + ".pkl" for file in all_keys if file.split("_")[0] == series_id
    ]
    sparse_label = []
    dense_label = []
    anglez = []
    enmo = []
    for file in files:
        with open(Path(cfg.dir.processed_dir) / "train" / file, "rb") as f:
            batch = pickle.load(f)
        sparse_label.append(batch["sparse_label"].flatten())
        dense_label.append(batch["dense_label"].flatten())
        anglez.append(batch["feature"][:, :, 0].flatten())
        enmo.append(batch["feature"][:, :, 1].flatten())
    dense_label = np.concatenate(dense_label)
    anglez = np.concatenate(anglez)
    enmo = np.concatenate(enmo)

    return (sparse_label, dense_label, anglez, enmo, timestamp, max_time)


def expand_sparse_label(
    cfg: InferenceConfig,
    sparse_label: list | np.ndarray,
    dense_label: list | np.ndarray,
):
    expanded_label = np.zeros(len(dense_label))
    for i, label in enumerate(sparse_label):
        expanded_label[
            i * cfg.dataset.window_size : (i + 1) * cfg.dataset.window_size
        ] = label
    expanded_label = expanded_label[: len(dense_label)]
    return expanded_label


# finish calculating the score
def calculate_score(cfg: TrainConfig, series_id: str | None = None):
    pred_df = pl.read_csv(Path(cfg.dir.sub_dir) / "submission.csv")
    if series_id:
        pred_df = pred_df.filter(pl.col("series_id") == series_id)
    event_df = (
        pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv")
        .drop_nulls()
        .filter(pl.col("series_id").is_in(pred_df["series_id"].unique()))
        .drop_nulls()
    )
    print(event_df)
    print(pred_df)
    if series_id:
        return event_detection_ap(event_df.to_pandas(), pred_df.to_pandas())
    valid_event_df = event_df.filter(
        pl.col("series_id").is_in(cfg.split.valid_series_ids)
    ).to_pandas()

    valid_pred_df = pred_df.filter(
        pl.col("series_id").is_in(cfg.split.valid_series_ids)
    ).to_pandas()
    valid_score = event_detection_ap(valid_event_df, valid_pred_df)
    return valid_score


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    # Load the data
    with open(
        Path(cfg.dir.processed_dir) / "inference/__series_length__.pkl", "rb"
    ) as f:
        series_length = pickle.load(f)
    with open(Path(cfg.dir.sub_dir) / "predictions.pkl", "rb") as f:
        data = pickle.load(f)

    # Plot the data
    if cfg.dataset.series_ids[0]:
        series_id = cfg.dataset.series_ids[0]
    else:
        series_id = np.random.choice(list(series_length.keys()))

    # series_id = np.random.choice(list(series_length.keys()))
    print(series_id)

    valid_score = calculate_score(cfg)  # type: ignore
    print(f"valid score: {valid_score:.4f}")

    score = calculate_score(cfg, series_id)
    print(f"score: {score:.4f}")

    (
        sparse_label,
        dense_label,
        anglez,
        enmo,
        timestamp,
        this_series_length,
    ) = get_label(cfg, series_id)
    timestamp = timestamp.to_numpy()
    this_series_length = len(anglez)
    # this_series_length = 15000

    pred_df = pl.read_csv(Path(cfg.dir.sub_dir) / "submission.csv")
    pred_df = pred_df.filter(pl.col("series_id") == series_id)
    group = np.arange(len(pred_df) // 2)
    group = np.repeat(group, 2)
    if len(pred_df) % 2 != 0:
        group = np.append(group, group[-1] + 1)
    pred_df = pred_df.with_columns(pl.Series(name="group", values=group))

    dense_preds = np.zeros(this_series_length)
    pred_df = pred_df.pivot(
        index=["series_id", "group"], columns="event", values="step"
    )

    for row in pred_df.rows(named=True):
        dense_preds[row["onset"] : row["wakeup"]] = 1
    dense_preds = dense_preds[:this_series_length]
    dense_label = dense_label[:this_series_length]
    dense_logits = data[series_id][:this_series_length]
    anglez = anglez[:this_series_length]
    enmo = enmo[:this_series_length]
    timestamp = timestamp[:this_series_length]

    # insert axvlines at every window partition
    window_partition = timestamp[:: cfg.dataset.window_size]

    scale = 1080
    for row in pred_df.rows(named=True):
        start_idx = row["onset"] - scale
        end_idx = row["wakeup"] + scale
        pred_timestamp = timestamp[start_idx:end_idx]
        pred_anglez = anglez[start_idx:end_idx]
        pred_enmo = enmo[start_idx:end_idx]
        pred_dense_label = dense_label[start_idx:end_idx]
        pred_dense_preds = dense_preds[start_idx:end_idx]
        pred_dense_logits = dense_logits[start_idx:end_idx]
        window_mask = np.isin(pred_timestamp, window_partition)
        sub_window_part = pred_timestamp[window_mask]

        fig, ax = plt.subplots(4, 1)
        ax[0].plot(pred_timestamp, pred_anglez, label="AngleZ")
        ax[1].plot(pred_timestamp, pred_enmo, label="Enmo")
        ax[2].plot(
            pred_timestamp,
            pred_dense_label,
            label="Dense Label",
            alpha=1,
            c="black",
        )
        ax[2].plot(
            pred_timestamp,
            pred_dense_preds,
            label="Dense Preds",
            alpha=0.5,
            c="red",
            linestyle="--",
        )
        ax[2].vlines(
            x=sub_window_part,
            ymin=0,
            ymax=1,
            color="black",
            linestyle="--",
        )
        ax[3].plot(
            pred_timestamp,
            pred_dense_logits,
            label="Dense Logits",
            alpha=1,
            c="orange",
        )
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()
        # plotly_fig = tls.mpl_to_plotly(fig)  ## convert
        # iplot(plotly_fig, image_height=1080, image_width=1920)
        plt.show()


if __name__ == "__main__":
    main()
