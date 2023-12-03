import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import hydra
import polars as pl
from rich import print

from src.utils.metrics import event_detection_ap
from src.conf import InferenceConfig


def get_label(cfg: InferenceConfig, series_id: str):
    with open(
        Path(cfg.dir.processed_dir) / "train" / "__valid_keys__.pkl", "rb"
    ) as f:
        valid_keys = pickle.load(f)

    with open(
        Path(cfg.dir.processed_dir) / "train" / "__train_keys__.pkl", "rb"
    ) as f:
        train_keys = pickle.load(f)

    all_keys = valid_keys + train_keys

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

    return sparse_label, dense_label, anglez, enmo


def expand_sparse_label(
    cfg: InferenceConfig,
    sparse_label: list | np.ndarray,
    dense_label: list | np.ndarray,
):
    expanded_label = np.zeros(len(dense_label))
    for i, label in enumerate(sparse_label):
        expanded_label[i * cfg.window_size : (i + 1) * cfg.window_size] = label
    expanded_label = expanded_label[: len(dense_label)]
    return expanded_label


# finish calculating the score
# def calculate_score(cfg: InferenceConfig, series_id: str | None = None):
#    pred_df = pl.read_csv(Path(cfg.dir.sub_dir) / "submission.csv")
#    if series_id:
#        pred_df = pred_df.filter(pl.col("series_id") == series_id)
#    event_df = (
#        pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv")
#        .drop_nulls()
#        .filter(pl.col("series_id").is_in(pred_df["series_id"].unique()))
#        .drop_nulls()
#    )
#    print(event_df)
#    print(pred_df)
#    if series_id:
#        return event_detection_ap(event_df.to_pandas(), pred_df.to_pandas())
#
#    train_score = event_detection_ap(
#        event_df.filter(
#            pl.col("series_id").is_in(cfg.split.train_series_ids)
#        ).to_pandas(),
#        pred_df.filter(
#            pl.col("series_id").is_in(cfg.split.train_series_ids)
#        ).to_pandas(),
#    )
#    valid_score = event_detection_ap(
#        event_df.filter(
#            pl.col("series_id").is_in(cfg.split.valid_series_ids)
#        ).to_pandas(),
#        pred_df.filter(
#            pl.col("series_id").is_in(cfg.split.valid_series_ids)
#        ).to_pandas(),
#    )
#    return train_score, valid_score


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: InferenceConfig):
    # Load the data
    with open(
        Path(cfg.dir.processed_dir) / "inference/__series_length__.pkl", "rb"
    ) as f:
        series_length = pickle.load(f)
    with open(Path(cfg.dir.sub_dir) / "predictions.pkl", "rb") as f:
        data = pickle.load(f)

    pred_df = pl.read_csv(Path(cfg.dir.sub_dir) / "submission.csv")
    # Plot the data
    if cfg.series_ids[0]:
        series_id = cfg.series_ids[0]
    else:
        series_id = np.random.choice(list(series_length.keys()))

    print(series_id)

    # train_score, valid_score = calculate_score(cfg)  # type: ignore
    # print(f"train score: {train_score:.4f}", f"valid score: {valid_score:.4f}")

    # score = calculate_score(cfg, series_id)
    # print(f"score: {score:.4f}")

    sparse_label, dense_label, anglez, enmo = get_label(cfg, series_id)
    this_series_length = len(anglez)
    # this_series_length = 15000

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
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(anglez, label="AngleZ")
    ax[1].plot(enmo, label="Enmo")
    ax[2].plot(dense_label, label="Dense Label", alpha=1, c="black")
    ax[2].plot(
        dense_preds, label="Dense Preds", alpha=0.5, c="red", linestyle="--"
    )
    ax[3].plot(dense_logits, label="Dense Logits", alpha=1, c="orange")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.show()


if __name__ == "__main__":
    main()
