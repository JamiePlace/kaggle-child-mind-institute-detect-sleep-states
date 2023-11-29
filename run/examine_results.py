import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import hydra
import polars as pl
from rich import print

from src.utils.metrics import event_detection_ap
from src.conf import InferenceConfig


def get_label(cfg: InferenceConfig):
    key = cfg.series_ids[0]
    with open(
        Path(cfg.dir.processed_dir) / "train" / "__valid_keys__.pkl", "rb"
    ) as f:
        valid_keys = pickle.load(f)

    files = [file + ".pkl" for file in valid_keys if file.split("_")[0] == key]
    sparse_label = []
    dense_label = []
    anglez = []
    enmo = []
    for file in files:
        with open(Path(cfg.dir.processed_dir) / "train" / file, "rb") as f:
            batch = pickle.load(f)
        sparse_label.append(batch["sparse_label"])
        dense_label.append(batch["dense_label"])
        anglez.append(batch["feature"][:, 0])
        enmo.append(batch["feature"][:, 0])
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
def calculate_score(cfg: InferenceConfig):
    pred_df = pl.read_csv(Path(cfg.dir.sub_dir) / "submission.csv")
    event_df = (
        pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv")
        .drop_nulls()
        .filter(pl.col("series_id").is_in(pred_df["series_id"].unique()))
        .drop_nulls()
    )
    pred_df = pred_df.filter(pl.col("step") <= event_df["step"].max())
    print(event_df)
    print(pred_df)
    score = event_detection_ap(event_df.to_pandas(), pred_df.to_pandas())
    return score


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: InferenceConfig):
    score = calculate_score(cfg)
    print(f"score: {score:.4f}")
    # print(cfg.series_ids)
    ## Load the data
    # with open(Path(cfg.dir.sub_dir) / "predictions.pkl", "rb") as f:
    # data = pickle.load(f)
    ## Plot the data
    # dense_preds = data[cfg.series_ids[0]]
    # sparse_label, dense_label, anglez, enmo = get_label(cfg)
    # print(len(dense_label), len(dense_preds))
    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(anglez, label="AngleZ")
    # ax[1].plot(enmo, label="Enmo")
    # ax[2].plot(dense_label, label="Dense Label")
    # ax[2].plot(dense_preds, label="Dense Preds")
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # plt.show()


if __name__ == "__main__":
    main()
