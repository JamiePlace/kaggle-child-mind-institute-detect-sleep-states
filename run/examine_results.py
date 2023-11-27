import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import hydra

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


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: InferenceConfig):
    print(cfg.series_ids)
    # Load the data
    with open(Path(cfg.dir.sub_dir) / "predictions.pkl", "rb") as f:
        data = pickle.load(f)
    # Plot the data
    sparse_preds = data[cfg.series_ids[0]]
    sparse_label, dense_label, anglez, enmo = get_label(cfg)
    sparse_preds = expand_sparse_label(cfg, sparse_preds, dense_label)
    sparse_label = expand_sparse_label(cfg, sparse_label, dense_label)
    print(len(dense_label), len(sparse_preds), len(sparse_label))
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(anglez, label="AngleZ")
    ax[1].plot(enmo, label="Enmo")
    ax[2].plot(dense_label, label="Dense Label")
    ax[3].plot(sparse_preds, label="Predicted")
    ax[3].plot(sparse_label, label="Label")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.show()


if __name__ == "__main__":
    main()
