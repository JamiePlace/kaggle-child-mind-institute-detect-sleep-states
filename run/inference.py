from pathlib import Path
import time

# from rich import print
import pickle
import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.conf import InferenceConfig, TrainConfig
from src.datamodule.prectime import (
    TestDataset,
)
from src.models.common import get_model
from src.utils.common import trace
from src.utils.post_process import post_process_for_prec


def load_model(
    cfg: InferenceConfig,
) -> nn.Module:
    model = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=2,
    )

    # load weights
    if "jamie" in cfg.dir.data_dir:
        weight_path = (
            Path(cfg.dir.model_dir)
            / cfg.weight.exp_name
            / cfg.weight.run_name
            / "best_model.pth"
        )
    else:
        weight_path = Path(cfg.dir.model_dir) / "best_model.pth"
    model.load_state_dict(torch.load(weight_path))
    print('load weight from "{}"'.format(weight_path))
    return model


def get_test_dataloader(cfg: InferenceConfig) -> DataLoader:
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    test_dataset = TestDataset(cfg)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def inference(
    cfg: InferenceConfig,
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
) -> tuple[list[str], list[np.ndarray]]:
    model = model.to(device)
    model.eval()

    keys = []
    preds = []
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            # with torch.cuda.amp.autocast(enabled=True):  # type: ignore
            x = batch["feature"].float().to(device)
            if len(x.shape) != 4:
                raise MemoryError("x.shape is not 4")
            else:
                x = x.squeeze(0)
            if x.shape[0] != cfg.dataset.batch_size:
                raise MemoryError(
                    f"batch size is not {cfg.dataset.batch_size}... {x.shape[0]}, key: {batch['key']}"
                )
            model_output = model(x)
            prediction = model_output["dense_predictions"]
            prediction = prediction.detach().cpu().numpy()
            if np.isnan(prediction).any():
                raise MemoryError(
                    f"nan-detected in predictions... key {batch['key']}"
                )
            key = batch["key"]
            preds.append(prediction)
            keys.append(key)

    return keys, preds  # type: ignore


def make_submission(
    cfg: InferenceConfig | TrainConfig,
    preds: dict[str, np.ndarray],
) -> pl.DataFrame:
    sub_df = post_process_for_prec(cfg, preds)
    sub_df = sub_df.drop_nulls()

    return sub_df


def make_predictions(key_list: list[str], pred_list: list[np.ndarray]):
    grouped_preds = {}
    with trace("grouping predctions"):
        for i, key_sub_list in enumerate(key_list):
            for j, key_id in enumerate(key_sub_list):
                key = key_id.split("_")[0]
                if len(pred_list) == 0:
                    continue
                if key not in grouped_preds.keys():
                    grouped_preds[key] = []
                if type(pred_list[i]) == torch.Tensor:
                    grouped_preds[key].append(
                        pred_list[i].detach().cpu().numpy()
                    )
                else:
                    grouped_preds[key].append(pred_list[i])
    for key in grouped_preds.keys():
        grouped_preds[key] = np.vstack(grouped_preds[key])
        grouped_preds[key] = np.reshape(
            grouped_preds[key],
            (grouped_preds[key].shape[0] * grouped_preds[key].shape[1], -1),
        )
    return grouped_preds


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: InferenceConfig):
    with trace("load test dataloader"):
        test_dataloader = get_test_dataloader(cfg)
    with trace("load model"):
        model = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    # time.sleep(120)
    with trace("inference"):
        key_list, pred_list = inference(cfg, test_dataloader, model, device)

    grouped_preds = make_predictions(key_list, pred_list)
    with trace("saving predictions"):
        with open(Path(cfg.dir.sub_dir) / "predictions.pkl", "wb") as f:
            pickle.dump(grouped_preds, f)

    with trace("make submission"):
        series_length = test_dataloader.dataset.series_length  # type: ignore
        sub_df = make_submission(cfg, grouped_preds)
    with trace(f"written to {Path(cfg.dir.sub_dir) / 'submission.csv'}"):
        dir = Path(cfg.dir.sub_dir) / "submission.csv"
        sub_df.write_csv(dir)  # type: ignore


if __name__ == "__main__":
    main()
