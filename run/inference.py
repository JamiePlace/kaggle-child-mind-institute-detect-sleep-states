from pathlib import Path

from rich import print
import pickle
import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from tqdm import tqdm

from src.conf import TrainConfig, InferenceConfig
from src.datamodule.prectime import (
    TestDataset,
    load_features,
)
from src.models.common import get_model
from src.utils.common import trace
from src.utils.post_process import post_process_for_seg


def load_model(
    cfg: InferenceConfig,
) -> nn.Module:
    model = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=2,
    )

    # load weights
    weight_path = (
        Path(cfg.dir.model_dir)
        / cfg.weight.exp_name
        / cfg.weight.run_name
        / "best_model.pth"
    )
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
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def inference(
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
            with torch.cuda.amp.autocast(enabled=True):  # type: ignore
                x = batch["feature"].half().to(device)
                model_output = model(x)
                prediction = model_output["predictions"]
                prediction = prediction.argmax(dim=1)
                key = batch["key"]
                preds.append(prediction.detach().cpu().numpy())
                keys.extend(key)

    return keys, preds  # type: ignore


def make_submission(
    keys: list[str], preds: np.ndarray, downsample_rate, score_th, distance
) -> pl.DataFrame:
    sub_df = post_process_for_seg(
        keys,
        preds[:, :, [1, 2]],  # type: ignore
        score_th=score_th,
        distance=distance,  # type: ignore
    )

    return sub_df


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: InferenceConfig):
    seed_everything(cfg.seed)
    with trace("load test dataloader"):
        test_dataloader = get_test_dataloader(cfg)
    with trace("load model"):
        model = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with trace("inference"):
        keys, preds = inference(test_dataloader, model, device)
    preds = np.concatenate(preds)
    # cut off preds of padded chunk

    grouped_preds = {}
    with trace("grouping predctions"):
        # for each key there will be 1 prediction
        # later there will be cfg.window_size predictions
        for i, key in enumerate(keys):
            key = key.split("_")[0]
            if key not in grouped_preds.keys():
                grouped_preds[key] = [preds[i]]
            else:
                grouped_preds[key].append(preds[i])
    with trace("resizing predictions"):
        for key in grouped_preds.keys():
            original_length = len(grouped_preds[key])
            grouped_preds[key] = grouped_preds[key][
                : test_dataloader.dataset.series_length[key]  # type: ignore
            ]
            print(
                f"key: {key}, "
                f"original length: {original_length}, "
                f"new length: {len(grouped_preds[key])}"
            )

    with trace("saving predictions"):
        with open(Path(cfg.dir.sub_dir) / "predictions.pkl", "wb") as f:
            pickle.dump(grouped_preds, f)

    # with trace("make submission"):
    # sub_df = make_submission(
    # keys,
    # preds,
    # downsample_rate=cfg.downsample_rate,
    # score_th=cfg.pp.score_th,
    # distance=cfg.pp.distance,
    # )
    # with trace(f"written to {Path(cfg.dir.sub_dir) / 'submission.csv'}"):
    # dir = Path(cfg.dir.sub_dir) / "submission.csv"
    # sub_df.write_csv(dir)  # type: ignore


if __name__ == "__main__":
    main()
