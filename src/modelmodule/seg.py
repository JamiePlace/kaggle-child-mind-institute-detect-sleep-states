from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchvision.transforms.functional import resize
from transformers import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from src.conf import TrainConfig
from src.datamodule.seg import nearest_valid_size
from src.models.common import get_model
from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg


class SegModel(LightningModule):
    def __init__(
        self,
        cfg: TrainConfig,
        val_event_df: pl.DataFrame,
        feature_dim: int,
        num_classes: int,
        duration: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.val_event_df = val_event_df
        num_timesteps = nearest_valid_size(
            int(duration * cfg.upsample_rate), cfg.downsample_rate
        )
        self.model = get_model(
            cfg,
            feature_dim=feature_dim,
            n_classes=num_classes,
            num_timesteps=num_timesteps // cfg.downsample_rate,
        )
        self.duration = duration
        self.validation_step_outputs: list = []
        self.__best_loss = np.inf
        self.val_loss_non_improvement = 0
        self.best_state_dict: dict = {}

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Optional[torch.Tensor]]:
        return self.model(x, labels)

    def training_step(self, batch, batch_idx):
        return self.__share_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.__share_step(batch, "val")

    def __share_step(self, batch, mode: str) -> torch.Tensor:
        do_mixup = False
        do_cutmix = False
        if mode == "train":
            do_mixup = np.random.rand() < self.cfg.aug.mixup_prob
            do_cutmix = np.random.rand() < self.cfg.aug.cutmix_prob
        elif mode == "val":
            do_mixup = False
            do_cutmix = False

        output = self.model(
            batch["feature"], batch["label"], do_mixup, do_cutmix
        )
        loss: torch.Tensor = output["loss"]
        logits = output["logits"]  # (batch_size, n_timesteps, n_classes)

        if mode == "train":
            self.log(
                f"{mode}_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        elif mode == "val":
            resized_logits = resize(
                logits.sigmoid().detach().cpu(),
                size=[self.duration, logits.shape[2]],
                antialias=False,
            )
            resized_labels = resize(
                batch["label"].detach().cpu(),
                size=[self.duration, logits.shape[2]],
                antialias=False,
            )
            self.validation_step_outputs.append(
                (
                    batch["key"],
                    resized_labels.numpy(),
                    resized_logits.numpy(),
                    loss.detach().item(),
                )
            )
            self.log(
                f"{mode}_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        return loss

    def on_validation_epoch_end(self):
        keys = []
        for x in self.validation_step_outputs:
            keys.extend(x[0])
        labels = np.concatenate([x[1] for x in self.validation_step_outputs])
        preds = np.concatenate([x[2] for x in self.validation_step_outputs])
        losses = np.array([x[3] for x in self.validation_step_outputs])
        loss = losses.mean()

        val_pred_df = post_process_for_seg(
            keys=keys,
            preds=preds[:, :, [1, 2]],
            score_th=self.cfg.pp.score_th,
            distance=self.cfg.pp.distance,
        )
        # this is slow as all holy hell
        # TODO make quicker loser
        # this is slow when we make too many positive predictions
        # shall I bother fixing this? probably not
        # a good model should not make too many positive predictions
        score = event_detection_ap(
            self.val_event_df.to_pandas(), val_pred_df.to_pandas()
        )
        self.log(
            "val_score",
            score,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        if loss < self.__best_loss:
            np.save("keys.npy", np.array(keys))
            np.save("labels.npy", labels)
            np.save("preds.npy", preds)
            val_pred_df.write_csv("val_pred_df.csv")
            torch.save(self.model.state_dict(), "best_model.pth")
            print(f"Saved best model {self.__best_loss} -> {loss}")
            self.__best_loss = loss
            self.val_loss_non_improvement = 0
        else:
            self.val_loss_non_improvement += 1

        if (
            self.val_loss_non_improvement
            > self.cfg.trainer.early_stopping_patience
        ):
            print("Loading Previously Saved Best Model")
            self.model.load_state_dict(torch.load("best_model.pth"))
            self.val_loss_non_improvement = 0

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.scheduler.num_warmup_steps,
            num_training_steps=self.trainer.max_steps,
            power=self.cfg.scheduler.power,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
