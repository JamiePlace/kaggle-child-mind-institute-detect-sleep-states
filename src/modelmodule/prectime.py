from typing import Any, Optional

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
)

from src.conf import TrainConfig
from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg
from src.models.prectimemodel import PrecTimeModel


class PrecTime(LightningModule):
    def __init__(
        self,
        cfg: TrainConfig,
        feature_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = PrecTimeModel(
            in_channels=feature_dim, n_classes=num_classes
        )
        self.__best_loss = np.inf
        self.validation_loss = []
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
        output = self.model(
            batch["feature"],
            batch["dense_label"],
        )
        loss: torch.Tensor = output["loss"]

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
            self.log(
                f"{mode}_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
            self.validation_loss.append(loss.detach().item())

        return loss

    def on_validation_epoch_end(self):
        # keys = []
        # for x in self.validation_step_outputs:
        #    keys.extend(x[0])
        # labels = np.concatenate([x[1] for x in self.validation_step_outputs])
        # preds = np.concatenate([x[2] for x in self.validation_step_outputs])
        # losses = np.array([x[3] for x in self.validation_step_outputs])
        loss = np.array(self.validation_loss).mean()

        # val_pred_df = post_process_for_seg(
        #    keys=keys,
        #    preds=preds[:, :, [1, 2]],
        #    score_th=self.cfg.pp.score_th,
        #    distance=self.cfg.pp.distance,
        # )

        if loss < self.__best_loss:
            # np.save("keys.npy", np.array(keys))
            #    np.save("labels.npy", labels)
            #    np.save("preds.npy", preds)
            #    val_pred_df.write_csv("val_pred_df.csv")
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
